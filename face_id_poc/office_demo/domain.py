"""Local reference storage and registered-only admission policy."""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import threading
import time
import unicodedata
import uuid
from collections import Counter, deque
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np


def unit(value):
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    if vector.size != 512 or not np.isfinite(vector).all():
        raise ValueError("Expected a finite 512-dimensional embedding")
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        raise ValueError("Empty face embedding")
    return vector / norm


def safe_name(value):
    value = unicodedata.normalize("NFKC", str(value)).strip()
    if not value or len(value) > 80 or any(ord(c) < 32 for c in value):
        raise ValueError("Enter a name of 1-80 printable characters")
    if value in (".", "..") or re.search(r'[/\\:*?"<>|]', value):
        raise ValueError("Name contains a character not allowed in a folder name")
    value = value.rstrip(". ")
    if not value or value.upper().split(".")[0] in {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}:
        raise ValueError("Choose a portable folder name")
    return value


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.chmod(0o600)
    temporary.replace(path)


class QdrantIndex:
    def __init__(self, url, collection="camex_office_refs_v1"):
        from qdrant_client import QdrantClient
        target = urlsplit(url)
        default_grpc = 16344 if target.hostname in ("localhost", "127.0.0.1") and target.port == 16343 else 6334
        grpc_port = int(os.environ.get("CAMEX_OFFICE_QDRANT_GRPC_PORT", default_grpc))
        self.client = QdrantClient(url=url, grpc_port=grpc_port, prefer_grpc=True, timeout=10)
        self.collection = collection

    def replace(self, references):
        from qdrant_client import models
        if not self.client.collection_exists(self.collection):
            self.client.create_collection(self.collection, vectors_config=models.VectorParams(
                size=512, distance=models.Distance.COSINE))
        if references:
            self.client.upsert(self.collection, points=[models.PointStruct(
                id=r["id"], vector=r["vector"].tolist(),
                payload={k: r[k] for k in ("person_id", "pool", "id")})
                for r in references], wait=True)
            remove = models.Filter(must_not=[models.HasIdCondition(has_id=[r["id"] for r in references])])
        else:
            remove = models.Filter()
        self.client.delete(self.collection, points_selector=models.FilterSelector(filter=remove), wait=True)

    def scores(self, vector, references):
        from qdrant_client import models
        if not references:
            return {}
        result = self.client.query_points(self.collection, query=unit(vector).tolist(),
            limit=len(references), with_payload=False, with_vectors=False,
            search_params=models.SearchParams(exact=True)).points
        allowed = {r["id"] for r in references}
        return {str(p.id): float(p.score) for p in result if str(p.id) in allowed}


class Gallery:
    """SQLite is authoritative; failed index updates cannot publish new references."""
    def __init__(self, root, index):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.db_path = self.root / "database" / "office.sqlite3"
        self.db_path.parent.mkdir(exist_ok=True, mode=0o700)
        self.index = index
        self.lock = threading.RLock()
        self.healthy = False
        self.error = "Index not initialized"
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS people(
                  id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE, anchor_version INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS refs(
                  id TEXT PRIMARY KEY, person_id TEXT NOT NULL, pool TEXT NOT NULL,
                  score REAL NOT NULL, vector BLOB NOT NULL, path TEXT NOT NULL,
                  digest TEXT NOT NULL, metadata TEXT NOT NULL, anchor_version INTEGER NOT NULL,
                  created REAL NOT NULL);
                CREATE TABLE IF NOT EXISTS audit(
                  id INTEGER PRIMARY KEY AUTOINCREMENT, timestamp REAL, action TEXT, details TEXT);
                CREATE TABLE IF NOT EXISTS state(key TEXT PRIMARY KEY, value INTEGER NOT NULL);
                INSERT OR IGNORE INTO state VALUES('registered_revision',0);
                CREATE TABLE IF NOT EXISTS cleanup(path TEXT PRIMARY KEY);
            """)
        self.db_path.chmod(0o600)
        self.cleanup_pending()

    def db(self):
        return sqlite3.connect(self.db_path, timeout=10)

    def _audit(self, db, action, details):
        db.execute("INSERT INTO audit(timestamp,action,details) VALUES(?,?,?)",
                   (time.time(), action, json.dumps(details, allow_nan=False)))

    def cleanup_pending(self):
        with self.lock, self.db() as db:
            for (path,) in db.execute("SELECT path FROM cleanup").fetchall():
                try:
                    destination = Path(path)
                    if not destination.is_relative_to(self.root):
                        continue
                    destination.unlink(missing_ok=True)
                    db.execute("DELETE FROM cleanup WHERE path=?", (path,))
                except OSError:
                    pass

    def registered_revision(self, db=None):
        if db is None:
            with self.db() as connection:
                return self.registered_revision(connection)
        return db.execute("SELECT value FROM state WHERE key='registered_revision'").fetchone()[0]

    def pending_cleanup_count(self):
        with self.db() as db:
            return db.execute("SELECT COUNT(*) FROM cleanup").fetchone()[0]

    def create_person(self, name):
        name = safe_name(name)
        with self.lock, self.db() as db:
            row = db.execute("SELECT id FROM people WHERE name=?", (name,)).fetchone()
            if row:
                return row[0]
            identifier = str(uuid.uuid4())
            db.execute("INSERT INTO people(id,name) VALUES(?,?)", (identifier, name))
            self._audit(db, "create_person", {"person_id": identifier})
            return identifier

    def references(self, active_only=True, db=None):
        if db is None:
            with self.db() as connection:
                return self.references(active_only, connection)
        rows = db.execute("""SELECT r.*,p.name,p.anchor_version FROM refs r
                            JOIN people p ON p.id=r.person_id ORDER BY r.created""").fetchall()
        result = []
        for row in rows:
            active = row[2] == "registered" or row[8] == row[11]
            if active_only and not active:
                continue
            result.append(dict(id=row[0], person_id=row[1], pool=row[2], score=row[3],
                vector=np.frombuffer(row[4], np.float32).copy(), path=row[5], digest=row[6],
                metadata=json.loads(row[7]), anchor_version=row[8], created=row[9],
                name=row[10], person_version=row[11], active=active))
        return result

    def people(self):
        with self.lock, self.db() as db:
            people = [dict(id=r[0], name=r[1], anchor_version=r[2], registered=[], live=[])
                      for r in db.execute("SELECT * FROM people ORDER BY name")]
            by_id = {p["id"]: p for p in people}
            for ref in self.references(False, db):
                item = {k: ref[k] for k in ("id", "score", "created", "active")}
                item.update(ref["metadata"])
                item.update(url=f'/api/photo/{ref["id"]}', source=ref["metadata"].get("source", "upload"))
                by_id[ref["person_id"]][ref["pool"]].append(item)
            for p in people:
                for pool in ("registered", "live"):
                    p[pool].sort(key=lambda r: (-r["score"], r["id"]))
            return people

    def sync(self):
        with self.lock:
            try:
                self.index.replace(self.references())
                self.healthy, self.error = True, None
            except Exception:
                self.healthy, self.error = False, "Qdrant index unavailable; recognition is paused"
                raise RuntimeError(self.error)

    def _manifest(self, person_id):
        people = [p for p in self.people() if p["id"] == person_id]
        if people:
            person = people[0]
            folder = self.root / "Face_ID" / "people" / person_id / person["name"]
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            write_json(folder / "metadata.json", person)

    def put(self, person_id, pool, image, vector, quality, metadata=None, expected_anchor_version=None,
            expected_registered_revision=None):
        if pool not in ("registered", "live"):
            raise ValueError("Invalid reference collection")
        if not quality.get("accepted"):
            return {"accepted": False, "reason": "Photo failed minimum quality checks"}
        score = float(quality["score"])
        if not math.isfinite(score):
            raise ValueError("Invalid quality score")
        vector = unit(vector)
        digest = hashlib.sha256(image).hexdigest()
        with self.lock, self.db() as db:
            person = db.execute("SELECT name,anchor_version FROM people WHERE id=?", (person_id,)).fetchone()
            if not person:
                raise ValueError("Person not found")
            if expected_anchor_version is not None and person[1] != expected_anchor_version:
                raise ValueError("Registered references changed; candidate must be checked again")
            if expected_registered_revision is not None and self.registered_revision(db) != expected_registered_revision:
                raise ValueError("Registered gallery changed; candidate margin must be checked again")
            old = self.references(False, db)
            own = [r for r in old if r["person_id"] == person_id and r["pool"] == pool]
            active = [r for r in own if r["active"]]
            if any(r["digest"] == digest or float(r["vector"] @ vector) > .9985 for r in active):
                return {"accepted": False, "reason": "Duplicate or near-identical reference"}
            inactive = [r for r in own if not r["active"]]
            loser = min(inactive or own, key=lambda r: (r["score"], -r["created"])) if len(own) >= 5 else None
            if loser and loser["active"] and score <= loser["score"]:
                return {"accepted": False, "reason": "Five better or equal references already stored"}
            identifier = str(uuid.uuid4())
            folder = self.root / "Face_ID" / "people" / person_id / person[0] / pool
            folder.mkdir(parents=True, exist_ok=True, mode=0o700)
            path = folder / (identifier + ".jpg")
            temp = path.with_suffix(".pending")
            temp.write_bytes(image)
            temp.chmod(0o600)
            version = person[1] + (pool == "registered")
            details = dict(metadata or {}, quality=quality, width=quality.get("width"),
                           scoring_version="office-quality-v1")
            try:
                if pool == "registered":
                    db.execute("UPDATE people SET anchor_version=? WHERE id=?", (version, person_id))
                    db.execute("UPDATE state SET value=value+1 WHERE key='registered_revision'")
                if loser:
                    db.execute("DELETE FROM refs WHERE id=?", (loser["id"],))
                    db.execute("INSERT OR IGNORE INTO cleanup VALUES(?)", (loser["path"],))
                db.execute("INSERT INTO refs VALUES(?,?,?,?,?,?,?,?,?,?)", (
                    identifier, person_id, pool, score, vector.tobytes(), str(path), digest,
                    json.dumps(details, allow_nan=False), version, time.time()))
                self.index.replace(self.references(True, db))
                temp.replace(path)
                self._audit(db, "reference_added", {"person_id": person_id, "pool": pool,
                    "reference_id": identifier, "replaced": loser["id"] if loser else None})
                db.commit()
                self.healthy, self.error = True, None
            except Exception:
                db.rollback()
                temp.unlink(missing_ok=True)
                path.unlink(missing_ok=True)
                try:
                    self.index.replace([r for r in old if r["active"]])
                except Exception:
                    self.healthy, self.error = False, "Index recovery required; recognition is paused"
                raise
            self.cleanup_pending()
            self._manifest(person_id)
            return {"accepted": True, "id": identifier, "replaced": loser["id"] if loser else None}

    def photo_path(self, identifier):
        with self.db() as db:
            row = db.execute("SELECT path FROM refs WHERE id=?", (identifier,)).fetchone()
        if not row or not Path(row[0]).is_file():
            raise ValueError("Photo not found")
        return Path(row[0])

    def delete_photo(self, identifier):
        with self.lock, self.db() as db:
            old = self.references(False, db)
            ref = next((r for r in old if r["id"] == identifier), None)
            if not ref:
                raise ValueError("Photo not found")
            db.execute("DELETE FROM refs WHERE id=?", (identifier,))
            db.execute("INSERT OR IGNORE INTO cleanup VALUES(?)", (ref["path"],))
            if ref["pool"] == "registered":
                db.execute("UPDATE people SET anchor_version=anchor_version+1 WHERE id=?", (ref["person_id"],))
                db.execute("UPDATE state SET value=value+1 WHERE key='registered_revision'")
            try:
                self.index.replace(self.references(True, db))
                self._audit(db, "reference_deleted", {"reference_id": identifier})
                db.commit()
            except Exception:
                db.rollback()
                self.healthy = False
                raise
            self.cleanup_pending()
            self._manifest(ref["person_id"])

    def delete_person(self, identifier):
        with self.lock, self.db() as db:
            old = self.references(False, db)
            removed = [r for r in old if r["person_id"] == identifier]
            paths = [r["path"] for r in removed] + [str(p) for p in
                (self.root / "Face_ID" / "people" / identifier).glob("*/metadata.json")]
            db.executemany("INSERT OR IGNORE INTO cleanup VALUES(?)", [(p,) for p in paths])
            db.execute("DELETE FROM refs WHERE person_id=?", (identifier,))
            db.execute("DELETE FROM people WHERE id=?", (identifier,))
            if any(r["pool"] == "registered" for r in removed):
                db.execute("UPDATE state SET value=value+1 WHERE key='registered_revision'")
            try:
                self.index.replace(self.references(True, db))
                self._audit(db, "person_deleted", {"person_id": identifier})
                db.commit()
            except Exception:
                db.rollback()
                self.healthy = False
                raise
            self.cleanup_pending()

    def search(self, vector):
        with self.lock:
            if not self.healthy:
                raise RuntimeError(self.error or "Index unavailable")
            refs = self.references()
            registered_revision = self.registered_revision()
            if not refs:
                return []
            try:
                scores = self.index.scores(unit(vector), refs)
            except Exception:
                self.healthy, self.error = False, "Qdrant query failed; recognition is paused"
                raise RuntimeError(self.error)
            groups = {}
            for ref in refs:
                if ref["id"] not in scores:
                    continue
                p = groups.setdefault(ref["person_id"], dict(person_id=ref["person_id"], name=ref["name"],
                    anchor_version=ref["person_version"], registered_revision=registered_revision,
                    registered=[], live=[], reference_scores=[]))
                p[ref["pool"]].append(scores[ref["id"]])
                p["reference_scores"].append(dict(id=ref["id"], pool=ref["pool"], score=scores[ref["id"]]))
            for p in groups.values():
                p["score"] = max(p["registered"] + p["live"])
                p["registered_score"] = max(p["registered"], default=-1.0)
            return sorted(groups.values(), key=lambda p: -p["score"])


def registered_gate(ranked, person_id, threshold, margin):
    target = next((p for p in ranked if p["person_id"] == person_id), None)
    if not target:
        return False, "Registered identity not present", {}
    count = len(target["registered"])
    required = {2: 2, 3: 3, 4: 3, 5: 4}.get(count)
    if required is None:
        return False, "At least two trusted registered references required", {"count": count}
    agree = sum(s >= threshold for s in target["registered"])
    competitor = max((p["registered_score"] for p in ranked if p["person_id"] != person_id), default=-1.0)
    gap = target["registered_score"] - competitor
    details = dict(agree=agree, required=required, count=count, margin=gap,
                   registered_score=target["registered_score"], threshold=threshold,
                   matching_reference_ids=[r["id"] for r in target.get("reference_scores", [])
                       if r["pool"] == "registered" and r["score"] >= threshold])
    if agree < required:
        return False, "Insufficient registered-only agreement", details
    if gap < margin:
        return False, "Competing registered identity too close", details
    return True, "Registered-only gate passed", details


class TrackEvidence:
    def __init__(self):
        self.observations = deque(maxlen=5)
        self.last_observation = -1e9

    def clear(self):
        self.observations.clear()
        self.last_observation = -1e9

    def add(self, ranked, now, frame_id, settings):
        if now - self.last_observation < .4:
            return self.summary(now)
        if any(x["frame_id"] == frame_id for x in self.observations):
            return self.summary(now)
        while self.observations and now - self.observations[0]["time"] > 8:
            self.observations.popleft()
        best = ranked[0] if ranked else None
        gap = best["score"] - ranked[1]["score"] if best and len(ranked) > 1 else 2.0
        accepted = bool(best and best["score"] >= settings["match_threshold"] and gap >= settings["match_margin"])
        person_id = best["person_id"] if accepted else None
        gate, reason, details = registered_gate(ranked, person_id, settings["admission_threshold"],
                                                settings["admission_margin"]) if person_id else (False, "No qualifying identity", {})
        self.observations.append(dict(time=now, frame_id=frame_id, person_id=person_id,
            name=best["name"] if accepted else None, score=best["score"] if best else None,
            strict=gate, strict_reason=reason, strict_details=details))
        self.last_observation = now
        return self.summary(now)

    def summary(self, now):
        values = [x for x in self.observations if now - x["time"] <= 8]
        if not values:
            return dict(state="Unknown", name=None, person_id=None, votes=0, observations=0, admission=False)
        counts = Counter(x["person_id"] for x in values if x["person_id"])
        ordered = counts.most_common()
        latest = values[-1]
        leader = ordered[0][0] if ordered and (len(ordered) == 1 or ordered[0][1] > ordered[1][1]) else None
        supported = leader is not None and latest["person_id"] == leader
        votes = counts.get(leader, 0) if supported else 0
        same = [x for x in values if x["person_id"] == leader] if supported else []
        confirmed = votes >= 3 and same[-1]["time"] - same[0]["time"] >= 1.0
        strict = []
        if supported:
            for item in reversed(values):
                if item["person_id"] != leader or not item["strict"]:
                    break
                strict.append(item)
            strict.reverse()
        admission = bool(confirmed and latest["strict"] and len(strict) >= 3 and
                         strict[-1]["time"] - strict[0]["time"] >= 1.0)
        return dict(state="Confirmed" if confirmed else "Tentative" if supported else "Unknown",
            name=latest["name"] if supported else None, person_id=leader if supported else None,
            score=latest["score"], votes=votes, observations=len(values), admission=admission,
            admission_reason=latest["strict_reason"], admission_details=latest["strict_details"])
