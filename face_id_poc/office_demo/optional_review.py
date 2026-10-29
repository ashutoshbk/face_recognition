"""Bounded, local-only review tools for explicitly consented office sessions.

Unknown snapshots are never embedded or enrolled here. Notices are review items,
not identity/security decisions, and cannot trigger external actions.
"""
from __future__ import annotations

import hashlib
import math
import queue
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path


REVIEW_DEFAULTS = dict(unknown_capture_enabled=False, flagged_review_enabled=False,
    review_consent=False, unknown_retention_hours=24., unknown_max_records=500,
    review_notice_cooldown=60., review_match_threshold=.68, review_match_margin=.10)
REVIEW_BOOLS = {"unknown_capture_enabled", "flagged_review_enabled", "review_consent"}
REVIEW_RANGES = dict(unknown_retention_hours=(1, 168), unknown_max_records=(1, 500),
    review_notice_cooldown=(10, 3600), review_match_threshold=(.68, 1), review_match_margin=(.10, .5))


def validate_review_settings(proposed):
    if (proposed["unknown_capture_enabled"] or proposed["flagged_review_enabled"]) and not proposed["review_consent"]:
        raise ValueError("Confirm a controlled session with consenting participants before enabling review tools")


class ReviewStore:
    MAX_BYTES = 250_000_000

    def __init__(self, root, clock=time.time):
        self.root = Path(root) / "review"
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.images = self.root / "images"
        self.images.mkdir(exist_ok=True, mode=0o700)
        self.path = self.root / "review.sqlite3"
        self.clock = clock
        self.lock = threading.RLock()
        with self.db() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS unknowns(
                  id TEXT PRIMARY KEY, source_key TEXT UNIQUE, camera_id TEXT, camera_name TEXT,
                  track_id TEXT, first_seen REAL, last_seen REAL, expires REAL);
                CREATE TABLE IF NOT EXISTS photos(
                  id TEXT PRIMARY KEY, owner TEXT, path TEXT, score REAL, width REAL,
                  digest TEXT, bytes INTEGER, created REAL);
                CREATE TABLE IF NOT EXISTS flags(
                  person_id TEXT PRIMARY KEY, reason TEXT, created REAL, expires REAL);
                CREATE TABLE IF NOT EXISTS notices(
                  id TEXT PRIMARY KEY, person_id TEXT, person_name TEXT, reason TEXT,
                  camera_id TEXT, camera_name TEXT, track_id TEXT, source_key TEXT,
                  created REAL, expires REAL, score REAL, margin REAL, votes INTEGER,
                  reference_id TEXT, status TEXT DEFAULT 'pending');
                CREATE TABLE IF NOT EXISTS cleanup(path TEXT PRIMARY KEY);
                CREATE TABLE IF NOT EXISTS audit(
                  id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL, action TEXT, subject TEXT);
            """)
        self.path.chmod(0o600)
        self.cleanup()

    def db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def audit(self, db, action, subject):
        db.execute("INSERT INTO audit(created,action,subject) VALUES(?,?,?)", (self.clock(), action, subject))
        db.execute("DELETE FROM audit WHERE id NOT IN (SELECT id FROM audit ORDER BY id DESC LIMIT 2000)")

    def _remove(self, db, owner, table):
        for row in db.execute("SELECT path FROM photos WHERE owner=?", (owner,)).fetchall():
            db.execute("INSERT OR IGNORE INTO cleanup VALUES(?)", (row["path"],))
        db.execute("DELETE FROM photos WHERE owner=?", (owner,))
        db.execute(f"DELETE FROM {table} WHERE id=?", (owner,))

    def cleanup(self, max_records=500):
        with self.lock, self.db() as db:
            now = self.clock()
            for table in ("unknowns", "notices"):
                for row in db.execute(f"SELECT id FROM {table} WHERE expires<=?", (now,)).fetchall():
                    self._remove(db, row["id"], table)
                excess = max(0, db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] - int(max_records))
                for row in db.execute(f"SELECT id FROM {table} ORDER BY created LIMIT ?" if table == "notices"
                                      else "SELECT id FROM unknowns ORDER BY first_seen LIMIT ?", (excess,)).fetchall():
                    self._remove(db, row["id"], table)
            db.execute("DELETE FROM flags WHERE expires IS NOT NULL AND expires<=?", (now,))
            while (db.execute("SELECT COALESCE(SUM(bytes),0) FROM photos").fetchone()[0] or 0) > self.MAX_BYTES:
                row = db.execute("SELECT id,'unknowns' AS kind,first_seen AS created FROM unknowns UNION ALL "
                                 "SELECT id,'notices' AS kind,created FROM notices ORDER BY created LIMIT 1").fetchone()
                if not row:
                    break
                self._remove(db, row["id"], row["kind"])
            db.commit()
            for row in db.execute("SELECT path FROM cleanup").fetchall():
                path = self.images / row["path"]
                try:
                    path.unlink(missing_ok=True)
                    db.execute("DELETE FROM cleanup WHERE path=?", (row["path"],))
                except OSError:
                    pass
            known = {row[0] for row in db.execute("SELECT path FROM photos")}
            for path in self.images.iterdir():
                if path.name not in known and self.clock() - path.stat().st_mtime > 60:
                    path.unlink(missing_ok=True)

    def _photo(self, db, owner, jpeg, quality):
        if not jpeg or len(jpeg) > 2_000_000:
            return False
        digest = hashlib.sha256(jpeg).hexdigest()
        rows = db.execute("SELECT * FROM photos WHERE owner=? ORDER BY score,created DESC", (owner,)).fetchall()
        if any(row["digest"] == digest for row in rows):
            return False
        if len(rows) >= 5 and quality["score"] <= rows[0]["score"]:
            return False
        identifier = uuid.uuid4().hex
        path = self.images / (identifier + ".jpg")
        temporary = path.with_suffix(".pending")
        try:
            temporary.write_bytes(jpeg)
            temporary.chmod(0o600)
            temporary.replace(path)
            db.execute("INSERT INTO photos VALUES(?,?,?,?,?,?,?,?)", (identifier, owner, path.name,
                       quality["score"], quality["width"], digest, len(jpeg), self.clock()))
            if len(rows) >= 5:
                db.execute("DELETE FROM photos WHERE id=?", (rows[0]["id"],))
                db.execute("INSERT OR IGNORE INTO cleanup VALUES(?)", (rows[0]["path"],))
        except Exception:
            temporary.unlink(missing_ok=True)
            path.unlink(missing_ok=True)
            raise
        return True

    def save_unknown(self, item, settings):
        if not item.get("image") or len(item["image"]) > 2_000_000:
            return False
        with self.lock, self.db() as db:
            row = db.execute("SELECT * FROM unknowns WHERE source_key=?", (item["source_key"],)).fetchone()
            if row is None:
                identifier = uuid.uuid4().hex
                db.execute("INSERT INTO unknowns VALUES(?,?,?,?,?,?,?,?)", (identifier, item["source_key"],
                           item["camera_id"], item["camera_name"], item["track_id"], self.clock(), self.clock(),
                           self.clock()+settings["unknown_retention_hours"]*3600))
                self.audit(db, "unknown_created", identifier)
            else:
                identifier = row["id"]
                db.execute("UPDATE unknowns SET last_seen=? WHERE id=?", (self.clock(), identifier))
            saved = self._photo(db, identifier, item["image"], item["quality"])
        self.cleanup(settings["unknown_max_records"])
        return saved

    def set_flag(self, person_id, reason, expires_hours=None):
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 200 or any(ord(c) < 32 for c in reason):
            raise ValueError("Enter a printable review reason of 1-200 characters")
        if expires_hours is not None and (isinstance(expires_hours, bool) or not isinstance(expires_hours, (float, int))
                                         or not math.isfinite(expires_hours) or not 1 <= expires_hours <= 168):
            raise ValueError("Expiry must be 1-168 hours, or no expiry")
        with self.lock, self.db() as db:
            db.execute("INSERT OR REPLACE INTO flags VALUES(?,?,?,?)", (person_id, reason.strip(), self.clock(),
                       self.clock()+expires_hours*3600 if expires_hours is not None else None))
            self.audit(db, "flag_set", person_id)

    def remove_flag(self, person_id):
        with self.lock, self.db() as db:
            db.execute("DELETE FROM flags WHERE person_id=?", (person_id,))
            self.audit(db, "flag_removed", person_id)

    def flags(self):
        with self.lock, self.db() as db:
            return [dict(r) for r in db.execute("SELECT * FROM flags WHERE expires IS NULL OR expires>?", (self.clock(),))]

    def remove_person(self, person_id):
        with self.lock, self.db() as db:
            db.execute("DELETE FROM flags WHERE person_id=?", (person_id,))
            for row in db.execute("SELECT id FROM notices WHERE person_id=?", (person_id,)).fetchall():
                self._remove(db, row["id"], "notices")
            self.audit(db, "person_review_removed", person_id)
        self.cleanup()

    def remove_reference(self, reference_id):
        with self.lock, self.db() as db:
            for row in db.execute("SELECT id FROM notices WHERE reference_id=?", (reference_id,)).fetchall():
                self._remove(db, row["id"], "notices")
        self.cleanup()

    def save_notice(self, item, person, flag, settings, votes):
        if not item.get("image") or len(item["image"]) > 2_000_000:
            return False
        with self.lock, self.db() as db:
            prior = db.execute("SELECT created FROM notices WHERE person_id=? AND camera_id=? ORDER BY created DESC LIMIT 1",
                               (person["id"], item["camera_id"])).fetchone()
            if prior and self.clock()-prior["created"] < settings["review_notice_cooldown"]:
                return False
            if db.execute("SELECT 1 FROM notices WHERE source_key=? AND person_id=?", (item["source_key"], person["id"])).fetchone():
                return False
            identifier = uuid.uuid4().hex
            refs = person["registered"]
            reference = max(refs, key=lambda r: r["score"])["id"]
            db.execute("INSERT INTO notices VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", (identifier, person["id"],
                       person["name"], flag["reason"], item["camera_id"], item["camera_name"], item["track_id"],
                       item["source_key"], self.clock(), self.clock()+settings["unknown_retention_hours"]*3600,
                       item["score"], item["margin"], votes, reference, "pending"))
            self._photo(db, identifier, item["image"], item["quality"])
            self.audit(db, "notice_created", identifier)
        self.cleanup()
        return True

    def list(self, kind, offset=0, limit=50):
        if kind not in ("unknowns", "notices"):
            raise ValueError("Unknown review collection")
        self.cleanup()
        with self.lock, self.db() as db:
            order = "first_seen" if kind == "unknowns" else "created"
            rows = [dict(r) for r in db.execute(f"SELECT * FROM {kind} ORDER BY {order} DESC LIMIT ? OFFSET ?",
                                               (limit, offset))]
            for row in rows:
                row.pop("source_key", None)
                row["photos"] = [dict(id=p["id"], score=p["score"], width=p["width"],
                    created=p["created"], url=f'/api/review/photos/{p["id"]}')
                    for p in db.execute("SELECT * FROM photos WHERE owner=? ORDER BY score DESC", (row["id"],))]
                if kind == "notices":
                    row["reference_url"] = "/api/photo/" + row.pop("reference_id")
            return dict(items=rows, total=db.execute(f"SELECT COUNT(*) FROM {kind}").fetchone()[0], offset=offset, limit=limit)

    def photo(self, identifier):
        with self.lock, self.db() as db:
            row = db.execute("SELECT * FROM photos WHERE id=?", (identifier,)).fetchone()
            if not row:
                raise ValueError("Review photo was removed or expired")
            active = db.execute("SELECT 1 FROM unknowns WHERE id=? AND expires>? UNION ALL "
                                "SELECT 1 FROM notices WHERE id=? AND expires>?", (row["owner"], self.clock(), row["owner"], self.clock())).fetchone()
            if not active:
                raise ValueError("Review photo expired")
            return (self.images / row["path"]).read_bytes()

    def delete(self, kind, identifier):
        if kind not in ("unknowns", "notices"):
            raise ValueError("Unknown review collection")
        with self.lock, self.db() as db:
            self._remove(db, identifier, kind)
            self.audit(db, kind+"_deleted", identifier)
        self.cleanup()

    def acknowledge(self, identifier):
        with self.lock, self.db() as db:
            db.execute("UPDATE notices SET status='reviewed' WHERE id=? AND expires>?", (identifier, self.clock()))
            self.audit(db, "notice_reviewed", identifier)

    def counts(self):
        with self.lock, self.db() as db:
            now = self.clock()
            return dict(unknowns=db.execute("SELECT COUNT(*) FROM unknowns WHERE expires>?", (now,)).fetchone()[0],
                        notices=db.execute("SELECT COUNT(*) FROM notices WHERE expires>?", (now,)).fetchone()[0],
                        pending_notices=db.execute("SELECT COUNT(*) FROM notices WHERE status='pending' AND expires>?", (now,)).fetchone()[0],
                        storage_bytes=db.execute("SELECT COALESCE(SUM(bytes),0) FROM photos").fetchone()[0])


class ReviewService:
    def __init__(self, root, gallery, settings, current=None):
        self.store = ReviewStore(root)
        self.gallery = gallery
        self.settings = dict(settings)
        self.queue = queue.Queue(maxsize=64)
        self.lock = threading.RLock()
        self.epoch = 0
        self.streaks = OrderedDict()
        self.stop_event = threading.Event()
        self.thread = None
        self.metrics = dict(enqueued=0, queue_dropped=0, stale_dropped=0, quality_rejected=0,
                            saved_unknown=0, notices_created=0, errors=0)
        self.error = None
        self.current = current or (lambda item: True)

    def configure(self, settings):
        with self.lock:
            retention_changed = self.settings["unknown_retention_hours"] != settings["unknown_retention_hours"]
            limit_changed = self.settings["unknown_max_records"] != settings["unknown_max_records"]
            self.settings = dict(settings)
            self.epoch += 1
            self.streaks.clear()
            while True:
                try:
                    self.queue.get_nowait()
                except queue.Empty:
                    break
            if retention_changed:
                with self.store.lock, self.store.db() as db:
                    duration = settings["unknown_retention_hours"]*3600
                    db.execute("UPDATE unknowns SET expires=MIN(expires,first_seen+?)", (duration,))
                    db.execute("UPDATE notices SET expires=MIN(expires,created+?)", (duration,))
            if retention_changed or limit_changed:
                self.store.cleanup(settings["unknown_max_records"])

    def start(self):
        self.thread = threading.Thread(target=self._worker, name="office-private-review", daemon=True)
        self.thread.start()

    def submit(self, item):
        if not self.lock.acquire(blocking=False):
            self.metrics["queue_dropped"] += 1
            return
        try:
            if not self.settings["review_consent"] or not (self.settings["unknown_capture_enabled"] or self.settings["flagged_review_enabled"]):
                return
            item = dict(item, review_epoch=self.epoch)
            try:
                self.queue.put_nowait(item)
                self.metrics["enqueued"] += 1
            except queue.Full:
                self.metrics["queue_dropped"] += 1
        finally:
            self.lock.release()

    def process(self, item):
        with self.lock:
            settings = self.settings
            if item["review_epoch"] != self.epoch or not 0 <= time.monotonic()-item["captured"] <= 2 or not self.current(item):
                self.metrics["stale_dropped"] += 1
                return
            if not settings["review_consent"]:
                return
            quality = item["quality"]
            if (not quality.get("accepted") or quality.get("score", 0) < 50 or quality.get("width", 0) < 70
                    or not math.isfinite(quality.get("score", 0)) or not math.isfinite(quality.get("width", 0))):
                self.metrics["quality_rejected"] += 1
                self.streaks.pop(item["source_key"], None)
                return
            category = "unknown" if item["unknown"] else item.get("person_id")
            key = item["source_key"]
            old = self.streaks.get(key)
            if not old or old["category"] != category or item["captured"]-old["last"] > 3:
                old = dict(category=category, count=0, first=item["captured"], last=-1e9, sequence=None)
            if item["sequence"] == old["sequence"] or item["captured"]-old["last"] < .4:
                return
            old.update(count=old["count"]+1, last=item["captured"], sequence=item["sequence"])
            self.streaks[key] = old
            self.streaks.move_to_end(key)
            while len(self.streaks) > 500:
                self.streaks.popitem(last=False)
            if item["unknown"] and settings["unknown_capture_enabled"]:
                if old["count"] >= 3 and item["captured"]-old["first"] >= 1:
                    self.metrics["saved_unknown"] += int(self.store.save_unknown(item, settings))
            elif settings["flagged_review_enabled"] and item.get("strict_match"):
                person = next((p for p in self.gallery.people() if p["id"] == item.get("person_id")), None)
                flag = next((f for f in self.store.flags() if f["person_id"] == item.get("person_id")), None)
                required = 5 if person and len(person["registered"]) == 1 else 3
                if person and flag and old["count"] >= required and item["captured"]-old["first"] >= (3 if required==5 else 1):
                    self.metrics["notices_created"] += int(self.store.save_notice(item, person, flag, settings, old["count"]))
            else:
                self.streaks.pop(key, None)

    def _worker(self):
        while not self.stop_event.is_set():
            try:
                item = self.queue.get(timeout=1)
                self.process(item)
                self.error = None
            except queue.Empty:
                pass
            except Exception as error:
                self.metrics["errors"] += 1
                self.error = type(error).__name__ + ": review item could not be saved"
            try:
                self.store.cleanup(self.settings["unknown_max_records"])
            except Exception:
                self.error = "Review retention cleanup needs attention"

    def snapshot(self):
        return dict(**self.store.counts(), flags=self.store.flags(), metrics=dict(self.metrics),
                    queue_size=self.queue.qsize(), error=self.error, max_storage_bytes=self.store.MAX_BYTES)

    def close(self):
        self.stop_event.set()
        self.configure(self.settings)
        if self.thread:
            self.thread.join(timeout=3)
