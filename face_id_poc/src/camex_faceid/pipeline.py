from collections import deque
from time import perf_counter
from numbers import Integral

import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

from .identity import IdentityState
from .models import nms, quality
from .tracking import HeadTracker
from .head_evidence import update_head_evidence


def body_support(box, bodies):
    box = np.asarray(box, float)
    center = (box[:2] + box[2:]) / 2
    area = np.prod(np.maximum(0, box[2:] - box[:2]))
    for body in bodies:
        b = body[:4]
        overlap = np.prod(
            np.maximum(0, np.minimum(box[2:], b[2:]) - np.maximum(box[:2], b[:2]))
        )
        if overlap / max(area, 1) > 0.5 and center[1] < b[1] + 0.65 * (b[3] - b[1]):
            return True
    return False


def supported_heads(heads, bodies, faces, require_body=True, trusted_heads=()):
    def supported(box):
        if not require_body or body_support(box, bodies):
            return True
        box = np.asarray(box, float)
        for old in trusted_heads:
            old = np.asarray(old, float)
            small = np.maximum(np.minimum(box[2:] - box[:2], old[2:] - old[:2]), 1)
            ratio = np.maximum(box[2:] - box[:2], old[2:] - old[:2]) / small
            delta = ((box[:2] + box[2:]) - (old[:2] + old[2:])) / 2
            if max(ratio) <= 2.5 and np.linalg.norm(delta / small) <= 0.5:
                return True
        return False

    heads = [h.copy() for h in heads if supported(h[:4])]
    for face in faces:
        box = face["bbox"]
        if not supported(box):
            continue
        matches = []
        for i, h in enumerate(heads):
            overlap = np.prod(
                np.maximum(0, np.minimum(h[2:4], box[2:]) - np.maximum(h[:2], box[:2]))
            )
            matches.append((overlap / max(np.prod(box[2:] - box[:2]), 1), i))
        if matches and max(matches)[0] > 0.5:
            i = max(matches)[1]
            heads[i][4] = max(heads[i][4], face["score"])
        else:
            size = box[2:] - box[:2]
            expanded = np.r_[
                box[:2] - size * [0.15, 0.35],
                box[2:] + size * [0.15, 0.1],
                face["score"],
            ]
            heads.append(expanded)
    if not heads:
        return np.empty((0, 5), np.float32)
    heads = np.asarray(heads, np.float32)
    return heads[nms(heads[:, :4], heads[:, 4], 0.4)]


def association(heads, faces, return_ambiguous=False):
    """Exclusive containment assignment, with ambiguity rejection."""
    if not heads or not faces:
        return ({}, set()) if return_ambiguous else {}
    scores = np.zeros((len(heads), len(faces)))
    for i, h in enumerate(heads):
        hb = np.array(h["bbox"], float)
        pad = (hb[2:] - hb[:2]) * 0.15
        hb = np.r_[hb[:2] - pad, hb[2:] + pad]
        for j, f in enumerate(faces):
            fb = f["bbox"]
            area = np.prod(np.maximum(0, fb[2:] - fb[:2]))
            overlap = np.prod(
                np.maximum(0, np.minimum(hb[2:], fb[2:]) - np.maximum(hb[:2], fb[:2]))
            )
            distance = np.linalg.norm((fb[:2] + fb[2:] - hb[:2] - hb[2:]) / 2) / max(
                np.linalg.norm(hb[2:] - hb[:2]), 1
            )
            scores[i, j] = overlap / max(area, 1e-6) - 0.25 * distance
    rows, cols = linear_sum_assignment(-scores)
    matches = {}
    ambiguous_heads = set()
    for i, j in zip(rows, cols):
        rivals = np.delete(scores[:, j], i)
        ambiguous = len(rivals) and scores[i, j] - max(rivals) < 0.08
        other_faces = np.delete(scores[i], j)
        ambiguous = ambiguous or (
            len(other_faces) and scores[i, j] - max(other_faces) < 0.08
        )
        if scores[i, j] >= 0.55 and not ambiguous:
            matches[int(i)] = int(j)
        elif scores[i, j] >= 0.55 and ambiguous:
            ambiguous_heads.add(int(i))
            ambiguous_heads.update(
                int(k)
                for k in range(len(heads))
                if scores[k, j] >= 0.55 and abs(scores[i, j] - scores[k, j]) < 0.08
            )
    return (matches, ambiguous_heads) if return_ambiguous else matches


class FaceIDService:
    """One instance per stream. The client owns decoding, storage and threading."""

    def __init__(self, networks, gallery, config, camera_id="camera_01"):
        self.net, self.gallery, self.cfg = networks, gallery, config
        self.camera_id = camera_id
        self.tracker = HeadTracker(
            fps=config["fps"],
            threshold=config["head_threshold"],
            buffer_seconds=config["tracker_buffer_seconds"],
        )
        self.states, self.last_seen, self.evidence, self.crops = {}, {}, {}, {}
        self.recent_boxes, self.head_proof = {}, {}
        self.support_seen = {}
        self.previous_gray = None
        self.last_timestamp = None
        self.last_face_time = -100000
        self.sequence = 0
        self.diagnostics = {
            "face_passes": 0,
            "roi_passes": 0,
            "embeddings": 0,
            "quality_rejected": 0,
            "duplicate_identity_suppressed": 0,
        }

    def process(self, frame, timestamp_ms):
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, Integral)
            or timestamp_ms < 0
        ):
            raise ValueError("Timestamp must be a nonnegative integer in milliseconds")
        if self.last_timestamp is not None and timestamp_ms <= self.last_timestamp:
            raise ValueError(
                "Frame timestamps must increase; duplicate/stale frame rejected"
            )
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("Expected uint8 BGR image")
        start = perf_counter()
        prior_identities = {
            tid: s.person_id for tid, s in self.states.items() if s.person_id
        }
        self.last_timestamp = timestamp_ms
        self.sequence += 1
        raw, bodies = self.net.heads_and_bodies(frame, self.cfg["head_low_threshold"])
        full_faces = self.net.faces(frame)
        self.diagnostics["face_passes"] += 1
        face_pass = timestamp_ms - self.last_face_time >= self.cfg["face_interval_ms"]
        faces = full_faces
        if face_pass:
            self.last_face_time = timestamp_ms
            if self.cfg["roi_face_detection"]:
                h, w = frame.shape[:2]
                proposals = list(raw)
                proposals.extend(
                    np.r_[b[0], 0.5]
                    for b in self.recent_boxes.values()
                    if timestamp_ms - b[1] <= 1000
                )
                if proposals:
                    proposals = np.asarray(proposals, float)
                    proposals = proposals[nms(proposals[:, :4], proposals[:, 4], 0.4)][
                        : self.cfg["max_roi_passes"]
                    ]
                for proposal in proposals:
                    x1, y1, x2, y2 = proposal[:4]
                    size = max(x2 - x1, y2 - y1) * 2.2
                    center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
                    lo = np.maximum(np.floor(center - size / 2).astype(int), 0)
                    hi = np.minimum(np.ceil(center + size / 2).astype(int), [w, h])
                    if min(hi - lo) < 30:
                        continue
                    for f in self.net.faces(frame[lo[1] : hi[1], lo[0] : hi[0]]):
                        f["bbox"] += np.tile(lo, 2)
                        f["landmarks"] += lo
                        faces.append(f)
                    self.diagnostics["roi_passes"] += 1
        if faces:
            keep = nms(
                np.array([f["bbox"] for f in faces]),
                np.array([f["score"] for f in faces]),
                0.35,
            )
            faces = [faces[i] for i in keep]
        trusted = [
            b[0]
            for tid, b in self.recent_boxes.items()
            if timestamp_ms - self.support_seen.get(tid, -100000) <= 1000
            and self.head_proof.get(tid, {}).get("verified", False)
        ]
        raw = supported_heads(
            raw, bodies, faces, self.cfg.get("require_body_support", True), trusted
        )
        tracks = self.tracker.update(raw, frame.shape, timestamp_ms)
        for tid in getattr(self.tracker, "ambiguous_track_ids", set()):
            if tid in self.states:
                self.states[tid].suspend()
                self.crops[tid].clear()
        for t in tracks:
            tid = t["track_id"]
            if tid not in self.states:
                self.states[tid] = IdentityState(self.cfg)
                self.crops[tid] = deque(maxlen=12)
            if tid in self.last_seen and timestamp_ms - self.last_seen[tid] > 700:
                self.states[tid].suspend()
                self.crops[tid].clear()
            self.last_seen[tid] = timestamp_ms
            self.recent_boxes[tid] = (t["bbox"], timestamp_ms)
            self.states[tid].expire(timestamp_ms)
        matched, ambiguous_heads = association(tracks, faces, return_ambiguous=True)
        for i, t in enumerate(tracks):
            if body_support(t["bbox"], bodies) or i in matched:
                self.support_seen[t["track_id"]] = timestamp_ms
        for i in ambiguous_heads:
            self.states[tracks[i]["track_id"]].suspend()
            self.crops[tracks[i]["track_id"]].clear()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        diff = (
            cv2.absdiff(gray, self.previous_gray)
            if self.previous_gray is not None and self.previous_gray.shape == gray.shape
            else None
        )
        self.previous_gray = gray

        results, events = [], []
        for i, track in enumerate(tracks):
            tid = track["track_id"]
            state = self.states[tid]
            previous_identity = state.person_id
            face = faces[matched[i]] if i in matched else None
            q, score, margin, changed = None, None, None, False
            q = quality(frame, face, self.cfg) if face is not None else None
            proof = self.head_proof.setdefault(tid, {"verified": False, "motion": 0})
            box = np.asarray(track["bbox"]).astype(int)
            x1, x2 = np.clip(box[[0, 2]], 0, gray.shape[1])
            y1, y2 = np.clip(box[[1, 3]], 0, gray.shape[0])
            movement = (
                float(diff[y1:y2, x1:x2].mean())
                if diff is not None and x2 > x1 and y2 > y1
                else 0
            )
            update_head_evidence(
                proof,
                timestamp_ms,
                track["score"],
                face is not None,
                movement,
                self.cfg,
            )
            if self.cfg.get("require_head_evidence", True) and not proof["verified"]:
                self.diagnostics["unverified_head_frames"] = (
                    self.diagnostics.get("unverified_head_frames", 0) + 1
                )
                continue
            if face is not None and face_pass:
                if q["usable"]:
                    embedding, crop = self.net.embedding(frame, face["landmarks"])
                    self.diagnostics["embeddings"] += 1
                    candidate, score, margin = self.gallery.match(embedding)
                    self.crops[tid].append(
                        {
                            "candidate": candidate,
                            "crop": crop,
                            "timestamp_ms": timestamp_ms,
                            "quality": q["rank"],
                            "similarity": score,
                            "margin": margin,
                        }
                    )
                    changed = state.observe(timestamp_ms, candidate, score, margin)
                else:
                    self.diagnostics["quality_rejected"] += 1
            pid = state.person_id
            if tid in self.evidence:
                self.evidence[tid]["revoked"] = pid != self.evidence[tid]["person_id"]
            if previous_identity and previous_identity != pid:
                events.append(
                    {
                        "event": "identity_revoked",
                        "camera_id": self.camera_id,
                        "timestamp_ms": timestamp_ms,
                        "track_id": tid,
                        "person_id": previous_identity,
                        "display_name": self.gallery.by_id[previous_identity][
                            "display_name"
                        ],
                        "similarity": score,
                    }
                )
            if pid:
                p = self.gallery.by_id[pid]
                valid = [
                    c
                    for c in self.crops[tid]
                    if c["candidate"] == pid
                    and c["similarity"] >= self.cfg["match_threshold"]
                    and c["margin"] >= self.cfg["match_margin"]
                    and timestamp_ms - c["timestamp_ms"]
                    <= self.cfg["evidence_window_ms"]
                ]
                if valid:
                    best = max(valid, key=lambda c: c["quality"])
                    old = self.evidence.get(tid)
                    if (
                        old is None
                        or old["person_id"] != pid
                        or best["quality"] > old["quality"]
                        or timestamp_ms - old["timestamp_ms"] > 10000
                    ):
                        self.evidence[tid] = {
                            **best,
                            "person_id": pid,
                            "display_name": p["display_name"],
                            "reference": self.gallery.references[pid],
                            "track_id": tid,
                        }
                status = "identified"
                if changed:
                    events.append(
                        {
                            "event": "identity_confirmed",
                            "camera_id": self.camera_id,
                            "timestamp_ms": timestamp_ms,
                            "track_id": tid,
                            "person_id": pid,
                            "display_name": p["display_name"],
                            "similarity": score,
                        }
                    )
            elif state.samples:
                status = (
                    "pending"
                    if any(s[1] is not None for s in state.samples)
                    else "unknown"
                )
            else:
                status = "no_usable_face"
            if i in ambiguous_heads:
                status = "pending"
            if score is None and state.samples:
                score, margin = state.samples[-1][2:]
            results.append(
                {
                    "track_id": tid,
                    "head_bbox": [float(x) for x in track["bbox"]],
                    "head_score": float(track["score"]),
                    "face_bbox": None if face is None else face["bbox"].tolist(),
                    "recognition_state": status,
                    "person_id": pid,
                    "display_name": (
                        self.gallery.by_id[pid]["display_name"] if pid else None
                    ),
                    "similarity": score,
                    "margin": margin,
                    "quality": q,
                    "last_face_confirmed_ms": state.last_confirmed,
                    "face_observed_now": face is not None,
                }
            )
        # A single registered person cannot occupy two distinct heads in one frame.
        by_person = {}
        for r in results:
            if r["person_id"]:
                by_person.setdefault(r["person_id"], []).append(r)
        suppressed = set()
        for group in by_person.values():
            if len(group) < 2:
                continue
            # Competing claims are unresolved; do not award identity by recency.
            for r in group:
                self.states[r["track_id"]].suspend()
                self.crops[r["track_id"]].clear()
                suppressed.add(r["track_id"])
                events.append(
                    {
                        "event": "identity_revoked",
                        "camera_id": self.camera_id,
                        "timestamp_ms": timestamp_ms,
                        "track_id": r["track_id"],
                        "person_id": r["person_id"],
                        "display_name": r["display_name"],
                        "similarity": r["similarity"],
                        "reason": "simultaneous competing claims",
                    }
                )
                r.update(
                    person_id=None,
                    display_name=None,
                    recognition_state="pending",
                    last_face_confirmed_ms=None,
                )
                if r["track_id"] in self.evidence:
                    self.evidence[r["track_id"]]["revoked"] = True
                self.diagnostics["duplicate_identity_suppressed"] += 1
        events = [
            e
            for e in events
            if e["track_id"] not in suppressed or e["event"] == "identity_revoked"
        ]
        for tid in list(self.states):
            if (
                timestamp_ms - self.last_seen.get(tid, 0)
                > self.cfg["tracker_buffer_seconds"] * 1000
            ):
                self.states[tid].suspend()
            if timestamp_ms - self.last_seen.get(tid, 0) > 30000:
                del self.states[tid]
                self.crops.pop(tid, None)
                self.last_seen.pop(tid, None)
                self.recent_boxes.pop(tid, None)
                self.head_proof.pop(tid, None)
                self.support_seen.pop(tid, None)
        for tid, previous in prior_identities.items():
            state = self.states.get(tid)
            if state is None or state.person_id != previous:
                if tid in self.evidence:
                    self.evidence[tid]["revoked"] = True
                if not any(
                    e["event"] == "identity_revoked"
                    and e["track_id"] == tid
                    and e["person_id"] == previous
                    for e in events
                ):
                    events.append(
                        {
                            "event": "identity_revoked",
                            "camera_id": self.camera_id,
                            "timestamp_ms": timestamp_ms,
                            "track_id": tid,
                            "person_id": previous,
                            "display_name": self.gallery.by_id[previous][
                                "display_name"
                            ],
                            "reason": "expired, lost, ambiguous or contradicted",
                        }
                    )
        # Evidence is a bounded recent-confirmations gallery, not an identity database.
        for tid in list(self.evidence):
            if timestamp_ms - self.evidence[tid]["timestamp_ms"] > 60000:
                del self.evidence[tid]
        return {
            "schema_version": "1.0",
            "camera_id": self.camera_id,
            "timestamp_ms": timestamp_ms,
            "sequence": self.sequence,
            "processing_ms": (perf_counter() - start) * 1000,
            "tracks": results,
            "events": events,
        }
