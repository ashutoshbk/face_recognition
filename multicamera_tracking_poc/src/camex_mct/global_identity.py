from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class GlobalRecord:
    global_id: int
    prototype: np.ndarray
    created_s: float
    last_seen_s: float
    observations: int = 1
    active_cameras: dict[str, float] = field(default_factory=dict)
    gallery: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=2048))
    last_gallery_sample_s: float = float("-inf")


class GlobalIdentityStore:
    def __init__(
        self,
        similarity_threshold: float,
        ambiguity_margin: float,
        active_timeout_seconds: float,
        gallery_retention_seconds: float,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ambiguity_margin = ambiguity_margin
        self.active_timeout_seconds = active_timeout_seconds
        self.gallery_retention_seconds = gallery_retention_seconds
        self.next_global_id = 1
        self.records: dict[int, GlobalRecord] = {}
        self.local_assignments: dict[tuple[str, int], int] = {}
        self.assignment_confidence: dict[tuple[str, int], float] = {}
        self.assignment_status: dict[tuple[str, int], str] = {}
        self.active_local_ids_by_camera: dict[str, set[int]] = {}
        self.new_assignments = 0
        self.reidentified_assignments = 0
        self.same_camera_reconnects = 0
        self.cross_camera_reidentified_assignments = 0
        self.match_scores: list[float] = []

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector if norm <= 1e-12 else vector / norm

    def has_assignment(self, camera_id: str, local_track_id: int) -> bool:
        return (camera_id, local_track_id) in self.local_assignments

    def begin_frame(
        self,
        camera_id: str,
        active_local_track_ids: set[int],
        timestamp_s: float,
    ) -> None:
        self.active_local_ids_by_camera[camera_id] = set(active_local_track_ids)

    def _eligible(
        self,
        record: GlobalRecord,
        camera_id: str,
        local_track_id: int,
        timestamp_s: float,
    ) -> bool:
        if timestamp_s - record.last_seen_s > self.gallery_retention_seconds:
            return False
        if camera_id in self.active_local_ids_by_camera:
            for active_local_id in self.active_local_ids_by_camera[camera_id]:
                if active_local_id == local_track_id:
                    continue
                assigned = self.local_assignments.get((camera_id, active_local_id))
                if assigned == record.global_id:
                    return False
            return True
        same_camera_seen = record.active_cameras.get(camera_id)
        if same_camera_seen is None:
            return True
        return timestamp_s - same_camera_seen > self.active_timeout_seconds

    @staticmethod
    def _record_similarity(record: GlobalRecord, embedding: np.ndarray) -> float:
        prototype_score = float(np.dot(record.prototype, embedding))
        if not record.gallery:
            return prototype_score
        gallery_scores = sorted(
            (float(np.dot(item, embedding)) for item in record.gallery),
            reverse=True,
        )
        gallery_score = float(np.mean(gallery_scores[:3]))
        best_anchor_score = gallery_scores[0]
        consensus_score = 0.50 * prototype_score + 0.50 * gallery_score
        historical_score = 0.20 * prototype_score + 0.80 * best_anchor_score
        return max(consensus_score, historical_score)

    @staticmethod
    def _sample_gallery(
        record: GlobalRecord,
        embedding: np.ndarray,
        timestamp_s: float,
    ) -> None:
        if not record.gallery:
            record.gallery.append(embedding.copy())
            record.last_gallery_sample_s = timestamp_s
            return
        best_similarity = max(float(np.dot(item, embedding)) for item in record.gallery)
        sample_due = timestamp_s - record.last_gallery_sample_s >= 1.0
        materially_new_view = best_similarity < 0.90
        if sample_due or materially_new_view:
            record.gallery.append(embedding.copy())
            record.last_gallery_sample_s = timestamp_s

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp_s: float,
    ) -> tuple[int, float, str]:
        key = (camera_id, local_track_id)
        embedding = self._normalize(embedding.astype(np.float32, copy=False))
        if key in self.local_assignments:
            global_id = self.local_assignments[key]
            record = self.records[global_id]
            similarity = float(np.dot(record.prototype, embedding))
            if similarity > 0.45:
                record.prototype = self._normalize(0.98 * record.prototype + 0.02 * embedding)
                record.observations += 1
                self._sample_gallery(record, embedding, timestamp_s)
            record.last_seen_s = timestamp_s
            record.active_cameras[camera_id] = timestamp_s
            return (
                global_id,
                self.assignment_confidence[key],
                self.assignment_status[key],
            )

        candidates: list[tuple[float, int]] = []
        for global_id, record in self.records.items():
            if self._eligible(record, camera_id, local_track_id, timestamp_s):
                candidates.append((self._record_similarity(record, embedding), global_id))
        candidates.sort(reverse=True)
        best_score = candidates[0][0] if candidates else -1.0
        second_score = candidates[1][0] if len(candidates) > 1 else -1.0
        has_margin = len(candidates) < 2 or best_score - second_score >= self.ambiguity_margin

        if candidates and best_score >= self.similarity_threshold and has_margin:
            global_id = candidates[0][1]
            confidence = best_score
            record = self.records[global_id]
            if camera_id in record.active_cameras:
                status = "same_camera_reconnected"
                self.same_camera_reconnects += 1
            else:
                status = "reidentified"
                self.cross_camera_reidentified_assignments += 1
            record.prototype = self._normalize(0.95 * record.prototype + 0.05 * embedding)
            record.last_seen_s = timestamp_s
            record.observations += 1
            record.active_cameras[camera_id] = timestamp_s
            self._sample_gallery(record, embedding, timestamp_s)
            self.reidentified_assignments += 1
            self.match_scores.append(best_score)
        else:
            global_id = self.next_global_id
            self.next_global_id += 1
            status = "new"
            confidence = 0.0
            record = GlobalRecord(
                global_id=global_id,
                prototype=embedding.copy(),
                created_s=timestamp_s,
                last_seen_s=timestamp_s,
                active_cameras={camera_id: timestamp_s},
            )
            self._sample_gallery(record, embedding, timestamp_s)
            self.records[global_id] = record
            self.new_assignments += 1

        self.local_assignments[key] = global_id
        self.assignment_confidence[key] = confidence
        self.assignment_status[key] = status
        return global_id, confidence, status

    def heartbeat(
        self,
        camera_id: str,
        local_track_id: int,
        timestamp_s: float,
    ) -> tuple[int | None, float, str]:
        key = (camera_id, local_track_id)
        global_id = self.local_assignments.get(key)
        if global_id is None:
            return None, 0.0, "pending"
        record = self.records[global_id]
        record.last_seen_s = timestamp_s
        record.active_cameras[camera_id] = timestamp_s
        return (
            global_id,
            self.assignment_confidence[key],
            self.assignment_status[key],
        )

    def summary(self) -> dict[str, float | int]:
        mean_score = float(np.mean(self.match_scores)) if self.match_scores else 0.0
        return {
            "global_identities": len(self.records),
            "new_assignments": self.new_assignments,
            "reidentified_assignments": self.reidentified_assignments,
            "same_camera_reconnects": self.same_camera_reconnects,
            "cross_camera_reidentified_assignments": self.cross_camera_reidentified_assignments,
            "mean_reid_match_similarity": mean_score,
        }
