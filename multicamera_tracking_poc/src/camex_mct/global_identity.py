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
    local_tracks: set[tuple[str, int]] = field(default_factory=set)


@dataclass
class LocalEvidence:
    embeddings: deque[np.ndarray]
    first_seen_s: float
    last_seen_s: float
    observations: int = 0
    last_recheck_s: float = float("-inf")


class GlobalIdentityStore:
    """Site-wide identity gallery with guarded, evidence-based reconciliation."""

    def __init__(
        self,
        similarity_threshold: float,
        ambiguity_margin: float,
        active_timeout_seconds: float,
        gallery_retention_seconds: float,
        same_camera_overlap_tolerance_seconds: float = 0.2,
        minimum_track_observations: int = 1,
        evidence_window_size: int = 24,
        reassignment_interval_seconds: float = 0.5,
        reassignment_grace_seconds: float = 10.0,
        reassignment_threshold: float | None = None,
        reassignment_margin: float | None = None,
        record_reconciliation_enabled: bool = False,
        record_reconciliation_interval_seconds: float = 0.5,
        record_reconciliation_similarity_threshold: float = 0.72,
        record_reconciliation_margin: float = 0.04,
        record_reconciliation_max_transition_seconds: float = 10.0,
        record_reconciliation_min_cameras: int = 2,
        record_reconciliation_min_observations: int = 8,
    ) -> None:
        self.similarity_threshold = similarity_threshold
        self.ambiguity_margin = ambiguity_margin
        self.active_timeout_seconds = active_timeout_seconds
        self.gallery_retention_seconds = gallery_retention_seconds
        self.same_camera_overlap_tolerance_seconds = max(0.0, same_camera_overlap_tolerance_seconds)
        self.minimum_track_observations = max(1, minimum_track_observations)
        self.evidence_window_size = max(self.minimum_track_observations, evidence_window_size)
        self.reassignment_interval_seconds = max(0.0, reassignment_interval_seconds)
        self.reassignment_grace_seconds = max(0.0, reassignment_grace_seconds)
        self.reassignment_threshold = (
            similarity_threshold if reassignment_threshold is None else reassignment_threshold
        )
        self.reassignment_margin = (
            ambiguity_margin if reassignment_margin is None else reassignment_margin
        )
        self.record_reconciliation_enabled = record_reconciliation_enabled
        self.record_reconciliation_interval_seconds = max(
            0.0, record_reconciliation_interval_seconds
        )
        self.record_reconciliation_similarity_threshold = record_reconciliation_similarity_threshold
        self.record_reconciliation_margin = max(0.0, record_reconciliation_margin)
        self.record_reconciliation_max_transition_seconds = max(
            0.0, record_reconciliation_max_transition_seconds
        )
        self.record_reconciliation_min_cameras = max(2, record_reconciliation_min_cameras)
        self.record_reconciliation_min_observations = max(1, record_reconciliation_min_observations)

        self.next_global_id = 1
        self.records: dict[int, GlobalRecord] = {}
        self.local_assignments: dict[tuple[str, int], int] = {}
        self.local_evidence: dict[tuple[str, int], LocalEvidence] = {}
        self.assignment_confidence: dict[tuple[str, int], float] = {}
        self.assignment_status: dict[tuple[str, int], str] = {}
        self.active_local_ids_by_camera: dict[str, set[int]] = {}
        self.global_id_aliases: dict[int, int] = {}
        self.reconciliation_history: list[dict[str, object]] = []
        self._pending_reconciliation_events: deque[dict[str, object]] = deque()
        self.last_record_reconciliation_s = float("-inf")

        self.new_assignments = 0
        self.reidentified_assignments = 0
        self.same_camera_reconnects = 0
        self.cross_camera_reidentified_assignments = 0
        self.corrected_assignments = 0
        self.retired_duplicate_records = 0
        self.pending_observations = 0
        self.active_conflict_splits = 0
        self.record_reconciliations = 0
        self.identity_alias_events = 0
        self.match_scores: list[float] = []

    @staticmethod
    def _normalize(vector: np.ndarray) -> np.ndarray:
        norm = float(np.linalg.norm(vector))
        return vector if norm <= 1e-12 else vector / norm

    def resolve_global_id(self, global_id: int) -> int:
        """Resolve any retired ID through the runtime alias chain."""
        resolved = global_id
        visited: set[int] = set()
        while resolved in self.global_id_aliases:
            if resolved in visited:
                raise RuntimeError("global identity alias cycle detected")
            visited.add(resolved)
            resolved = self.global_id_aliases[resolved]
        return resolved

    def drain_reconciliation_events(self) -> tuple[dict[str, object], ...]:
        """Return each new reconciliation once for publication by the service."""
        events = tuple(self._pending_reconciliation_events)
        self._pending_reconciliation_events.clear()
        return events

    def has_assignment(self, camera_id: str, local_track_id: int) -> bool:
        return (camera_id, local_track_id) in self.local_assignments

    def begin_frame(
        self,
        camera_id: str,
        active_local_track_ids: set[int],
        timestamp_s: float,
    ) -> None:
        self.active_local_ids_by_camera[camera_id] = set(active_local_track_ids)
        assignments_by_global_id: dict[int, list[int]] = {}
        for local_track_id in active_local_track_ids:
            global_id = self.local_assignments.get((camera_id, local_track_id))
            if global_id is not None:
                assignments_by_global_id.setdefault(global_id, []).append(local_track_id)

        for global_id, local_track_ids in assignments_by_global_id.items():
            if len(local_track_ids) < 2:
                continue
            keep_local_id = max(
                local_track_ids,
                key=lambda item: (
                    self.local_evidence[(camera_id, item)].last_seen_s,
                    self.local_evidence[(camera_id, item)].observations,
                ),
            )
            record = self.records[global_id]
            for local_track_id in local_track_ids:
                if local_track_id == keep_local_id:
                    continue
                key = (camera_id, local_track_id)
                self.local_assignments.pop(key, None)
                self.assignment_confidence.pop(key, None)
                self.assignment_status.pop(key, None)
                record.local_tracks.discard(key)
                self.active_conflict_splits += 1

    def _eligible(
        self,
        record: GlobalRecord,
        camera_id: str,
        local_track_id: int,
        timestamp_s: float,
    ) -> bool:
        if timestamp_s - record.last_seen_s > self.gallery_retention_seconds:
            return False

        for active_local_id in self.active_local_ids_by_camera.get(camera_id, set()):
            if active_local_id == local_track_id:
                continue
            if self.local_assignments.get((camera_id, active_local_id)) == record.global_id:
                return False

        current_key = (camera_id, local_track_id)
        current_evidence = self.local_evidence.get(current_key)
        if current_evidence is not None:
            for member_key in record.local_tracks:
                if member_key[0] != camera_id or member_key == current_key:
                    continue
                member_evidence = self.local_evidence.get(member_key)
                if member_evidence is None:
                    continue
                overlap_seconds = min(
                    current_evidence.last_seen_s, member_evidence.last_seen_s
                ) - max(current_evidence.first_seen_s, member_evidence.first_seen_s)
                if overlap_seconds > self.same_camera_overlap_tolerance_seconds:
                    return False

        if camera_id in self.active_local_ids_by_camera:
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
            (float(np.dot(item, embedding)) for item in record.gallery), reverse=True
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

    def _update_evidence(
        self,
        key: tuple[str, int],
        embedding: np.ndarray,
        timestamp_s: float,
    ) -> LocalEvidence:
        evidence = self.local_evidence.get(key)
        if evidence is None:
            evidence = LocalEvidence(
                embeddings=deque(maxlen=self.evidence_window_size),
                first_seen_s=timestamp_s,
                last_seen_s=timestamp_s,
            )
            self.local_evidence[key] = evidence
        evidence.embeddings.append(embedding.copy())
        evidence.last_seen_s = timestamp_s
        evidence.observations += 1
        return evidence

    def _evidence_prototype(self, evidence: LocalEvidence) -> np.ndarray:
        embeddings = np.stack(evidence.embeddings)
        centroid = self._normalize(np.mean(embeddings, axis=0))
        if len(embeddings) < 4:
            return centroid.astype(np.float32)
        similarities = embeddings @ centroid
        retained_count = max(3, int(np.ceil(len(embeddings) * 0.70)))
        retained_indices = np.argsort(similarities)[-retained_count:]
        return self._normalize(np.mean(embeddings[retained_indices], axis=0)).astype(np.float32)

    @staticmethod
    def _interval_gap(first: LocalEvidence, second: LocalEvidence) -> float:
        if first.last_seen_s < second.first_seen_s:
            return second.first_seen_s - first.last_seen_s
        if second.last_seen_s < first.first_seen_s:
            return first.first_seen_s - second.last_seen_s
        return 0.0

    @staticmethod
    def _record_cameras(record: GlobalRecord) -> set[str]:
        return {camera_id for camera_id, _ in record.local_tracks}

    def _records_historically_compatible(self, first: GlobalRecord, second: GlobalRecord) -> bool:
        for first_key in first.local_tracks:
            first_evidence = self.local_evidence.get(first_key)
            if first_evidence is None:
                continue
            for second_key in second.local_tracks:
                if first_key[0] != second_key[0]:
                    continue
                second_evidence = self.local_evidence.get(second_key)
                if second_evidence is None:
                    continue
                overlap_seconds = min(
                    first_evidence.last_seen_s, second_evidence.last_seen_s
                ) - max(first_evidence.first_seen_s, second_evidence.first_seen_s)
                if overlap_seconds > self.same_camera_overlap_tolerance_seconds:
                    return False
        return True

    def _record_reconciliation_score(
        self, fragment: GlobalRecord, established: GlobalRecord
    ) -> tuple[float, float, tuple[str, int] | None, tuple[str, int] | None]:
        best_score = -1.0
        best_gap = float("inf")
        best_fragment_key: tuple[str, int] | None = None
        best_established_key: tuple[str, int] | None = None
        for fragment_key in fragment.local_tracks:
            fragment_evidence = self.local_evidence.get(fragment_key)
            if (
                fragment_evidence is None
                or fragment_evidence.observations < self.record_reconciliation_min_observations
            ):
                continue
            fragment_prototype = self._evidence_prototype(fragment_evidence)
            for established_key in established.local_tracks:
                if fragment_key[0] == established_key[0]:
                    continue
                established_evidence = self.local_evidence.get(established_key)
                if (
                    established_evidence is None
                    or established_evidence.observations
                    < self.record_reconciliation_min_observations
                ):
                    continue
                gap = self._interval_gap(fragment_evidence, established_evidence)
                if gap > self.record_reconciliation_max_transition_seconds:
                    continue
                established_prototype = self._evidence_prototype(established_evidence)
                score = float(np.dot(fragment_prototype, established_prototype))
                if score > best_score:
                    best_score = score
                    best_gap = gap
                    best_fragment_key = fragment_key
                    best_established_key = established_key
        return best_score, best_gap, best_fragment_key, best_established_key

    @staticmethod
    def _ranking_has_margin(ranking: list[tuple[float, int]], margin: float) -> bool:
        return len(ranking) < 2 or ranking[0][0] - ranking[1][0] >= margin

    def _merge_record(
        self,
        source_global_id: int,
        target_global_id: int,
        score: float,
        transition_gap_s: float,
        source_track: tuple[str, int] | None,
        target_track: tuple[str, int] | None,
        timestamp_s: float,
    ) -> None:
        source = self.records[source_global_id]
        target = self.records[target_global_id]
        source_observations = max(source.observations, 1)
        target_observations = max(target.observations, 1)
        total_observations = source_observations + target_observations
        target.prototype = self._normalize(
            (source_observations * source.prototype + target_observations * target.prototype)
            / total_observations
        ).astype(np.float32)
        target.observations = total_observations
        target.created_s = min(target.created_s, source.created_s)
        target.last_seen_s = max(target.last_seen_s, source.last_seen_s)
        for camera_id, seen_s in source.active_cameras.items():
            target.active_cameras[camera_id] = max(
                target.active_cameras.get(camera_id, float("-inf")), seen_s
            )
        for embedding in source.gallery:
            target.gallery.append(embedding.copy())
        target.last_gallery_sample_s = max(
            target.last_gallery_sample_s, source.last_gallery_sample_s
        )

        source_tracks = sorted(source.local_tracks)
        for key in source_tracks:
            self.local_assignments[key] = target_global_id
            self.assignment_confidence[key] = score
            self.assignment_status[key] = "record_reconciled"
            target.local_tracks.add(key)
        del self.records[source_global_id]
        self.record_reconciliations += 1
        self._register_alias(
            source_global_id=source_global_id,
            target_global_id=target_global_id,
            score=score,
            transition_gap_s=transition_gap_s,
            source_track=source_track,
            target_track=target_track,
            source_track_count=len(source_tracks),
            target_camera_count=len(self._record_cameras(target)),
            timestamp_s=timestamp_s,
            reason="record_reconciled",
        )

    def _register_alias(
        self,
        source_global_id: int,
        target_global_id: int,
        score: float,
        transition_gap_s: float,
        source_track: tuple[str, int] | None,
        target_track: tuple[str, int] | None,
        source_track_count: int,
        target_camera_count: int,
        timestamp_s: float,
        reason: str,
    ) -> None:
        target_global_id = self.resolve_global_id(target_global_id)
        for alias, destination in list(self.global_id_aliases.items()):
            if destination == source_global_id:
                self.global_id_aliases[alias] = target_global_id
        self.global_id_aliases[source_global_id] = target_global_id
        self.identity_alias_events += 1
        event = {
            "sequence": self.identity_alias_events,
            "source_global_id": source_global_id,
            "canonical_global_id": target_global_id,
            "similarity": score,
            "transition_gap_s": transition_gap_s,
            "source_track": source_track,
            "target_track": target_track,
            "source_track_count": source_track_count,
            "target_camera_count": target_camera_count,
            "reconciled_at_s": timestamp_s,
            "reason": reason,
        }
        self.reconciliation_history.append(event)
        self._pending_reconciliation_events.append(event.copy())

    def _maybe_reconcile_records(self, timestamp_s: float) -> None:
        if not self.record_reconciliation_enabled:
            return
        if (
            timestamp_s - self.last_record_reconciliation_s
            < self.record_reconciliation_interval_seconds
        ):
            return
        self.last_record_reconciliation_s = timestamp_s

        while True:
            established_ids = [
                global_id
                for global_id, record in self.records.items()
                if len(self._record_cameras(record)) >= self.record_reconciliation_min_cameras
            ]
            fragment_ids = [
                global_id
                for global_id, record in self.records.items()
                if len(self._record_cameras(record)) < self.record_reconciliation_min_cameras
            ]
            if not established_ids or not fragment_ids:
                return

            pair_details: dict[
                tuple[int, int],
                tuple[float, float, tuple[str, int] | None, tuple[str, int] | None],
            ] = {}
            fragment_rankings: dict[int, list[tuple[float, int]]] = {}
            established_rankings: dict[int, list[tuple[float, int]]] = {}
            for fragment_id in fragment_ids:
                fragment = self.records[fragment_id]
                ranking: list[tuple[float, int]] = []
                for established_id in established_ids:
                    established = self.records[established_id]
                    if not self._records_historically_compatible(fragment, established):
                        continue
                    details = self._record_reconciliation_score(fragment, established)
                    pair_details[(fragment_id, established_id)] = details
                    ranking.append((details[0], established_id))
                    established_rankings.setdefault(established_id, []).append(
                        (details[0], fragment_id)
                    )
                ranking.sort(reverse=True)
                fragment_rankings[fragment_id] = ranking
            for ranking in established_rankings.values():
                ranking.sort(reverse=True)

            proposals: list[tuple[float, int, int]] = []
            for fragment_id, ranking in fragment_rankings.items():
                if not ranking:
                    continue
                score, established_id = ranking[0]
                if score < self.record_reconciliation_similarity_threshold:
                    continue
                if not self._ranking_has_margin(ranking, self.record_reconciliation_margin):
                    continue
                reverse_ranking = established_rankings[established_id]
                if reverse_ranking[0][1] != fragment_id:
                    continue
                if not self._ranking_has_margin(reverse_ranking, self.record_reconciliation_margin):
                    continue
                proposals.append((score, fragment_id, established_id))
            if not proposals:
                return

            score, fragment_id, established_id = max(proposals)
            details = pair_details[(fragment_id, established_id)]
            self._merge_record(
                fragment_id,
                established_id,
                score,
                details[1],
                details[2],
                details[3],
                timestamp_s,
            )

    def _rank_candidates(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp_s: float,
        excluded_global_id: int | None = None,
    ) -> list[tuple[float, int]]:
        candidates: list[tuple[float, int]] = []
        for global_id, record in self.records.items():
            if global_id == excluded_global_id:
                continue
            if self._eligible(record, camera_id, local_track_id, timestamp_s):
                candidates.append((self._record_similarity(record, embedding), global_id))
        candidates.sort(reverse=True)
        return candidates

    @staticmethod
    def _has_margin(candidates: list[tuple[float, int]], margin: float) -> bool:
        return len(candidates) < 2 or candidates[0][0] - candidates[1][0] >= margin

    def _update_record(
        self,
        record: GlobalRecord,
        key: tuple[str, int],
        embedding: np.ndarray,
        timestamp_s: float,
        prototype_weight: float,
    ) -> None:
        similarity = float(np.dot(record.prototype, embedding))
        if similarity > 0.45:
            record.prototype = self._normalize(
                (1.0 - prototype_weight) * record.prototype + prototype_weight * embedding
            )
            record.observations += 1
            self._sample_gallery(record, embedding, timestamp_s)
        record.last_seen_s = timestamp_s
        record.active_cameras[key[0]] = timestamp_s
        record.local_tracks.add(key)

    def _attach_existing(
        self,
        key: tuple[str, int],
        global_id: int,
        confidence: float,
        embedding: np.ndarray,
        timestamp_s: float,
        corrected: bool = False,
    ) -> tuple[int, float, str]:
        camera_id = key[0]
        record = self.records[global_id]
        cross_camera = any(track_camera != camera_id for track_camera, _ in record.local_tracks)
        if corrected:
            status = "corrected_reidentified" if cross_camera else "corrected_same_camera"
            self.corrected_assignments += 1
        elif cross_camera:
            status = "reidentified"
        else:
            status = "same_camera_reconnected"
        if cross_camera:
            self.cross_camera_reidentified_assignments += 1
        else:
            self.same_camera_reconnects += 1
        self.reidentified_assignments += 1
        self.match_scores.append(confidence)
        self.local_assignments[key] = global_id
        self.assignment_confidence[key] = confidence
        self.assignment_status[key] = status
        self._update_record(record, key, embedding, timestamp_s, prototype_weight=0.05)
        self._maybe_reconcile_records(timestamp_s)
        resolved_global_id = self.local_assignments[key]
        if resolved_global_id != global_id:
            return resolved_global_id, confidence, "record_reconciled"
        return global_id, confidence, status

    def _create_record(
        self,
        key: tuple[str, int],
        embedding: np.ndarray,
        timestamp_s: float,
    ) -> tuple[int, float, str]:
        global_id = self.next_global_id
        self.next_global_id += 1
        record = GlobalRecord(
            global_id=global_id,
            prototype=embedding.copy(),
            created_s=timestamp_s,
            last_seen_s=timestamp_s,
            active_cameras={key[0]: timestamp_s},
            local_tracks={key},
        )
        self._sample_gallery(record, embedding, timestamp_s)
        self.records[global_id] = record
        self.local_assignments[key] = global_id
        self.assignment_confidence[key] = 0.0
        self.assignment_status[key] = "new"
        self.new_assignments += 1
        self._maybe_reconcile_records(timestamp_s)
        resolved_global_id = self.local_assignments[key]
        if resolved_global_id != global_id:
            return resolved_global_id, 0.0, "record_reconciled"
        return global_id, 0.0, "new"

    def assign(
        self,
        camera_id: str,
        local_track_id: int,
        embedding: np.ndarray,
        timestamp_s: float,
    ) -> tuple[int | None, float, str]:
        key = (camera_id, local_track_id)
        embedding = self._normalize(embedding.astype(np.float32, copy=False))
        evidence = self._update_evidence(key, embedding, timestamp_s)
        local_prototype = self._evidence_prototype(evidence)

        if evidence.observations < self.minimum_track_observations:
            self.pending_observations += 1
            return None, 0.0, "pending"

        if key in self.local_assignments:
            global_id = self.local_assignments[key]
            record = self.records[global_id]
            recheck_due = (
                timestamp_s - evidence.last_recheck_s >= self.reassignment_interval_seconds
            )
            singleton_provisional = (
                record.local_tracks == {key}
                and timestamp_s - record.created_s <= self.reassignment_grace_seconds
            )
            if recheck_due and singleton_provisional:
                evidence.last_recheck_s = timestamp_s
                candidates = self._rank_candidates(
                    camera_id,
                    local_track_id,
                    local_prototype,
                    timestamp_s,
                    excluded_global_id=global_id,
                )
                if (
                    candidates
                    and candidates[0][0] >= self.reassignment_threshold
                    and self._has_margin(candidates, self.reassignment_margin)
                ):
                    target_score, target_global_id = candidates[0]
                    target_record = self.records[target_global_id]
                    target_track = min(
                        target_record.local_tracks,
                        key=lambda item: self._interval_gap(evidence, self.local_evidence[item]),
                    )
                    transition_gap_s = self._interval_gap(
                        evidence, self.local_evidence[target_track]
                    )
                    record.local_tracks.remove(key)
                    if not record.local_tracks:
                        del self.records[global_id]
                        self.retired_duplicate_records += 1
                    self._register_alias(
                        source_global_id=global_id,
                        target_global_id=target_global_id,
                        score=target_score,
                        transition_gap_s=transition_gap_s,
                        source_track=key,
                        target_track=target_track,
                        source_track_count=1,
                        target_camera_count=len(self._record_cameras(target_record)),
                        timestamp_s=timestamp_s,
                        reason="provisional_reassigned",
                    )
                    return self._attach_existing(
                        key,
                        target_global_id,
                        target_score,
                        local_prototype,
                        timestamp_s,
                        corrected=True,
                    )

            self._update_record(
                record,
                key,
                local_prototype,
                timestamp_s,
                prototype_weight=0.02,
            )
            self._maybe_reconcile_records(timestamp_s)
            global_id = self.local_assignments[key]
            return (
                global_id,
                self.assignment_confidence[key],
                self.assignment_status[key],
            )

        candidates = self._rank_candidates(
            camera_id,
            local_track_id,
            local_prototype,
            timestamp_s,
        )
        best_score = candidates[0][0] if candidates else -1.0
        has_margin = self._has_margin(candidates, self.ambiguity_margin)
        if candidates and best_score >= self.similarity_threshold and has_margin:
            return self._attach_existing(
                key,
                candidates[0][1],
                best_score,
                local_prototype,
                timestamp_s,
            )
        return self._create_record(key, local_prototype, timestamp_s)

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
        global_id = self.resolve_global_id(global_id)
        self.local_assignments[key] = global_id
        record = self.records[global_id]
        record.last_seen_s = timestamp_s
        record.active_cameras[camera_id] = timestamp_s
        return (
            global_id,
            self.assignment_confidence[key],
            self.assignment_status[key],
        )

    def summary(self) -> dict[str, object]:
        mean_score = float(np.mean(self.match_scores)) if self.match_scores else 0.0
        return {
            "global_identities": len(self.records),
            "new_assignments": self.new_assignments,
            "reidentified_assignments": self.reidentified_assignments,
            "same_camera_reconnects": self.same_camera_reconnects,
            "cross_camera_reidentified_assignments": (self.cross_camera_reidentified_assignments),
            "corrected_assignments": self.corrected_assignments,
            "retired_duplicate_records": self.retired_duplicate_records,
            "pending_observations": self.pending_observations,
            "active_conflict_splits": self.active_conflict_splits,
            "record_reconciliations": self.record_reconciliations,
            "identity_alias_events": self.identity_alias_events,
            "global_id_aliases": {
                str(source): self.resolve_global_id(target)
                for source, target in sorted(self.global_id_aliases.items())
            },
            "reconciliation_history": self.reconciliation_history,
            "mean_reid_match_similarity": mean_score,
        }
