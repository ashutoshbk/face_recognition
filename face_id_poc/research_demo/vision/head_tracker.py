"""Model-free head tracking over source-pixel detections and source time."""

from __future__ import annotations

from itertools import count
from math import isfinite
from numbers import Integral, Real
from types import SimpleNamespace

import numpy as np

from .vendor.bytetrack.basetrack import TrackState
from .vendor.bytetrack.byte_tracker import BYTETracker


class HeadTracker:
    """One ByteTrack session with independent, never-recycled integer IDs.

    ``fps`` is the nominal rate of calls to ``update``, not necessarily the
    camera's capture rate. Lost tracks survive at most ``buffer_seconds`` of
    source time and ``floor(fps * buffer_seconds)`` update steps. Motion uses
    the upstream one-step-per-update Kalman model.

    ``threshold`` is inclusive for high-score association and new tracks.
    Scores above 0.1 but below it can update already-tracked heads through
    ByteTrack's second association. New arrivals after the first frame need
    a second matching observation to become confirmed, as in upstream.

    ``geometry_fallback`` adds tightly gated center/scale matching to IoU.
    Near-tied candidates are deferred rather than assigning or creating an
    arbitrary identity; ``association_ambiguous`` reports this for the last
    valid update and ``ambiguous_track_ids`` identifies affected existing IDs,
    including tracks absent from the output. This is geometric evidence, not
    a guarantee of person ID.

    A source gap strictly greater than 2000 ms resets the tracking lifecycle,
    but not the session's ID counter. This class is for serial updates of one
    source; separate sources should use separate instances.
    """

    def __init__(
        self,
        fps: float = 10,
        threshold: float = 0.4,
        buffer_seconds: float = 1.0,
        match_threshold: float = 0.8,
        *,
        geometry_fallback: bool = True,
    ) -> None:
        if not isinstance(geometry_fallback, bool):
            raise ValueError("geometry_fallback must be a bool")
        for name, value in (
            ("fps", fps),
            ("threshold", threshold),
            ("buffer_seconds", buffer_seconds),
            ("match_threshold", match_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not isfinite(value)
            ):
                raise ValueError(f"{name} must be a finite real number")
        if fps <= 0:
            raise ValueError("fps must be positive")
        if not 0 < threshold <= 1:
            raise ValueError("threshold must be in (0, 1]")
        if buffer_seconds < 0 or not isfinite(fps * buffer_seconds):
            raise ValueError(
                "buffer_seconds must be nonnegative with finite fps * buffer_seconds"
            )
        if not 0 <= match_threshold <= 1:
            raise ValueError("match_threshold must be in [0, 1]")

        self._fps = float(fps)
        self._buffer_ms = float(buffer_seconds) * 1000
        self._args = SimpleNamespace(
            track_thresh=float(threshold),
            track_buffer=float(buffer_seconds) * 30,
            match_thresh=float(match_threshold),
            mot20=False,
            geometry_fallback=geometry_fallback,
        )
        self._id_counter = count(1)
        self._last_timestamp_ms: int | None = None
        self._last_seen_ms: dict[int, int] = {}
        self._tracker = self._new_tracker()

    def _new_tracker(self) -> BYTETracker:
        tracker = BYTETracker(
            self._args, frame_rate=self._fps, id_counter=self._id_counter
        )
        tracker.buffer_size = int(self._fps * (self._buffer_ms / 1000))
        tracker.max_time_lost = tracker.buffer_size
        # The public threshold controls births as well as high-score matches.
        tracker.det_thresh = self._args.track_thresh
        return tracker

    @property
    def association_ambiguous(self) -> bool:
        """Whether the last valid update deferred competing geometric matches.

        Deferred observations do not update tracks or create new IDs. Other
        unambiguous tracks can still be returned. False does not certify an
        identity: complete occlusion/crossing cannot be resolved by geometry.
        """
        return self._tracker.association_ambiguous

    @property
    def ambiguous_track_ids(self) -> set[int]:
        """Existing local IDs deferred on the last valid update, as a copy.

        Consumers can immediately suspend live identities for these IDs even
        when no observations are returned. This is a per-update diagnostic,
        not a persistent identity state; it clears on the next unambiguous
        update (including an empty one) or lifecycle reset. Invalid updates
        leave it unchanged. IDs are not merged, recycled, or reassigned.
        """
        return set(self._tracker.ambiguous_track_ids)

    @staticmethod
    def _prepare_detections(detections, frame_shape) -> tuple[np.ndarray, int, int]:
        try:
            height, width = frame_shape[:2]
        except (TypeError, ValueError, IndexError) as exc:
            raise ValueError("frame_shape must contain positive integer H, W") from exc
        if any(
            isinstance(n, (bool, np.bool_)) or not isinstance(n, Integral) or n <= 0
            for n in (height, width)
        ):
            raise ValueError("frame_shape must contain positive integer H, W")
        height, width = int(height), int(width)

        if detections is None:
            boxes = np.empty((0, 5), dtype=float)
        else:
            try:
                boxes = np.array(detections, dtype=float, copy=True)
            except (TypeError, ValueError) as exc:
                raise ValueError("detections must be a numeric Nx5 array") from exc
            if boxes.shape == (0,):
                boxes = boxes.reshape(0, 5)
        if boxes.ndim != 2 or boxes.shape[1] != 5:
            raise ValueError("detections must have shape (N, 5): x1, y1, x2, y2, score")
        if not np.isfinite(boxes).all():
            raise ValueError("detections must contain only finite values")
        if np.any((boxes[:, 4] < 0) | (boxes[:, 4] > 1)):
            raise ValueError("detection scores must be in [0, 1]")

        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height)
        valid = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
        return boxes[valid], height, width

    def update(self, detections, frame_shape, timestamp_ms: int) -> list[dict]:
        """Associate source-pixel ``Nx5 [x1, y1, x2, y2, score]`` detections.

        ``frame_shape`` is ``(H, W, ...)``. ``None``, ``[]`` and ``(0, 5)``
        arrays mean no detections, and still advance time and track state.
        Boxes are copied, clipped to the frame, and zero-area boxes discarded;
        there is no person-aspect-ratio or minimum-area filter.

        Returns only confirmed, active tracks matched/created in this update:
        ``{track_id: int, bbox: [float, float, float, float], score: float}``.
        Bboxes are Kalman-corrected source-pixel XYXY, clipped to the frame;
        scores are from the current observations. Lost/predicted-only tracks
        are never returned. IDs start at 1; tentative tracks may consume IDs.

        Raises ``ValueError`` for invalid input or non-integer/non-increasing
        source timestamps, before mutating the session. Integer timestamps
        can have any origin and must increase even for empty input.
        """
        if isinstance(timestamp_ms, (bool, np.bool_)) or not isinstance(
            timestamp_ms, Integral
        ):
            raise ValueError("timestamp_ms must be an integer source timestamp")
        timestamp_ms = int(timestamp_ms)
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms <= self._last_timestamp_ms
        ):
            raise ValueError("timestamp_ms must be strictly increasing")
        boxes, height, width = self._prepare_detections(detections, frame_shape)

        if (
            self._last_timestamp_ms is not None
            and timestamp_ms - self._last_timestamp_ms > 2000
        ):
            self._tracker = self._new_tracker()
            self._last_seen_ms.clear()

        # Source-time expiry also handles skipped input frames without replaying
        # synthetic frames or allowing stale identities into the assignment.
        for name in ("tracked_stracks", "lost_stracks"):
            survivors = []
            for track in getattr(self._tracker, name):
                if timestamp_ms - self._last_seen_ms[track.track_id] > self._buffer_ms:
                    track.mark_removed()
                else:
                    survivors.append(track)
            setattr(self._tracker, name, survivors)

        active = self._tracker.update(boxes, (height, width), (height, width))
        self._last_timestamp_ms = timestamp_ms
        live = self._tracker.tracked_stracks + self._tracker.lost_stracks
        self._last_seen_ms = {
            track.track_id: (
                timestamp_ms
                if track.frame_id == self._tracker.frame_id
                else self._last_seen_ms[track.track_id]
            )
            for track in live
        }

        result = []
        for track in active:
            if (
                track.state != TrackState.Tracked
                or not track.is_activated
                or track.frame_id != self._tracker.frame_id
            ):
                continue
            bbox = track.tlbr.copy()
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, width)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, height)
            if not np.isfinite(bbox).all() or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            result.append(
                {
                    "track_id": int(track.track_id),
                    "bbox": bbox.tolist(),
                    "score": float(track.score),
                }
            )
        return sorted(result, key=lambda track: track["track_id"])
