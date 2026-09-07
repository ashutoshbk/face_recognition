from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

if "float" not in np.__dict__:
    np.float = float  # type: ignore[attr-defined]

from .dependencies import configure_third_party


@dataclass
class TrackObservation:
    camera_id: str
    local_track_id: int
    raw_track_id: int
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float
    track_age: int
    global_id: int | None = None
    global_confidence: float = 0.0
    identity_status: str = "pending"
    appearance_embedding: np.ndarray | None = None


class CameraTracker:
    def __init__(
        self,
        camera_id: str,
        frame_rate: int,
        track_threshold: float,
        track_buffer: int,
        match_threshold: float,
        minimum_box_area: float,
    ) -> None:
        configure_third_party()
        from yolox.tracker.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_thresh=track_threshold,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            mot20=False,
        )
        self.camera_id = camera_id
        self.tracker = BYTETracker(args, frame_rate=frame_rate)
        self.minimum_box_area = minimum_box_area
        self.raw_to_local: dict[int, int] = {}
        self.next_local_id = 1

    def update(
        self,
        detections: np.ndarray,
        frame_shape: tuple[int, int, int],
        detector_input_size: tuple[int, int],
    ) -> list[TrackObservation]:
        height, width = frame_shape[:2]
        targets = self.tracker.update(detections, [height, width], detector_input_size)
        observations: list[TrackObservation] = []
        for target in targets:
            x1, y1, x2, y2 = (float(value) for value in target.tlbr)
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            if box_width * box_height < self.minimum_box_area:
                continue
            if box_height <= 0 or box_width / box_height > 1.6:
                continue
            raw_id = int(target.track_id)
            if raw_id not in self.raw_to_local:
                self.raw_to_local[raw_id] = self.next_local_id
                self.next_local_id += 1
            observations.append(
                TrackObservation(
                    camera_id=self.camera_id,
                    local_track_id=self.raw_to_local[raw_id],
                    raw_track_id=raw_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    detection_confidence=float(target.score),
                    track_age=int(target.tracklet_len) + 1,
                )
            )
        return observations


class _PrecomputedEncoder:
    def __init__(self) -> None:
        self.features = np.empty((0, 512), dtype=np.float32)

    def set_features(self, features: np.ndarray) -> None:
        self.features = features

    def inference(self, image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        if len(boxes) != len(self.features):
            raise RuntimeError("BoT-SORT detection and appearance feature counts differ")
        return self.features


class AppearanceCameraTracker:
    """BoT-SORT with externally batched full-body ReID embeddings."""

    def __init__(
        self,
        camera_id: str,
        frame_rate: int,
        track_high_threshold: float,
        track_low_threshold: float,
        new_track_threshold: float,
        track_buffer: int,
        match_threshold: float,
        proximity_threshold: float,
        appearance_threshold: float,
        minimum_box_area: float,
    ) -> None:
        configure_third_party()
        try:
            from tracker.bot_sort import BoTSORT
        except ImportError as exc:
            raise RuntimeError("BoT-SORT is unavailable; run `make setup` first") from exc
        args = SimpleNamespace(
            track_high_thresh=track_high_threshold,
            track_low_thresh=track_low_threshold,
            new_track_thresh=new_track_threshold,
            track_buffer=track_buffer,
            match_thresh=match_threshold,
            proximity_thresh=proximity_threshold,
            appearance_thresh=appearance_threshold,
            with_reid=False,
            cmc_method="none",
            mot20=False,
            name=camera_id,
            ablation=False,
        )
        self.camera_id = camera_id
        self.tracker = BoTSORT(args, frame_rate=frame_rate)
        self.encoder = _PrecomputedEncoder()
        self.tracker.encoder = self.encoder
        self.tracker.args.with_reid = True
        self.track_high_threshold = track_high_threshold
        self.minimum_box_area = minimum_box_area
        self.raw_to_local: dict[int, int] = {}
        self.next_local_id = 1

    @staticmethod
    def _to_image_coordinates(
        detections: np.ndarray,
        frame_shape: tuple[int, int, int],
        detector_input_size: tuple[int, int],
    ) -> np.ndarray:
        if detections.size == 0:
            return np.empty((0, 5), dtype=np.float32)
        height, width = frame_shape[:2]
        ratio = min(
            detector_input_size[0] / height,
            detector_input_size[1] / width,
        )
        scaled = detections.astype(np.float32, copy=True)
        scaled[:, :4] /= ratio
        scaled[:, [0, 2]] = np.clip(scaled[:, [0, 2]], 0, width - 1)
        scaled[:, [1, 3]] = np.clip(scaled[:, [1, 3]], 0, height - 1)
        return scaled

    def appearance_boxes(
        self,
        detections: np.ndarray,
        frame_shape: tuple[int, int, int],
        detector_input_size: tuple[int, int],
    ) -> np.ndarray:
        scaled = self._to_image_coordinates(detections, frame_shape, detector_input_size)
        if scaled.size == 0:
            return np.empty((0, 4), dtype=np.float32)
        return scaled[scaled[:, 4] > self.track_high_threshold, :4]

    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray,
        detector_input_size: tuple[int, int],
        features: np.ndarray,
    ) -> list[TrackObservation]:
        scaled = self._to_image_coordinates(detections, frame.shape, detector_input_size)
        self.encoder.set_features(features)
        targets = self.tracker.update(scaled, frame)
        observations: list[TrackObservation] = []
        for target in targets:
            x1, y1, x2, y2 = (float(value) for value in target.tlbr)
            box_width = max(0.0, x2 - x1)
            box_height = max(0.0, y2 - y1)
            if box_width * box_height < self.minimum_box_area:
                continue
            if box_height <= 0 or box_width / box_height > 1.6:
                continue
            raw_id = int(target.track_id)
            if raw_id not in self.raw_to_local:
                self.raw_to_local[raw_id] = self.next_local_id
                self.next_local_id += 1
            observations.append(
                TrackObservation(
                    camera_id=self.camera_id,
                    local_track_id=self.raw_to_local[raw_id],
                    raw_track_id=raw_id,
                    bbox_xyxy=(x1, y1, x2, y2),
                    detection_confidence=float(target.score),
                    track_age=int(target.tracklet_len) + 1,
                    appearance_embedding=(
                        target.smooth_feat.copy() if target.smooth_feat is not None else None
                    ),
                )
            )
        return observations
