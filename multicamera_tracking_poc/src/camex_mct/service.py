from __future__ import annotations

import copy
import os
import queue
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from .global_identity import GlobalIdentityStore
from .tracking import AppearanceCameraTracker, TrackObservation
from .types import FramePacket, FrameResult, TrackResult

ResultCallback = Callable[[FrameResult], None]


class Detector(Protocol):
    test_size: tuple[int, int]

    def infer(self, frames: Sequence[np.ndarray]) -> tuple[list[np.ndarray], dict[str, float]]: ...


class Embedder(Protocol):
    def embed(self, crops: Sequence[np.ndarray]) -> tuple[np.ndarray, float]: ...


class CameraTracker(Protocol):
    def appearance_boxes(
        self,
        detections: np.ndarray,
        frame_shape: tuple[int, int, int],
        detector_input_size: tuple[int, int],
    ) -> np.ndarray: ...

    def update(
        self,
        detections: np.ndarray,
        frame: np.ndarray,
        detector_input_size: tuple[int, int],
        features: np.ndarray,
    ) -> list[TrackObservation]: ...


class Engine(Protocol):
    camera_ids: tuple[str, ...]

    def process_batch(self, packets: Sequence[FramePacket]) -> list[FrameResult]: ...

    def summary(self) -> dict[str, Any]: ...

    def close(self) -> None: ...


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")

    root = (config_path.parent / config.pop("project_root", "..")).resolve()
    for section, fields in {
        "detector": ("model", "experiment", "torchscript_model"),
        "reid": ("model", "source"),
    }.items():
        values = config.get(section, {})
        for field in fields:
            if field in values and values[field]:
                candidate = Path(values[field]).expanduser()
                if not candidate.is_absolute():
                    dependency_root = os.environ.get("CAMEX_MCT_THIRD_PARTY")
                    if dependency_root and candidate.parts[0] == "third_party":
                        values[field] = str(
                            Path(dependency_root).expanduser().resolve()
                            / Path(*candidate.parts[1:])
                        )
                    else:
                        values[field] = str(root / candidate)
    config["project_root"] = str(root)
    return config


def _camera_ids(config: Mapping[str, Any]) -> tuple[str, ...]:
    cameras = config.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError("config.cameras must be a non-empty list")
    identifiers = tuple(
        str(item["camera_id"] if isinstance(item, dict) else item) for item in cameras
    )
    if any(not item for item in identifiers):
        raise ValueError("camera IDs must not be empty")
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("camera IDs must be unique")
    return identifiers


def _crop_box(
    frame: np.ndarray,
    box: np.ndarray,
    minimum_width: int,
    minimum_height: int,
) -> np.ndarray | None:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = box
    left = max(0, min(width - 1, int(x1)))
    top = max(0, min(height - 1, int(y1)))
    right = max(left + 1, min(width, int(x2)))
    bottom = max(top + 1, min(height, int(y2)))
    if right - left < minimum_width or bottom - top < minimum_height:
        return None
    return frame[top:bottom, left:right].copy()


class TrackingEngine:
    """Synchronous batched detector, per-camera SCT, and global ReID core."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        detector: Detector | None = None,
        embedder: Embedder | None = None,
        tracker_factory: Callable[[str, Mapping[str, Any]], CameraTracker] | None = None,
        identity_store: GlobalIdentityStore | None = None,
        warmup: bool = True,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.site_id = str(self.config.get("site_id", "default-site"))
        self.camera_ids = _camera_ids(self.config)
        runtime = self.config.get("runtime", {})
        self.model_batch_size = int(runtime.get("model_batch_size", len(self.camera_ids)))
        if self.model_batch_size < len(self.camera_ids):
            raise ValueError("model_batch_size cannot be smaller than camera count")
        if self.model_batch_size > 20:
            raise ValueError("v0.1 supports at most 20 configured cameras")

        if detector is None or embedder is None:
            from .models import OsNetEmbedder, YoloXPersonDetector

        detector_config = self.config["detector"]
        self.detector = detector or YoloXPersonDetector(
            experiment_path=detector_config["experiment"],
            checkpoint_path=detector_config["model"],
            input_size=int(detector_config.get("input_size", 416)),
            confidence=float(detector_config.get("confidence", 0.25)),
            nms=float(detector_config.get("nms", 0.45)),
            torchscript_path=detector_config.get("torchscript_model"),
            preprocess_workers=int(detector_config.get("preprocess_workers", 1)),
        )
        reid_config = self.config["reid"]
        self.embedder = embedder or OsNetEmbedder(
            source_file=reid_config["source"],
            checkpoint_path=reid_config["model"],
            architecture=reid_config.get("architecture", "osnet_ibn_x1_0"),
            batch_sizes=tuple(reid_config.get("batch_sizes", (8, 16, 32))),
            preprocess_workers=int(reid_config.get("preprocess_workers", 1)),
        )

        tracker_config = self.config["tracker"]
        target_fps = int(runtime.get("target_fps", 10))
        if tracker_factory is None:

            def default_tracker_factory(camera_id: str, values: Mapping[str, Any]) -> CameraTracker:
                return AppearanceCameraTracker(
                    camera_id=camera_id,
                    frame_rate=target_fps,
                    track_high_threshold=float(values["track_high_threshold"]),
                    track_low_threshold=float(values["track_low_threshold"]),
                    new_track_threshold=float(values["new_track_threshold"]),
                    track_buffer=int(values["track_buffer"]),
                    match_threshold=float(values["match_threshold"]),
                    proximity_threshold=float(values["proximity_threshold"]),
                    appearance_threshold=float(values["appearance_threshold"]),
                    minimum_box_area=float(values["minimum_box_area"]),
                )

            tracker_factory = default_tracker_factory
        self.trackers = {
            camera_id: tracker_factory(camera_id, tracker_config) for camera_id in self.camera_ids
        }
        self.identity_store = identity_store or GlobalIdentityStore(
            **self.config["global_identity"]
        )
        self.minimum_crop_width = int(reid_config.get("minimum_crop_width", 12))
        self.minimum_crop_height = int(reid_config.get("minimum_crop_height", 30))
        self._lock = threading.Lock()
        self._frames_processed = 0
        self._batches_processed = 0
        self._closed = False

        if warmup:
            if hasattr(self.detector, "warmup"):
                self.detector.warmup(batch_size=self.model_batch_size, iterations=2)
            if hasattr(self.embedder, "warmup"):
                self.embedder.warmup(iterations=1)

    def process_batch(self, packets: Sequence[FramePacket]) -> list[FrameResult]:
        if self._closed:
            raise RuntimeError("tracking engine is closed")
        if not packets:
            return []
        if len(packets) > self.model_batch_size:
            raise ValueError("packet count exceeds configured model_batch_size")

        order = {camera_id: index for index, camera_id in enumerate(self.camera_ids)}
        sorted_packets = sorted(packets, key=lambda item: order.get(item.camera_id, 10**9))
        seen: set[str] = set()
        for packet in sorted_packets:
            packet.validate()
            if packet.camera_id not in order:
                raise ValueError(f"unknown camera_id: {packet.camera_id}")
            if packet.camera_id in seen:
                raise ValueError(f"duplicate camera packet: {packet.camera_id}")
            seen.add(packet.camera_id)

        with self._lock:
            started = time.perf_counter()
            real_frames = [np.ascontiguousarray(packet.frame) for packet in sorted_packets]
            padding_frame = np.zeros_like(real_frames[0])
            model_frames = real_frames + [
                padding_frame for _ in range(self.model_batch_size - len(real_frames))
            ]
            detections, _ = self.detector.infer(model_frames)
            detections = detections[: len(real_frames)]

            crop_counts: list[int] = []
            crops: list[np.ndarray] = []
            for packet, camera_detections in zip(sorted_packets, detections, strict=False):
                tracker = self.trackers[packet.camera_id]
                boxes = tracker.appearance_boxes(
                    camera_detections,
                    packet.frame.shape,
                    self.detector.test_size,
                )
                crop_counts.append(len(boxes))
                for box in boxes:
                    crop = _crop_box(
                        packet.frame,
                        box,
                        self.minimum_crop_width,
                        self.minimum_crop_height,
                    )
                    crops.append(
                        crop if crop is not None else np.zeros((256, 128, 3), dtype=np.uint8)
                    )

            embeddings, _ = self.embedder.embed(crops)
            observations: dict[str, list[TrackObservation]] = {}
            offset = 0
            for packet, camera_detections, crop_count in zip(
                sorted_packets, detections, crop_counts, strict=False
            ):
                camera_embeddings = embeddings[offset : offset + crop_count]
                offset += crop_count
                observations[packet.camera_id] = self.trackers[packet.camera_id].update(
                    camera_detections,
                    packet.frame,
                    self.detector.test_size,
                    camera_embeddings,
                )

            result_tracks: dict[str, tuple[TrackResult, ...]] = {}
            for packet in sorted_packets:
                camera_observations = observations[packet.camera_id]
                timestamp_s = packet.timestamp_ms / 1000.0
                self.identity_store.begin_frame(
                    packet.camera_id,
                    {item.local_track_id for item in camera_observations},
                    timestamp_s,
                )
                tracks: list[TrackResult] = []
                for observation in camera_observations:
                    if observation.appearance_embedding is None:
                        assignment = self.identity_store.heartbeat(
                            packet.camera_id,
                            observation.local_track_id,
                            timestamp_s,
                        )
                    else:
                        assignment = self.identity_store.assign(
                            packet.camera_id,
                            observation.local_track_id,
                            observation.appearance_embedding,
                            timestamp_s,
                        )
                    observation.global_id = assignment[0]
                    observation.global_confidence = assignment[1]
                    observation.identity_status = assignment[2]
                    tracks.append(
                        TrackResult(
                            local_id=observation.local_track_id,
                            global_id=observation.global_id,
                            bbox_xyxy=observation.bbox_xyxy,
                            detection_confidence=observation.detection_confidence,
                            reid_confidence=observation.global_confidence,
                            status=observation.identity_status,
                        )
                    )
                result_tracks[packet.camera_id] = tuple(tracks)

            latency_ms = (time.perf_counter() - started) * 1000.0
            self._frames_processed += len(sorted_packets)
            self._batches_processed += 1
            return [
                FrameResult(
                    site_id=self.site_id,
                    camera_id=packet.camera_id,
                    timestamp_ms=packet.timestamp_ms,
                    tracks=result_tracks[packet.camera_id],
                    processing_latency_ms=latency_ms,
                )
                for packet in sorted_packets
            ]

    def summary(self) -> dict[str, Any]:
        return {
            "site_id": self.site_id,
            "camera_count": len(self.camera_ids),
            "model_batch_size": self.model_batch_size,
            "batches_processed": self._batches_processed,
            "frames_processed": self._frames_processed,
            **self.identity_store.summary(),
        }

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for component in (self.detector, self.embedder):
            closer = getattr(component, "close", None)
            if closer is not None:
                closer()


class TrackingService:
    """Nonblocking latest-frame adapter around :class:`TrackingEngine`."""

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        engine: Engine | None = None,
        on_result: ResultCallback | None = None,
    ) -> None:
        self.config = copy.deepcopy(dict(config))
        self.camera_ids = _camera_ids(self.config)
        runtime = self.config.get("runtime", {})
        self.target_fps = float(runtime.get("target_fps", 10))
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")
        self.engine = engine or TrackingEngine(self.config)
        self.on_result = on_result
        self._condition = threading.Condition()
        self._pending: dict[str, FramePacket] = {}
        self._last_timestamp_ms: dict[str, int] = {}
        self._results: queue.Queue[FrameResult] = queue.Queue(
            maxsize=int(runtime.get("result_queue_size", 256))
        )
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._fatal_error: BaseException | None = None
        self._submitted = 0
        self._replaced = 0
        self._stale = 0
        self._result_queue_drops = 0
        self._callback_errors = 0

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        on_result: ResultCallback | None = None,
    ) -> TrackingService:
        return cls(load_config(path), on_result=on_result)

    def start(self, on_result: ResultCallback | None = None) -> None:
        with self._condition:
            if self._thread is not None:
                raise RuntimeError("tracking service is already started")
            if on_result is not None:
                self.on_result = on_result
            self._stopping = False
            self._thread = threading.Thread(
                target=self._run,
                name="camex-mct-batcher",
                daemon=True,
            )
            self._thread.start()

    def submit_frame(
        self,
        camera_id: str,
        frame: np.ndarray,
        timestamp_ms: int,
    ) -> bool:
        packet = FramePacket(camera_id, frame, int(timestamp_ms))
        packet.validate()
        if camera_id not in self.camera_ids:
            raise ValueError(f"unknown camera_id: {camera_id}")
        with self._condition:
            if self._thread is None or self._stopping:
                raise RuntimeError("tracking service is not running")
            previous_timestamp = self._last_timestamp_ms.get(camera_id)
            if previous_timestamp is not None and timestamp_ms <= previous_timestamp:
                self._stale += 1
                return False
            if camera_id in self._pending:
                self._replaced += 1
            self._pending[camera_id] = packet
            self._last_timestamp_ms[camera_id] = int(timestamp_ms)
            self._submitted += 1
            self._condition.notify_all()
        return True

    def poll_result(self, timeout: float | None = None) -> FrameResult:
        return self._results.get(timeout=timeout)

    def _publish(self, result: FrameResult) -> None:
        if self.on_result is not None:
            try:
                self.on_result(result)
            except Exception:
                self._callback_errors += 1
        try:
            self._results.put_nowait(result)
        except queue.Full:
            try:
                self._results.get_nowait()
            except queue.Empty:
                pass
            self._result_queue_drops += 1
            self._results.put_nowait(result)

    def _run(self) -> None:
        interval = 1.0 / self.target_fps
        next_tick = time.monotonic() + interval
        try:
            while True:
                with self._condition:
                    while not self._stopping and time.monotonic() < next_tick:
                        self._condition.wait(timeout=next_tick - time.monotonic())
                    if self._stopping and not self._pending:
                        return
                    packets = [
                        self._pending.pop(camera_id)
                        for camera_id in self.camera_ids
                        if camera_id in self._pending
                    ]
                if packets:
                    for result in self.engine.process_batch(packets):
                        self._publish(result)
                next_tick = max(next_tick + interval, time.monotonic())
        except BaseException as exc:
            self._fatal_error = exc
            with self._condition:
                self._stopping = True
                self._condition.notify_all()

    def summary(self) -> dict[str, Any]:
        return {
            "submitted_frames": self._submitted,
            "replaced_before_processing": self._replaced,
            "rejected_stale_frames": self._stale,
            "result_queue_drops": self._result_queue_drops,
            "callback_errors": self._callback_errors,
            **self.engine.summary(),
        }

    def close(self) -> None:
        with self._condition:
            if self._thread is None:
                self.engine.close()
                return
            self._stopping = True
            self._condition.notify_all()
        self._thread.join(timeout=30)
        if self._thread.is_alive():
            raise TimeoutError("tracking service did not stop within 30 seconds")
        self._thread = None
        self.engine.close()
        if self._fatal_error is not None:
            raise RuntimeError("tracking worker failed") from self._fatal_error
