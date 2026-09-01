from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from camex_mct.global_identity import GlobalIdentityStore
from camex_mct.service import TrackingEngine, TrackingService
from camex_mct.tracking import TrackObservation
from camex_mct.types import FramePacket, FrameResult


class FakeDetector:
    test_size = (416, 416)

    def __init__(self) -> None:
        self.last_batch_size = 0

    def infer(self, frames: Sequence[np.ndarray]):
        self.last_batch_size = len(frames)
        detections = [np.asarray([[1.0, 1.0, 20.0, 80.0, 0.9]], dtype=np.float32) for _ in frames]
        return detections, {}


class FakeEmbedder:
    def embed(self, crops: Sequence[np.ndarray]):
        return np.tile(
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            (len(crops), 1),
        ), 0.0


class FakeTracker:
    def __init__(self, camera_id: str) -> None:
        self.camera_id = camera_id

    def appearance_boxes(self, detections, frame_shape, detector_input_size):
        return np.asarray([[1.0, 1.0, 20.0, 80.0]], dtype=np.float32)

    def update(self, detections, frame, detector_input_size, features):
        return [
            TrackObservation(
                camera_id=self.camera_id,
                local_track_id=1,
                raw_track_id=1,
                bbox_xyxy=(1.0, 1.0, 20.0, 80.0),
                detection_confidence=0.9,
                track_age=1,
                appearance_embedding=features[0],
            )
        ]


def engine_config() -> dict:
    return {
        "site_id": "office",
        "runtime": {"target_fps": 10, "model_batch_size": 3},
        "detector": {},
        "tracker": {
            "track_high_threshold": 0.35,
            "track_low_threshold": 0.1,
            "new_track_threshold": 0.45,
            "track_buffer": 150,
            "match_threshold": 0.8,
            "proximity_threshold": 0.7,
            "appearance_threshold": 0.25,
            "minimum_box_area": 600,
        },
        "reid": {"minimum_crop_width": 1, "minimum_crop_height": 1},
        "global_identity": {
            "similarity_threshold": 0.7,
            "ambiguity_margin": 0.035,
            "active_timeout_seconds": 2.0,
            "gallery_retention_seconds": 1800,
        },
        "cameras": ["cam01", "cam02"],
    }


def test_engine_pads_model_batch_and_connects_camera_ids() -> None:
    detector = FakeDetector()
    engine = TrackingEngine(
        engine_config(),
        detector=detector,
        embedder=FakeEmbedder(),
        tracker_factory=lambda camera_id, _: FakeTracker(camera_id),
        identity_store=GlobalIdentityStore(**engine_config()["global_identity"]),
        warmup=False,
    )
    frame = np.zeros((100, 50, 3), dtype=np.uint8)
    results = engine.process_batch(
        [
            FramePacket("cam01", frame, 100),
            FramePacket("cam02", frame, 100),
        ]
    )

    assert detector.last_batch_size == 3
    assert results[0].tracks[0].global_id == results[1].tracks[0].global_id
    assert results[1].tracks[0].status == "reidentified"
    engine.close()


class FakeEngine:
    camera_ids = ("cam01", "cam02")

    def __init__(self) -> None:
        self.batches: list[list[FramePacket]] = []
        self.closed = False

    def process_batch(self, packets: Sequence[FramePacket]) -> list[FrameResult]:
        self.batches.append(list(packets))
        return [
            FrameResult("office", packet.camera_id, packet.timestamp_ms, (), 0.1)
            for packet in packets
        ]

    def summary(self) -> dict:
        return {"fake_batches": len(self.batches)}

    def close(self) -> None:
        self.closed = True


def service_config(target_fps: int = 20) -> Mapping:
    return {
        "cameras": ["cam01", "cam02"],
        "runtime": {"target_fps": target_fps, "result_queue_size": 8},
    }


def test_service_batches_cameras_and_rejects_stale_frames() -> None:
    fake = FakeEngine()
    service = TrackingService(service_config(), engine=fake)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    service.start()
    assert service.submit_frame("cam01", frame, 100)
    assert service.submit_frame("cam02", frame, 100)
    assert not service.submit_frame("cam01", frame, 100)

    results = {service.poll_result(timeout=2).camera_id for _ in range(2)}
    service.close()

    assert results == {"cam01", "cam02"}
    assert len(fake.batches) == 1
    assert service.summary()["rejected_stale_frames"] == 1
    assert fake.closed


def test_service_keeps_only_latest_pending_frame() -> None:
    fake = FakeEngine()
    service = TrackingService(service_config(target_fps=1), engine=fake)
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    service.start()
    service.submit_frame("cam01", frame, 100)
    service.submit_frame("cam01", frame, 200)
    service.close()

    result = service.poll_result(timeout=1)
    assert result.timestamp_ms == 200
    assert service.summary()["replaced_before_processing"] == 1
