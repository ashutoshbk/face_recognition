from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from camex_mct.global_identity import GlobalIdentityStore
from camex_mct.service import TrackingEngine, TrackingService
from camex_mct.tracking import TrackObservation
from camex_mct.types import FramePacket, FrameResult, IdentityAliasEvent


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


class StaticEmbeddingTracker(FakeTracker):
    def __init__(self, camera_id: str, embedding: np.ndarray) -> None:
        super().__init__(camera_id)
        self.embedding = embedding

    def update(self, detections, frame, detector_input_size, features):
        observation = super().update(detections, frame, detector_input_size, features)[0]
        observation.appearance_embedding = self.embedding
        return [observation]


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


def test_engine_emits_generic_alias_after_three_camera_reconciliation() -> None:
    config = engine_config()
    config["cameras"] = ["reception", "workspace_1", "workspace_2"]
    config["global_identity"] = {
        **config["global_identity"],
        "similarity_threshold": 0.80,
        "record_reconciliation_enabled": True,
        "record_reconciliation_interval_seconds": 0.0,
        "record_reconciliation_similarity_threshold": 0.72,
        "record_reconciliation_margin": 0.04,
        "record_reconciliation_max_transition_seconds": 10.0,
        "record_reconciliation_min_observations": 1,
    }
    camera_embeddings = {
        "reception": np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        "workspace_1": np.asarray([0.65, 0.7599342, 0.0], dtype=np.float32),
        "workspace_2": np.asarray([0.74, 0.6726069, 0.0], dtype=np.float32),
    }
    engine = TrackingEngine(
        config,
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
        tracker_factory=lambda camera_id, _: StaticEmbeddingTracker(
            camera_id, camera_embeddings[camera_id]
        ),
        identity_store=GlobalIdentityStore(**config["global_identity"]),
        warmup=False,
    )
    frame = np.zeros((100, 50, 3), dtype=np.uint8)

    first = engine.process_batch([FramePacket("reception", frame, 0)])
    second = engine.process_batch([FramePacket("workspace_1", frame, 2000)])
    third = engine.process_batch([FramePacket("workspace_2", frame, 4000)])
    updates = engine.drain_identity_updates()

    first_id = first[0].tracks[0].global_id
    established_id = second[0].tracks[0].global_id
    assert first_id != established_id
    assert third[0].tracks[0].global_id == established_id
    assert engine.resolve_global_id(first_id) == established_id
    assert len(updates) == 1
    assert updates[0].source_global_id == first_id
    assert updates[0].canonical_global_id == established_id
    assert updates[0].source_camera_id == "reception"
    assert updates[0].target_camera_id in {"workspace_1", "workspace_2"}
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

    def drain_identity_updates(self) -> tuple[IdentityAliasEvent, ...]:
        return ()

    def resolve_global_id(self, global_id: int) -> int:
        return global_id

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


class FakeAliasEngine(FakeEngine):
    def __init__(self) -> None:
        super().__init__()
        self.updates: list[IdentityAliasEvent] = []

    def process_batch(self, packets: Sequence[FramePacket]) -> list[FrameResult]:
        results = super().process_batch(packets)
        self.updates.append(
            IdentityAliasEvent(
                site_id="office",
                sequence=1,
                timestamp_ms=packets[0].timestamp_ms,
                source_global_id=18,
                canonical_global_id=19,
                similarity=0.81,
                transition_gap_s=1.2,
            )
        )
        return results

    def drain_identity_updates(self) -> tuple[IdentityAliasEvent, ...]:
        updates = tuple(self.updates)
        self.updates.clear()
        return updates

    def resolve_global_id(self, global_id: int) -> int:
        return 19 if global_id == 18 else global_id


def test_service_publishes_runtime_identity_alias_and_resolves_old_id() -> None:
    fake = FakeAliasEngine()
    callback_updates: list[IdentityAliasEvent] = []
    service = TrackingService(
        service_config(target_fps=100),
        engine=fake,
        on_identity_update=callback_updates.append,
    )
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    service.start()
    service.submit_frame("cam01", frame, 100)

    update = service.poll_identity_update(timeout=2)
    result = service.poll_result(timeout=2)
    service.close()

    assert result.timestamp_ms == 100
    assert update.source_global_id == 18
    assert update.canonical_global_id == 19
    assert callback_updates == [update]
    assert service.resolve_global_id(18) == 19
