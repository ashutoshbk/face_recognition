from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

SCHEMA_VERSION = "1.0"
IDENTITY_EVENT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class FramePacket:
    camera_id: str
    frame: np.ndarray
    timestamp_ms: int

    def validate(self) -> None:
        if not self.camera_id:
            raise ValueError("camera_id must not be empty")
        if self.timestamp_ms < 0:
            raise ValueError("timestamp_ms must be non-negative")
        if not isinstance(self.frame, np.ndarray):
            raise TypeError("frame must be a NumPy array")
        if self.frame.dtype != np.uint8:
            raise TypeError("frame must use uint8 pixels")
        if self.frame.ndim != 3 or self.frame.shape[2] != 3:
            raise ValueError("frame must have BGR shape HxWx3")


@dataclass(frozen=True)
class TrackResult:
    local_id: int
    global_id: int | None
    bbox_xyxy: tuple[float, float, float, float]
    detection_confidence: float
    reid_confidence: float
    status: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_id": self.local_id,
            "global_id": self.global_id,
            "bbox_xyxy": [round(value, 1) for value in self.bbox_xyxy],
            "detection_confidence": round(self.detection_confidence, 4),
            "reid_confidence": round(self.reid_confidence, 4),
            "status": self.status,
        }


@dataclass(frozen=True)
class FrameResult:
    site_id: str
    camera_id: str
    timestamp_ms: int
    tracks: tuple[TrackResult, ...]
    processing_latency_ms: float
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "site_id": self.site_id,
            "camera_id": self.camera_id,
            "timestamp_ms": self.timestamp_ms,
            "processing_latency_ms": round(self.processing_latency_ms, 3),
            "tracks": [track.to_dict() for track in self.tracks],
        }


@dataclass(frozen=True)
class IdentityAliasEvent:
    """A runtime correction from a provisional global ID to its canonical ID."""

    site_id: str
    sequence: int
    timestamp_ms: int
    source_global_id: int
    canonical_global_id: int
    similarity: float
    transition_gap_s: float
    source_camera_id: str | None = None
    source_local_id: int | None = None
    target_camera_id: str | None = None
    target_local_id: int | None = None
    reason: str = "record_reconciled"
    schema_version: str = IDENTITY_EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "event": "global_id_alias",
            "site_id": self.site_id,
            "sequence": self.sequence,
            "timestamp_ms": self.timestamp_ms,
            "source_global_id": self.source_global_id,
            "canonical_global_id": self.canonical_global_id,
            "similarity": round(self.similarity, 4),
            "transition_gap_s": round(self.transition_gap_s, 3),
            "source_camera_id": self.source_camera_id,
            "source_local_id": self.source_local_id,
            "target_camera_id": self.target_camera_id,
            "target_local_id": self.target_local_id,
            "reason": self.reason,
        }
