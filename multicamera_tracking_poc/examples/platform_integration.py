"""Minimal example for a platform that already decodes its camera streams."""

from __future__ import annotations

import json

from camex_mct import FrameResult, TrackingService


def publish_to_platform(result: FrameResult) -> None:
    print(json.dumps(result.to_dict(), separators=(",", ":")))


service = TrackingService.from_yaml(
    "configs/live_20cam.example.yaml",
    on_result=publish_to_platform,
)
service.start()


def on_decoded_frame(camera_id, bgr_frame, timestamp_ms) -> bool:
    """Call this function from the platform's existing decoder callback."""
    return service.submit_frame(camera_id, bgr_frame, timestamp_ms)


def shutdown() -> None:
    service.close()
