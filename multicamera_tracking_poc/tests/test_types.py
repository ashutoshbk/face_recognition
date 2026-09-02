import numpy as np
import pytest

from camex_mct.types import FramePacket, FrameResult, IdentityAliasEvent, TrackResult


def test_frame_packet_requires_bgr_uint8() -> None:
    valid = FramePacket("cam01", np.zeros((10, 20, 3), dtype=np.uint8), 1)
    valid.validate()

    with pytest.raises(TypeError):
        FramePacket("cam01", np.zeros((10, 20, 3), dtype=np.float32), 1).validate()
    with pytest.raises(ValueError):
        FramePacket("cam01", np.zeros((10, 20), dtype=np.uint8), 1).validate()


def test_result_serialization_is_versioned_and_json_safe() -> None:
    result = FrameResult(
        site_id="office",
        camera_id="cam01",
        timestamp_ms=100,
        tracks=(
            TrackResult(
                local_id=2,
                global_id=7,
                bbox_xyxy=(1.25, 2.5, 30.75, 80.0),
                detection_confidence=0.91234,
                reid_confidence=0.83456,
                status="reidentified",
            ),
        ),
        processing_latency_ms=4.5678,
    ).to_dict()

    assert result["schema_version"] == "1.0"
    assert result["tracks"][0]["global_id"] == 7
    assert result["tracks"][0]["bbox_xyxy"] == [1.2, 2.5, 30.8, 80.0]


def test_identity_alias_serialization_is_versioned_and_json_safe() -> None:
    event = IdentityAliasEvent(
        site_id="office",
        sequence=3,
        timestamp_ms=89000,
        source_global_id=18,
        canonical_global_id=19,
        similarity=0.720234,
        transition_gap_s=6.812,
        source_camera_id="reception",
        source_local_id=12,
        target_camera_id="workspace_1",
        target_local_id=10,
    ).to_dict()

    assert event["event"] == "global_id_alias"
    assert event["source_global_id"] == 18
    assert event["canonical_global_id"] == 19
    assert event["similarity"] == 0.7202
