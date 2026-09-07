import copy

import numpy as np
import pytest

from camex_floor.geometry import (
    Projector,
    calibration_fingerprint,
    calibration_report,
    fit_homography,
    transform,
    validate_site,
)
from camex_floor.runtime import FloorFilter, FloorTrackingService, footpoint, DEFAULTS


def site(ids=("north", "south"), scale=1):
    c = {
        "label": "Camera",
        "image_size_wh": [640, 480],
        "distortion": {"model": "none"},
        "image_to_floor_h": [[scale, 0, 0], [0, scale, 0], [0, 0, 1]],
        "anchors": [
            {"image_px": p, "map_px": (np.array(p) * scale).tolist(), "role": r}
            for p, r in [
                ([0, 0], "fit"),
                ([600, 0], "fit"),
                ([0, 400], "fit"),
                ([600, 400], "fit"),
                ([100, 100], "check"),
                ([450, 350], "check"),
            ]
        ],
    }
    s = {
        "schema_version": "1.0",
        "site_id": "test",
        "floor": {
            "asset": "floor.png",
            "size_wh": [640 * scale, 480 * scale],
            "pixels_per_metre": 80 * scale,
        },
        "cameras": {cid: copy.deepcopy(c) for cid in ids},
    }
    for v in s["cameras"].values():
        v["approved_fingerprint"] = calibration_fingerprint(v, s["floor"])
    return s


def packet(cid, ts, gid="p", x=200, embedding=None):
    return {
        "camera_id": cid,
        "timestamp_ms": ts,
        "frame_size_wh": [640, 480],
        "tracks": [
            {
                "local_id": 1,
                "global_id": gid,
                "bbox_xyxy": [x - 25, 100, x + 25, 300],
                "detection_confidence": 0.9,
                "embedding": embedding or [1, 0],
            }
        ],
    }


def test_arbitrary_camera_counts():
    for count in (1, 3, 20):
        ids = [f"sensor-{i + 100}" for i in range(count)]
        service = FloorTrackingService(site(ids))
        result = service.process_batch(
            [packet(cid, 0, str(i), 50 + i * 25) for i, cid in enumerate(ids)]
        )
        assert result["metrics"]["packets"] == count


def test_camera_rename_and_order_do_not_change_positions():
    a = FloorTrackingService(site(("a", "b")))
    b = FloorTrackingService(site(("renamed-y", "renamed-x")))
    for ts in range(0, 1200, 100):
        r1 = a.process_batch([packet("a", ts, "1", 200), packet("b", ts, "2", 201)])
        r2 = b.process_batch(
            [packet("renamed-x", ts, "1", 200), packet("renamed-y", ts, "2", 201)]
        )
    assert len(r1["tracks"]) == len(r2["tracks"]) == 1
    np.testing.assert_allclose(r1["tracks"][0]["floor_px"], r2["tracks"][0]["floor_px"])


def test_causal_prefix_has_no_future_dependence():
    a = FloorTrackingService(site(), session_id="same")
    b = FloorTrackingService(site(), session_id="same")
    prefix = [
        a.process_batch([packet("north", ts, "1", 200), packet("south", ts, "2", 201)])
        for ts in range(0, 800, 100)
    ]
    full = [
        b.process_batch([packet("north", ts, "1", 200), packet("south", ts, "2", 201)])
        for ts in range(0, 1600, 100)
    ]
    for x, y in zip(prefix, full):
        x.pop("processing_ms")
        y.pop("processing_ms")
        assert x == y


def test_no_merge_without_appearance():
    svc = FloorTrackingService(site())
    for ts in range(0, 4000, 100):
        group = [packet("north", ts, "1"), packet("south", ts, "2")]
        for p in group:
            p["tracks"][0].pop("embedding")
        result = svc.process_batch(group)
    assert len(result["tracks"]) == 2


def test_same_camera_collision_blocks_alias():
    svc = FloorTrackingService(site())
    p = packet("north", 0, "1")
    p["tracks"].append({**p["tracks"][0], "global_id": "2", "local_id": 2})
    svc.process_batch([p])
    for ts in range(100, 4000, 100):
        result = svc.process_batch([packet("north", ts, "1"), packet("south", ts, "2")])
    assert len(result["tracks"]) == 2


def test_invalid_calibration_fails_closed_for_every_camera():
    s = site()
    s["cameras"]["south"]["anchors"] = []
    with pytest.raises(ValueError):
        FloorTrackingService(s)
    assert not calibration_report(s["cameras"]["south"], 80)["ready"]
    FloorTrackingService(s, allow_draft=True)


def test_metric_scale_invariance():
    a = FloorTrackingService(site())
    b = FloorTrackingService(site(scale=2))
    for ts in range(0, 1200, 100):
        group = [
            packet("north", ts, "1", 200 + ts / 100),
            packet("south", ts, "2", 201 + ts / 100),
        ]
        r1 = a.process_batch(group)
        r2 = b.process_batch(group)
    assert len(r1["tracks"]) == len(r2["tracks"]) == 1
    np.testing.assert_allclose(
        r1["tracks"][0]["position_m"], r2["tracks"][0]["position_m"]
    )


def test_no_cross_time_merge_and_stale_packet_rejected():
    svc = FloorTrackingService(site())
    svc.process_batch([packet("north", 1000)])
    result = svc.process_batch([packet("south", 0, "2")])
    assert result["metrics"]["stale_packets"] == 1


def test_state_restore_preserves_identity_and_sequence(tmp_path):
    s = site()
    svc = FloorTrackingService(s)
    for ts in range(0, 1200, 100):
        svc.process_batch([packet("north", ts, "1"), packet("south", ts, "2")])
    path = tmp_path / "state.json"
    svc.save_state(path)
    other = FloorTrackingService(s)
    other.restore_state(path)
    assert other.resolve_global_id("2") == svc.resolve_global_id("2")
    assert other.session_id == svc.session_id and other.sequence == svc.sequence
    s["site_id"] = "different"
    with pytest.raises(ValueError):
        FloorTrackingService(s).restore_state(path)


def test_calibration_change_invalidates_approval():
    s = site()
    s["cameras"]["north"]["image_to_floor_h"][0][2] = 5
    with pytest.raises(ValueError):
        validate_site(s)


def test_aspect_ratio_rejected_and_equivalent_resize_matches():
    s = site()
    p = Projector(s["cameras"]["north"], s["floor"])
    np.testing.assert_allclose(
        p.project([200, 300], [640, 480]), p.project([400, 600], [1280, 960])
    )
    with pytest.raises(ValueError):
        p.project([200, 300], [1280, 720])


def test_robust_fit_rejects_wrong_click():
    pairs = [
        ([0, 0], [10, 20]),
        ([600, 0], [610, 20]),
        ([0, 400], [10, 420]),
        ([600, 400], [610, 420]),
        ([300, 200], [310, 220]),
        ([100, 100], [700, 800]),
    ]
    h, a = fit_homography(
        [{"image_px": p, "map_px": q, "role": "fit"} for p, q in pairs]
    )
    np.testing.assert_allclose(transform(h, [150, 130]), [160, 150], atol=1e-4)
    assert a["outlier_indices"] == [5]


def test_holdouts_not_used_in_fit_and_bad_check_rejected():
    s = site()
    c = s["cameras"]["north"]
    c["anchors"][-1]["map_px"] = [0, 0]
    h, _ = fit_homography(c["anchors"])
    np.testing.assert_allclose(h, np.eye(3), atol=1e-9)
    assert not calibration_report(c, 80)["ready"]


def test_degenerate_fit_and_duplicate_holdout_rejected():
    with pytest.raises(ValueError):
        fit_homography([{"image_px": [i, i], "map_px": [i, i]} for i in range(4)])
    c = site()["cameras"]["north"]
    c["anchors"][-1]["image_px"] = [0, 0]
    assert not calibration_report(c, 80)["ready"]


def test_footpoint_resolution_invariance():
    track = packet("a", 0)["tracks"][0]
    points = np.zeros((17, 2))
    conf = np.zeros(17)
    points[[5, 6]] = [[195, 150], [205, 150]]
    points[[11, 12]] = [[195, 180], [205, 180]]
    conf[[5, 6, 11, 12]] = 0.9
    track["keypoints_xy"] = points.tolist()
    track["keypoints_conf"] = conf.tolist()
    p1, _, _ = footpoint(track, [640, 480], DEFAULTS)
    track["bbox_xyxy"] = (np.array(track["bbox_xyxy"]) * 2).tolist()
    track["keypoints_xy"] = (points * 2).tolist()
    p2, _, _ = footpoint(track, [1280, 960], DEFAULTS)
    np.testing.assert_allclose(p2, p1 * 2)


def test_static_nonhuman_waits_and_real_seated_pose_confirms():
    svc = FloorTrackingService(site())
    p = packet("north", 0)
    p["tracks"][0]["detection_confidence"] = 0.49
    for ts in range(0, 6000, 100):
        p["timestamp_ms"] = ts
        r = svc.process_batch([p])
        assert not r["tracks"]
    p["timestamp_ms"] = 6000
    p["tracks"][0]["keypoints_conf"] = [0.8] * 17
    p["tracks"][0]["keypoints_xy"] = [[200, 200]] * 17
    for ts in range(6000, 7000, 100):
        p["timestamp_ms"] = ts
        r = svc.process_batch([p])
    assert len(r["tracks"]) == 1


def test_stationary_smoothing_reduces_jitter():
    rng = np.random.default_rng(42)
    truth = np.array([3.0, 5.0])
    raw = truth + rng.normal(0, 0.12, (100, 2))
    f = FloorFilter(DEFAULTS)
    smooth = np.array([f.update(i * 100, p, 0.7) for i, p in enumerate(raw)])
    assert smooth[20:].std(0).mean() < raw[20:].std(0).mean() * 0.8


def test_static_unverified_object_is_not_confirmed_by_moderate_confidence_ramp():
    svc = FloorTrackingService(site())
    for i in range(120):
        p = packet("north", i * 100)
        p["tracks"][0]["detection_confidence"] = 0.49 if i < 80 else 0.68
        assert not svc.process_batch([p])["tracks"]


def test_no_state_growth_after_expiration():
    svc = FloorTrackingService(site())
    svc.process_batch([packet("north", 0)])
    svc.process_batch(
        [
            {
                "camera_id": "north",
                "timestamp_ms": 2000000,
                "frame_size_wh": [640, 480],
                "tracks": [],
            }
        ]
    )
    assert not svc.identities and not svc.local and not svc.latest and not svc.filters


def test_aliases_site_scoped():
    svc = FloorTrackingService(site())
    with pytest.raises(ValueError):
        svc.apply_identity_alias(
            {
                "site_id": "wrong",
                "source_global_id": "1",
                "canonical_global_id": "2",
                "timestamp_ms": 0,
            }
        )
