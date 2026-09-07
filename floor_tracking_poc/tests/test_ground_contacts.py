import numpy as np
from camex_floor.geometry import Projector
from camex_floor.runtime import DEFAULTS, FloorFilter, footpoint, FloorTrackingService
from camex_floor.detection_support import upper_body_agrees
from test_runtime import site, packet


def test_upper_pose_outside_a_chair_box_is_not_assigned_to_the_chair():
    p = np.zeros((17, 2))
    c = np.zeros(17)
    p[0] = [120, 70]
    p[5] = [110, 100]
    p[6] = [135, 100]
    c[[0, 5, 6]] = 0.9
    assert not upper_body_agrees([100, 160, 160, 280], p, c)
    assert upper_body_agrees([95, 60, 165, 300], p, c)


def test_pose_backend_without_ground_support_does_not_project_box_bottom():
    t = packet("a", 0)["tracks"][0]
    t["pose_available"] = True
    point, _, method = footpoint(t, [640, 480], DEFAULTS)
    assert point is None and method == "ground_contact_unobserved"


def test_gravity_extrapolation_follows_vertical_not_a_leaning_torso():
    camera = {
        "image_size_wh": [640, 480],
        "gravity_camera": [0, -1, 0],
        "distortion": {
            "model": "simple_divisional",
            "focal_xy": [600, 600],
            "principal_xy": [320, 240],
            "k1": 0,
        },
        "image_to_floor_h": np.eye(3).tolist(),
    }
    p = Projector(camera, {"size_wh": [640, 480]})
    estimate = p.gravity_foot([200, 120], [240, 180], 3, [640, 480])
    assert abs(estimate[0] - 240) < 1e-8 and estimate[1] > 180


def test_implausible_jump_starts_a_new_segment_instead_of_a_long_trail():
    cfg = {**DEFAULTS, "reject_trajectory_jumps": 1}
    f = FloorFilter(cfg)
    f.update(0, [1, 1], 0.9)
    f.update(100, [1.05, 1], 0.9)
    result = f.update(200, [10, 10], 0.9)
    assert f.discontinuity
    np.testing.assert_allclose(result, [10, 10])


def test_stationary_no_pose_track_loses_human_confirmation():
    svc = FloorTrackingService(site())
    p = packet("north", 0)
    p["tracks"][0]["pose_available"] = True
    for ts in range(0, 2000, 100):
        p["timestamp_ms"] = ts
        r = svc.process_batch([p])
    assert r["observations"][0]["status"] == "unverified_static"
