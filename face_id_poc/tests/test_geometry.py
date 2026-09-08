import numpy as np
from camex_faceid.models import nms, quality
from camex_faceid.pipeline import association
from camex_faceid.pipeline import supported_heads


def test_one_face_cannot_identify_two_heads():
    heads = [{"bbox": [10, 10, 60, 80]}, {"bbox": [12, 10, 62, 80]}]
    faces = [{"bbox": np.array([20, 20, 50, 60])}]
    assert len(association(heads, faces)) == 0  # ambiguous, not an arbitrary winner


def test_exclusive_faces():
    heads = [{"bbox": [10, 10, 60, 80]}, {"bbox": [110, 10, 160, 80]}]
    faces = [
        {"bbox": np.array([20, 20, 50, 60])},
        {"bbox": np.array([120, 20, 150, 60])},
    ]
    assert association(heads, faces) == {0: 0, 1: 1}


def test_nms_duplicate_suppression():
    b = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [20, 20, 30, 30]], float)
    assert nms(b, np.array([0.9, 0.8, 0.7])) == [0, 2]


def test_nonsquare_face_width_is_x_extent():
    cfg = dict(
        face_min_width=40,
        face_min_intereye=14,
        face_min_sharpness=25,
        face_max_pose_asymmetry=0.65,
    )
    face = {
        "bbox": np.array([100, 30, 160, 120]),
        "landmarks": np.array(
            [[110, 50], [150, 50], [130, 75], [115, 100], [145, 100]]
        ),
    }
    q = quality(np.zeros((200, 300, 3), np.uint8), face, cfg)
    assert q["width_px"] == 60
    assert "blurred" in q["reasons"]


def test_two_faces_in_one_head_are_ambiguous():
    heads = [{"bbox": [0, 0, 100, 100]}]
    faces = [{"bbox": np.array([10, 20, 45, 70])}, {"bbox": np.array([55, 20, 90, 70])}]
    assert association(heads, faces) == {}
    assert association(heads, faces, return_ambiguous=True)[1] == {0}


def test_invalid_vertical_landmarks():
    cfg = dict(
        face_min_width=40,
        face_min_intereye=14,
        face_min_sharpness=0,
        face_max_pose_asymmetry=0.65,
    )
    f = {
        "bbox": np.array([100, 30, 160, 120]),
        "landmarks": np.array(
            [[110, 500], [150, 500], [130, 550], [115, 580], [145, 580]]
        ),
    }
    assert not quality(np.zeros((200, 300, 3), np.uint8), f, cfg)["usable"]


def test_background_head_removed_and_body_face_recovers_head():
    heads = np.array([[0, 0, 20, 20, 0.8]], np.float32)
    bodies = np.array([[100, 20, 200, 290, 0.9]], np.float32)
    faces = [{"bbox": np.array([120, 30, 160, 80]), "score": 0.95}]
    found = supported_heads(heads, bodies, faces)
    assert len(found) == 1 and found[0, 0] > 100


def test_recent_verified_head_survives_missing_body_detection():
    h = np.array([[100, 100, 160, 180, 0.8]], np.float32)
    assert len(supported_heads(h, np.empty((0, 5)), [], trusted_heads=[h[0, :4]])) == 1
    assert (
        len(supported_heads(h, np.empty((0, 5)), [], trusted_heads=[[0, 0, 10, 10]]))
        == 0
    )
    assert (
        len(
            supported_heads(
                np.empty((0, 5)), np.empty((0, 5)), [], trusted_heads=[h[0, :4]]
            )
        )
        == 0
    )
