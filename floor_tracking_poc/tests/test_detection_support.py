import numpy as np
from camex_floor.detection_support import merge_pose_support


def test_strong_pose_recovers_a_missing_person_at_correct_scale():
    out = merge_pose_support(
        np.empty((0, 5)),
        [[100, 100, 180, 300]],
        [0.9],
        [[0.8] * 17],
        [1280, 720],
        [416, 416],
    )
    np.testing.assert_allclose(out[0, :4], np.array([100, 100, 180, 300]) * 416 / 1280)


def test_pose_support_does_not_duplicate_an_existing_detection():
    out = merge_pose_support(
        [[32.5, 32.5, 58.5, 97.5, 0.3]],
        [[100, 100, 180, 300]],
        [0.9],
        [[0.8] * 17],
        [1280, 720],
        [416, 416],
    )
    assert len(out) == 1 and out[0, 4] > 0.8


def test_weak_or_unstructured_pose_does_not_add_a_person():
    for score, joints in [(0.3, [0.9] * 17), (0.9, [0.1] * 17)]:
        assert (
            len(
                merge_pose_support(
                    [],
                    [[100, 100, 180, 300]],
                    [score],
                    [joints],
                    [1280, 720],
                    [416, 416],
                )
            )
            == 0
        )
