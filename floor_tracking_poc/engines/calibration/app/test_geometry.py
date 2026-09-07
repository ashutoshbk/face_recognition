import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).with_name("calibrate.py")
SPEC = importlib.util.spec_from_file_location("camex_calibrate", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_rotation_to_z():
    normal = np.array([0.2, -0.8, 0.56])
    normal /= np.linalg.norm(normal)
    rotated = MODULE.rotation_to_z(normal) @ normal
    assert np.allclose(rotated, np.array([0.0, 0.0, 1.0]), atol=1e-7)


def test_forward_alignment():
    forward = np.array([1.0, 1.0, 0.2])
    rotated = MODULE.rotation_forward_to_up(forward) @ forward
    assert abs(rotated[0]) < 1e-7
    assert rotated[1] > 0


def test_camera_id_normalization():
    assert MODULE.safe_camera_id(" Camera 1488 / north ") == "Camera_1488_north"


def test_polygon_mask():
    mask = MODULE.polygon_mask([[0, 0], [1, 0], [1, 1], [0, 1]], 32, 48)
    assert mask.shape == (32, 48)
    assert mask.mean() > 0.95
