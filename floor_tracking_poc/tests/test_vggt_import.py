import json

import cv2
import numpy as np
import pytest

from camex_floor.vggt_import import import_vggt, read_cloud
from camex_floor.geometry import validate_site


def test_independent_vggt_camera_import_produces_unapproved_local_layer(tmp_path):
    engine = tmp_path / "engine"
    (engine / "rectified").mkdir(parents=True)
    cv2.imwrite(str(engine / "rectified/frame.png"), np.zeros((480, 640, 3), np.uint8))
    geocalib = {
        "camera_model": "simple_divisional",
        "focal_xy_pixels": [[400, 400]],
        "principal_point_xy_pixels": [[320, 240]],
        "distortion_parameters": [[0]],
    }
    (engine / "geocalib_calibration.json").write_text(json.dumps(geocalib))
    (engine / "reconstruction_metrics.json").write_text(
        json.dumps({"floor_plane_offset": 1, "floor_support_ratio": 0.9})
    )
    np.savez(
        engine / "reconstruction_arrays.npz",
        extrinsics=np.array([np.column_stack([np.eye(3), [0, 0, 0]])]),
        intrinsics=np.array([[[200, 0, 259], [0, 200, 196], [0, 0, 1]]]),
        frame_forwards=[[0, 0, 1]],
        frame_centers=[[0, 0, 0]],
        level_normal_world=[0, -1, 0],
        floor_normal_world=[0, -1, 0],
    )
    dtype = np.dtype(
        [(a, "<f4") for a in "xyz"] + [(a, "u1") for a in ("red", "green", "blue")]
    )
    v = np.zeros(100, dtype=dtype)
    x, y = np.meshgrid(np.linspace(-1, 1, 10), np.linspace(1, 3, 10))
    v["x"] = x.ravel()
    v["y"] = y.ravel()
    v["z"] = -1
    v["green"] = 200
    header = (
        "ply\nformat binary_little_endian 1.0\nelement vertex 100\n"
        + "".join("property float " + a + "\n" for a in "xyz")
        + "".join("property uchar " + a + "\n" for a in ("red", "green", "blue"))
        + "end_header\n"
    )
    (engine / "camera_relative_top_aligned.ply").write_bytes(
        header.encode() + v.tobytes()
    )
    site_path = tmp_path / "site.json"
    site_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "revision": 0,
                "site_id": "any-building",
                "floor": {
                    "asset": "floor.png",
                    "size_wh": [1000, 800],
                    "pixels_per_metre": 80,
                },
                "cameras": {},
            }
        )
    )
    result = import_vggt(site_path, "arbitrary_device", engine)
    assert result["point_count"] == 100
    site = json.loads(site_path.read_text())
    camera = site["cameras"]["arbitrary_device"]
    assert (tmp_path / camera["point_layer"]).is_file()
    assert np.isfinite(camera["image_to_floor_h"]).all()
    with pytest.raises(ValueError):
        validate_site(site)


def test_truncated_cloud_rejected(tmp_path):
    p = tmp_path / "bad.ply"
    p.write_text("ply\nformat binary_little_endian 1.0\n")
    with pytest.raises(ValueError):
        read_cloud(p)
