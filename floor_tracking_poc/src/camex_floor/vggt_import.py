"""Import one independently reconstructed camera into the floor editor."""

import hashlib
import json
from pathlib import Path
import shutil
import uuid

import cv2
import numpy as np

from .geometry import transform
from .server import atomic_json


def rotation_to_z(normal):
    n = np.asarray(normal, float)
    n = n / np.linalg.norm(n)
    z = np.array([0.0, 0.0, 1.0])
    v = np.cross(n, z)
    s = np.linalg.norm(v)
    c = np.clip(n @ z, -1, 1)
    if s < 1e-9:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    x, y, z = v / s
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + k * s + k @ k * (1 - c)


def read_cloud(path):
    with Path(path).open("rb") as f:
        header = []
        for _ in range(100):
            line = f.readline().decode("ascii").strip()
            header.append(line)
            if line == "end_header":
                break
        else:
            raise ValueError("Invalid PLY header")
        if "format binary_little_endian 1.0" not in header:
            raise ValueError("Unsupported PLY format")
        expected = ["property float " + a for a in "xyz"] + [
            "property uchar " + a for a in ("red", "green", "blue")
        ]
        if [x for x in header if x.startswith("property ")] != expected:
            raise ValueError("Unsupported PLY properties")
        count = int(
            next(x.split()[-1] for x in header if x.startswith("element vertex "))
        )
        values = np.fromfile(
            f,
            dtype=np.dtype(
                [(a, "<f4") for a in "xyz"]
                + [(a, "u1") for a in ("red", "green", "blue")]
            ),
            count=count,
        )
    if len(values) != count:
        raise ValueError("Truncated cloud")
    return np.column_stack([values[a] for a in "xyz"]).astype(float), np.column_stack(
        [values[a] for a in ("red", "green", "blue")]
    )


def import_vggt(site_path, camera_id, directory, allow_provisional=False):
    site_path = Path(site_path)
    directory = Path(directory)
    site = json.loads(site_path.read_text())
    if camera_id in site["cameras"]:
        raise ValueError("Camera ID already exists")
    reference = sorted((directory / "rectified").glob("*.png"))
    if len(reference) != 1:
        raise ValueError(
            "Import one independently reconstructed reference frame per camera"
        )
    geocalib = json.loads((directory / "geocalib_calibration.json").read_text())
    metrics = json.loads((directory / "reconstruction_metrics.json").read_text())
    if geocalib["camera_model"] != "simple_divisional":
        raise ValueError("Unsupported GeoCalib lens model")
    provisional = (
        metrics.get("floor_plane_offset") is None
        or (metrics.get("floor_support_ratio") or 0) < 0.10
    )
    disagreement = (metrics.get("floor_gravity_angle_degrees") or 0) > 30
    use_gravity = (
        provisional
        or metrics.get("level_method_requested") == "gravity"
        or (disagreement and metrics.get("level_method_requested") != "floor")
    )
    provisional = provisional or disagreement
    if provisional and not allow_provisional:
        raise ValueError(
            "A supported floor polygon/plane is required for the point-layer importer; rerun the engine with --floor-polygon, or register the rectified image for direct landmark fitting"
        )
    with np.load(directory / "reconstruction_arrays.npz") as arrays:
        if len(arrays["extrinsics"]) != 1:
            raise ValueError("This importer requires one independent camera frame")
        level = rotation_to_z(arrays["level_normal_world"])
        forward = level @ arrays["frame_forwards"][0]
        angle = np.pi / 2 - np.arctan2(forward[1], forward[0])
        c, s = np.cos(angle), np.sin(angle)
        yaw = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
        world_to_top = yaw @ level
        camera_to_top = world_to_top @ arrays["extrinsics"][0, :, :3].T
        if use_gravity:
            cloud, _ = read_cloud(directory / "camera_relative_top_aligned.ply")
            gravity_world = (
                arrays["gravity_world"][0]
                if "gravity_world" in arrays
                else arrays["level_normal_world"]
            )
            n = world_to_top @ gravity_world
            n = n / np.linalg.norm(n)
            d = -float(np.percentile(cloud @ n, 15))
            if not np.isfinite(d) or d <= 1e-6:
                d = 1.0
        else:
            n = world_to_top @ arrays["floor_normal_world"]
            n = n / np.linalg.norm(n)
            d = float(
                arrays["floor_normal_world"] @ arrays["frame_centers"][0]
                + metrics["floor_plane_offset"]
            )
        k = arrays["intrinsics"][0].astype(float)
    if d <= 1e-6:
        raise ValueError("Ground plane does not lie below the camera")
    image = cv2.imread(str(reference[0]))
    height, width = image.shape[:2]
    resized_height = round(height * 518 / width / 14) * 14
    crop = max(0, (resized_height - 518) // 2)
    resize = np.array(
        [[518 / width, 0, 0], [0, resized_height / height, -crop], [0, 0, 1.0]]
    )
    k = np.linalg.inv(resize) @ k
    forward = np.array([0.0, 1.0, 0.0])
    forward -= n * (n @ forward)
    if np.linalg.norm(forward) < 1e-6:
        raise ValueError("Degenerate ground orientation")
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, n)
    right /= np.linalg.norm(right)
    basis = np.vstack([right, forward, n])
    origin = -d * n
    points, colors = read_cloud(directory / "camera_relative_top_aligned.ply")
    ground = (points - origin) @ basis.T / d
    lo, hi = np.percentile(ground[:, :2], [1, 99], axis=0)
    if np.min(hi - lo) < 1e-6:
        raise ValueError("Degenerate point-cloud footprint")
    scale = 1000 / np.max(hi - lo)
    w, h = np.ceil((hi - lo) * scale).astype(int) + 1
    raster = np.array(
        [[scale, 0, -lo[0] * scale], [0, -scale, hi[1] * scale], [0, 0, 1.0]]
    )
    indices = np.round(transform(raster, ground[:, :2])).astype(int)
    valid = (
        (indices[:, 0] >= 0)
        & (indices[:, 0] < w)
        & (indices[:, 1] >= 0)
        & (indices[:, 1] < h)
    )
    layer = np.zeros((h, w, 4), np.uint8)
    layer[indices[valid, 1], indices[valid, 0], :3] = colors[valid, ::-1]
    layer[indices[valid, 1], indices[valid, 0], 3] = 220
    layer = cv2.dilate(layer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    floor_layer = np.zeros_like(layer)
    floor_valid = valid & (np.abs(ground[:, 2]) <= 0.15)
    floor_layer[indices[floor_valid, 1], indices[floor_valid, 0], :3] = colors[
        floor_valid, ::-1
    ]
    floor_layer[indices[floor_valid, 1], indices[floor_valid, 0], 3] = 230
    floor_layer = cv2.dilate(
        floor_layer, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    )
    ground_to_image = (
        k @ camera_to_top.T @ np.column_stack([d * basis[0], d * basis[1], origin])
    )
    layer_to_image = ground_to_image @ np.linalg.inv(raster)
    assets = site_path.parent / "assets"
    assets.mkdir(exist_ok=True)
    prefix = uuid.uuid4().hex
    ref_name = f"{prefix}_reference.png"
    cloud_name = f"{prefix}_points.png"
    floor_name = f"{prefix}_floor_points.png"
    shutil.copy2(reference[0], assets / ref_name)
    cv2.imwrite(str(assets / cloud_name), layer)
    cv2.imwrite(str(assets / floor_name), floor_layer)
    floor_size = np.asarray(site["floor"]["size_wh"])
    rough_scale = min(floor_size / [w, h]) * 0.4
    placement = np.array(
        [
            [rough_scale, 0, (floor_size[0] - w * rough_scale) / 2],
            [0, rough_scale, (floor_size[1] - h * rough_scale) / 2],
            [0, 0, 1.0],
        ]
    )
    site["cameras"][camera_id] = {
        "label": camera_id,
        "image_size_wh": [width, height],
        "reference": "assets/" + ref_name,
        "reference_sha256": hashlib.sha256(
            (assets / ref_name).read_bytes()
        ).hexdigest(),
        "point_layer": "assets/" + floor_name,
        "all_point_layer": "assets/" + cloud_name,
        "layer_to_image_h": layer_to_image.tolist(),
        "image_to_floor_h": (placement @ np.linalg.inv(layer_to_image)).tolist(),
        "anchors": [],
        "local_geometry_status": "provisional"
        if provisional
        else "floor_plane_estimated",
        "camera_to_local_rotation": camera_to_top.tolist(),
        "gravity_camera": geocalib.get("gravity_camera", [None])[0],
        "camera_to_floor_R": (basis @ camera_to_top).tolist(),
        "camera_to_floor_t": (-basis @ origin).tolist(),
        "local_floor_normal": n.tolist(),
        "local_floor_offset": d,
        "local_pose_units": "VGGT arbitrary scale",
        "floor_gravity_disagreement": disagreement,
        "local_plane_source": "gravity-constrained preview"
        if use_gravity
        else "selected floor-plane estimate",
        "distortion": {
            "model": "simple_divisional",
            "focal_xy": geocalib["focal_xy_pixels"][0],
            "principal_xy": geocalib["principal_point_xy_pixels"][0],
            "k1": geocalib["distortion_parameters"][0][0],
        },
        "calibration_source": "VGGT local point layer; initial map placement is unapproved",
    }
    site["revision"] = site.get("revision", 0) + 1
    atomic_json(site_path, site)
    return {
        "camera_id": camera_id,
        "point_count": len(points),
        "layer_size_wh": [int(w), int(h)],
        "status": "draft; global landmarks and holdout approval required",
    }
