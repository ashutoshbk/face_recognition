#!/usr/bin/env python3
"""Single-camera GeoCalib undistortion and VGGT top-view reconstruction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms.functional import to_pil_image


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = PACKAGE_ROOT / "models/VGGT-1B-Commercial/model.safetensors"
DEFAULT_GEOCALIB_WEIGHTS = PACKAGE_ROOT / "assets/geocalib/distorted.tar"
EXPECTED_VGGT_SHA256 = "2b766b284359bc47ce26be107254621f685b758a0282082ff109f3ff02788b53"
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Undistort one camera view and generate a camera-relative top view."
    )
    parser.add_argument("--input", type=Path, help="Image file or directory with 1-6 images")
    parser.add_argument("--output", type=Path, help="Output directory")
    parser.add_argument("--camera-id", help="Stable camera identifier used in output metadata")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--geocalib-weights", type=Path, default=DEFAULT_GEOCALIB_WEIGHTS)
    parser.add_argument(
        "--camera-model",
        choices=["simple_divisional", "simple_radial", "radial", "pinhole"],
        default="simple_divisional",
    )
    parser.add_argument(
        "--level-method",
        choices=["auto", "gravity", "floor"],
        default="auto",
        help="Auto uses GeoCalib gravity unless a reliable floor fit strongly disagrees.",
    )
    parser.add_argument(
        "--floor-polygon",
        type=Path,
        help="Optional JSON file containing normalized floor polygon points.",
    )
    parser.add_argument("--confidence-percentile", type=float, default=55.0)
    parser.add_argument("--max-points", type=int, default=600_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def safe_camera_id(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("_.-")
    if not normalized:
        raise ValueError("camera ID is empty after normalization")
    return normalized[:96]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_list(value) -> list:
    if hasattr(value, "vec3d"):
        value = value.vec3d
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    elif hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value).tolist()


def collect_images(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported image type: {path.suffix}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(
        item for item in path.iterdir() if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not images:
        raise FileNotFoundError(f"No supported images found in {path}")
    if len(images) > 6:
        raise ValueError(f"Expected at most 6 images from one camera, found {len(images)}")
    return images


def input_fingerprint(paths: list[Path], configuration: dict) -> str:
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    digest.update(json.dumps(configuration, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def load_state_dict(path: Path) -> dict:
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        state = load_file(str(path), device="cpu")
    else:
        state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        for key in ("state_dict", "model"):
            if key in state and isinstance(state[key], dict):
                return state[key]
    return state


def rotation_to_z(normal: np.ndarray) -> np.ndarray:
    normal = normal.astype(np.float64)
    normal /= max(np.linalg.norm(normal), 1e-12)
    target = np.array([0.0, 0.0, 1.0])
    cross = np.cross(normal, target)
    sine = np.linalg.norm(cross)
    cosine = float(np.clip(normal @ target, -1.0, 1.0))
    if sine < 1e-10:
        return np.eye(3) if cosine > 0 else np.diag([1.0, -1.0, -1.0])
    axis = cross / sine
    skew = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + skew * sine + (skew @ skew) * (1.0 - cosine)


def rotation_forward_to_up(forward: np.ndarray) -> np.ndarray:
    horizontal = forward[:2]
    if np.linalg.norm(horizontal) < 1e-10:
        return np.eye(3)
    current = math.atan2(horizontal[1], horizontal[0])
    angle = math.pi / 2.0 - current
    cosine, sine = math.cos(angle), math.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]]
    )


def camera_centers_and_forwards(extrinsics: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    centers, forwards = [], []
    for extrinsic in extrinsics:
        rotation = extrinsic[:3, :3]
        translation = extrinsic[:3, 3]
        centers.append(-rotation.T @ translation)
        forwards.append(rotation.T @ np.array([0.0, 0.0, 1.0]))
    return np.asarray(centers), np.asarray(forwards)


def robust_plot_bounds(points_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    low, high = np.percentile(points_xy, [1, 99], axis=0)
    low = np.minimum(low, np.zeros(2))
    high = np.maximum(high, np.zeros(2))
    center = (low + high) / 2.0
    span = max(float(np.max(high - low)), 1e-4) * 1.12
    return center - span / 2.0, center + span / 2.0


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T.astype(np.float32)
    vertices["red"], vertices["green"], vertices["blue"] = colors.T.astype(np.uint8)
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with path.open("wb") as handle:
        handle.write(header.encode("ascii"))
        vertices.tofile(handle)


def load_polygon(path: Path | None, camera_id: str) -> list[list[float]] | None:
    if path is None:
        return None
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "polygon" in payload:
        polygon = payload["polygon"]
    elif isinstance(payload, dict) and "polygons" in payload:
        polygons = payload["polygons"]
        polygon = polygons.get(camera_id, polygons.get("default"))
    elif isinstance(payload, list):
        polygon = payload
    else:
        raise ValueError("Floor polygon JSON must contain `polygon`, `polygons`, or a point list")
    if not isinstance(polygon, list) or len(polygon) < 3:
        raise ValueError(f"No valid floor polygon found for camera {camera_id}")
    normalized = []
    for point in polygon:
        if not isinstance(point, list) or len(point) != 2:
            raise ValueError(f"Invalid floor polygon point: {point}")
        x, y = map(float, point)
        if not 0.0 <= x <= 1.0 or not 0.0 <= y <= 1.0:
            raise ValueError(f"Floor polygon values must be normalized to [0,1]: {point}")
        normalized.append([x, y])
    return normalized


def polygon_mask(polygon: list[list[float]], height: int, width: int) -> np.ndarray:
    points = [(round(x * (width - 1)), round(y * (height - 1))) for x, y in polygon]
    mask = Image.new("L", (width, height), 0)
    ImageDraw.Draw(mask).polygon(points, fill=255)
    return np.asarray(mask) > 0


def fit_plane_ransac(
    points: np.ndarray, rng: np.random.Generator, iterations: int = 1200
) -> tuple[np.ndarray, float, np.ndarray, float, int]:
    if len(points) < 200:
        raise RuntimeError("Too few valid points inside the floor polygon")
    if len(points) > 120_000:
        points = points[rng.choice(len(points), 120_000, replace=False)]
    center = np.median(points, axis=0)
    scene_scale = np.median(np.linalg.norm(points - center, axis=1))
    threshold = max(float(scene_scale) * 0.012, 1e-5)
    best_mask = np.zeros(len(points), dtype=bool)
    best_normal = None
    for _ in range(iterations):
        sample = points[rng.choice(len(points), 3, replace=False)]
        normal = np.cross(sample[1] - sample[0], sample[2] - sample[0])
        norm = np.linalg.norm(normal)
        if norm < 1e-10:
            continue
        normal /= norm
        offset = -normal @ sample[0]
        mask = np.abs(points @ normal + offset) < threshold
        if mask.sum() > best_mask.sum():
            best_mask = mask
            best_normal = normal
    if best_normal is None or int(best_mask.sum()) < 100:
        raise RuntimeError("A stable floor plane could not be estimated")
    inliers = points[best_mask]
    centroid = inliers.mean(axis=0)
    _, _, vh = np.linalg.svd(inliers - centroid, full_matrices=False)
    normal = vh[-1]
    offset = float(-normal @ centroid)
    refined = np.abs(points @ normal + offset) < threshold
    return normal, offset, points[refined], threshold, len(points)


def gravity_consensus(gravity_world: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gravity_world = gravity_world / np.maximum(
        np.linalg.norm(gravity_world, axis=1, keepdims=True), 1e-12
    )
    if len(gravity_world) == 1:
        return gravity_world[0], np.array([0.0])
    similarities = np.clip(gravity_world @ gravity_world.T, -1.0, 1.0)
    angles = np.degrees(np.arccos(similarities))
    medoid = int(np.argmin(np.median(angles, axis=1)))
    inliers = angles[medoid] <= 20.0
    if int(inliers.sum()) < max(1, math.ceil(len(gravity_world) * 0.6)):
        inliers[:] = True
    consensus = gravity_world[inliers].mean(axis=0)
    consensus /= max(np.linalg.norm(consensus), 1e-12)
    deviations = np.degrees(
        np.arccos(np.clip(gravity_world @ consensus, -1.0, 1.0))
    )
    return consensus, deviations


def save_before_after(path: Path, original: Path, rectified: Path, camera_id: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#f4f6f7")
    axes[0].imshow(Image.open(original))
    axes[0].set_title("Original", fontsize=14, weight="bold")
    axes[1].imshow(Image.open(rectified))
    axes[1].set_title("GeoCalib undistorted", fontsize=14, weight="bold")
    for axis in axes:
        axis.axis("off")
    fig.suptitle(camera_id, fontsize=16, weight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_camera(axis, low: np.ndarray, high: np.ndarray, horizontal_fov: float, name: str) -> None:
    extent = max(float(np.max(high - low)), 1e-4)
    cone_length = extent * 0.30
    half_angle = float(np.clip(horizontal_fov / 2.0, np.deg2rad(15), np.deg2rad(55)))
    half_width = math.tan(half_angle) * cone_length
    cone = np.array([[0.0, 0.0], [-half_width, cone_length], [half_width, cone_length]])
    color = "#d1495b"
    axis.fill(cone[:, 0], cone[:, 1], color=color, alpha=0.13, zorder=3)
    axis.plot([0, cone[1, 0]], [0, cone[1, 1]], color=color, linestyle="--", linewidth=1.5)
    axis.plot([0, cone[2, 0]], [0, cone[2, 1]], color=color, linestyle="--", linewidth=1.5)
    axis.scatter(0, 0, s=230, marker="^", color=color, edgecolor="white", linewidth=1.5, zorder=6)
    axis.arrow(
        0,
        0,
        0,
        extent * 0.13,
        width=extent * 0.006,
        head_width=extent * 0.04,
        color=color,
        length_includes_head=True,
        zorder=5,
    )
    axis.text(0, -extent * 0.035, name, ha="center", va="top", color=color, weight="bold")


def plot_top_view(axis, points: np.ndarray, colors: np.ndarray, horizontal_fov: float, camera_id: str):
    rng = np.random.default_rng(42)
    if len(points) > 220_000:
        keep = rng.choice(len(points), 220_000, replace=False)
        points, colors = points[keep], colors[keep]
    low, high = robust_plot_bounds(points[:, :2])
    axis.scatter(
        points[:, 0], points[:, 1], c=colors / 255.0, s=0.6, alpha=0.58, rasterized=True
    )
    draw_camera(axis, low, high, horizontal_fov, camera_id)
    axis.set_xlim(low[0], high[0])
    axis.set_ylim(low[1], high[1])
    axis.set_aspect("equal", adjustable="box")
    return low, high


def save_top_view(
    path: Path, points: np.ndarray, colors: np.ndarray, horizontal_fov: float, camera_id: str
) -> None:
    fig, axis = plt.subplots(figsize=(12, 12), facecolor="none")
    axis.set_facecolor("none")
    plot_top_view(axis, points, colors, horizontal_fov, camera_id)
    axis.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(path, dpi=200, transparent=True, bbox_inches=None, pad_inches=0)
    plt.close(fig)


def save_review(
    path: Path,
    original: Path,
    rectified: Path,
    points: np.ndarray,
    colors: np.ndarray,
    horizontal_fov: float,
    camera_id: str,
    level_source: str,
    floor_support: float | None,
    floor_gravity_angle: float | None,
) -> None:
    fig = plt.figure(figsize=(16, 11), facecolor="#f5f7f8")
    grid = fig.add_gridspec(2, 2, height_ratios=[0.56, 1.44], hspace=0.24, wspace=0.06)
    for column, (title, image_path) in enumerate(
        (("Original", original), ("GeoCalib undistorted", rectified))
    ):
        axis = fig.add_subplot(grid[0, column])
        axis.imshow(Image.open(image_path))
        axis.set_title(title, weight="bold")
        axis.axis("off")
    axis = fig.add_subplot(grid[1, :])
    plot_top_view(axis, points, colors, horizontal_fov, camera_id)
    axis.set_facecolor("white")
    axis.grid(color="#d9e0e4", linewidth=0.5, alpha=0.65)
    axis.set_xlabel("Camera-relative X (arbitrary scale)")
    axis.set_ylabel("Camera forward direction")
    details = [f"Level source: {level_source}"]
    if floor_support is not None:
        details.append(f"floor support {floor_support:.3f}")
    if floor_gravity_angle is not None:
        details.append(f"floor/gravity angle {floor_gravity_angle:.1f} deg")
    axis.set_title(" | ".join(details), fontsize=15, weight="bold", pad=12)
    fig.suptitle(
        f"{camera_id} | GeoCalib + VGGT single-camera top view\n"
        "Camera is at the origin and points upward; scale remains relative",
        fontsize=15,
        weight="bold",
        y=0.98,
    )
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def save_floor_mask_review(path: Path, image: Path, polygon: list[list[float]]) -> None:
    with Image.open(image).convert("RGB") as source:
        draw = ImageDraw.Draw(source, "RGBA")
        points = [(round(x * (source.width - 1)), round(y * (source.height - 1))) for x, y in polygon]
        draw.polygon(points, fill=(0, 229, 255, 40), outline=(0, 210, 235, 255), width=5)
        source.save(path)


def self_test(args: argparse.Namespace) -> None:
    checks = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "geocalib_weights": str(args.geocalib_weights.resolve()),
        "geocalib_weights_present": args.geocalib_weights.is_file(),
        "vggt_checkpoint": str(args.checkpoint.resolve()),
        "vggt_checkpoint_present": args.checkpoint.is_file(),
    }
    if torch.cuda.is_available():
        checks["gpu"] = torch.cuda.get_device_name(0)
        checks["gpu_memory_bytes"] = torch.cuda.get_device_properties(0).total_memory
    try:
        from geocalib import GeoCalib  # noqa: F401
        from vggt.models.vggt import VGGT  # noqa: F401

        checks["imports"] = "ok"
    except Exception as error:
        checks["imports"] = f"failed: {error}"
    print(json.dumps(checks, indent=2))
    required = checks["cuda_available"] and checks["geocalib_weights_present"] and checks["imports"] == "ok"
    if not required:
        raise SystemExit(2)


def run(args: argparse.Namespace) -> None:
    if args.input is None or args.output is None or args.camera_id is None:
        raise ValueError("--input, --output, and --camera-id are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 <= args.confidence_percentile < 100.0:
        raise ValueError("confidence percentile must be in [0,100)")
    if not args.geocalib_weights.is_file():
        raise FileNotFoundError(args.geocalib_weights)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Missing gated commercial checkpoint: {args.checkpoint}. See MODEL_SETUP.md."
        )

    camera_id = safe_camera_id(args.camera_id)
    image_paths = collect_images(args.input)
    sizes = [Image.open(path).size for path in image_paths]
    if len(set(sizes)) != 1:
        raise ValueError(f"All images from one camera must have the same dimensions: {sizes}")

    floor_polygon = load_polygon(args.floor_polygon, camera_id)
    configuration = {
        "camera_id": camera_id,
        "camera_model": args.camera_model,
        "level_method": args.level_method,
        "floor_polygon": floor_polygon,
        "confidence_percentile": args.confidence_percentile,
        "max_points": args.max_points,
        "seed": args.seed,
        "vggt_checkpoint_sha256": EXPECTED_VGGT_SHA256,
    }
    fingerprint = input_fingerprint(image_paths, configuration)
    output = args.output.expanduser().resolve()
    marker = output / ".calibration_complete"
    manifest_path = output / "run_manifest.json"
    if marker.exists() and manifest_path.exists() and not args.force:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old.get("fingerprint") == fingerprint:
            print(json.dumps({"status": "reused", "output": str(output)}, indent=2))
            return
        raise RuntimeError(f"Output contains a different completed run: {output}. Use --force or a new output.")
    output.mkdir(parents=True, exist_ok=True)
    rectified_dir = output / "rectified"
    rectified_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    from geocalib import GeoCalib

    geocalib_started = time.perf_counter()
    geocalib_model = GeoCalib(weights=str(args.geocalib_weights)).cuda().eval()
    raw_images = torch.stack([geocalib_model.load_image(path) for path in image_paths]).cuda()
    with torch.inference_mode():
        calibration = geocalib_model.calibrate(
            raw_images,
            camera_model=args.camera_model,
            shared_intrinsics=len(image_paths) > 1,
        )
    geocalib_seconds = time.perf_counter() - geocalib_started
    geocalib_camera = calibration["camera"]
    gravity_camera = calibration["gravity"].vec3d.detach().float().cpu().numpy()
    rectified_paths = []
    invalid_fractions = []
    for index, image_path in enumerate(image_paths):
        frame_camera = geocalib_camera[index : index + 1]
        frame = raw_images[index : index + 1]
        with torch.inference_mode():
            corrected = frame_camera.undistort_image(frame).clamp(0, 1)
            valid_image = frame_camera.undistort_image(torch.ones_like(frame))
        path = rectified_dir / f"{camera_id}__{index:03d}.png"
        to_pil_image(corrected[0].cpu()).save(path)
        rectified_paths.append(path)
        invalid_fractions.append(
            float((valid_image[0].mean(dim=0) < 0.995).float().mean().cpu())
        )

    distortion = (
        geocalib_camera.dist
        if hasattr(geocalib_camera, "dist")
        else geocalib_camera.new_zeros(geocalib_camera.f.shape)
    )
    geocalib_metadata = {
        "camera_id": camera_id,
        "camera_model": args.camera_model,
        "shared_intrinsics": len(image_paths) > 1,
        "input_size_wh": sizes[0],
        "frame_count": len(image_paths),
        "focal_xy_pixels": as_list(geocalib_camera.f),
        "principal_point_xy_pixels": as_list(geocalib_camera.c),
        "distortion_parameters": as_list(distortion),
        "horizontal_fov_degrees": as_list(torch.rad2deg(geocalib_camera.hfov)),
        "vertical_fov_degrees": as_list(torch.rad2deg(geocalib_camera.vfov)),
        "gravity_camera": gravity_camera.tolist(),
        "focal_uncertainty": as_list(calibration["focal_uncertainty"]),
        "gravity_uncertainty": as_list(calibration["gravity_uncertainty"]),
        "invalid_output_fraction_by_frame": invalid_fractions,
        "seconds": geocalib_seconds,
    }
    (output / "geocalib_calibration.json").write_text(
        json.dumps(geocalib_metadata, indent=2), encoding="utf-8"
    )
    save_before_after(output / "undistortion_review.png", image_paths[0], rectified_paths[0], camera_id)

    del geocalib_model, raw_images, calibration, geocalib_camera
    torch.cuda.empty_cache()

    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    vggt_model = VGGT(enable_point=False, enable_track=False)
    state = load_state_dict(args.checkpoint)
    missing, unexpected = vggt_model.load_state_dict(state, strict=False)
    allowed = [key for key in unexpected if key.startswith(("point_head.", "track_head."))]
    if missing or len(allowed) != len(unexpected):
        raise RuntimeError(
            f"Checkpoint mismatch: missing={missing[:8]}, unexpected={unexpected[:8]}"
        )
    del state
    vggt_model = vggt_model.eval().cuda()
    images = load_and_preprocess_images([str(path) for path in rectified_paths], mode="crop").cuda()

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    vggt_started = time.perf_counter()
    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
        predictions = vggt_model(images)
    torch.cuda.synchronize()
    vggt_seconds = time.perf_counter() - vggt_started
    peak_vram_bytes = int(torch.cuda.max_memory_allocated())

    extrinsics, intrinsics = pose_encoding_to_extri_intri(
        predictions["pose_enc"], images.shape[-2:]
    )
    extrinsics_np = extrinsics.squeeze(0).float().cpu().numpy()
    intrinsics_np = intrinsics.squeeze(0).float().cpu().numpy()
    depth_np = predictions["depth"].squeeze(0).float().cpu().numpy()
    confidence_np = predictions["depth_conf"].squeeze(0).float().cpu().numpy()
    image_np = (
        images.float().cpu().numpy().transpose(0, 2, 3, 1) * 255
    ).clip(0, 255).astype(np.uint8)
    world_points = unproject_depth_map_to_point_map(depth_np, extrinsics_np, intrinsics_np)

    confidence_threshold = float(np.percentile(confidence_np, args.confidence_percentile))
    valid = np.isfinite(world_points).all(axis=-1) & (confidence_np >= confidence_threshold)
    frame_centers, frame_forwards = camera_centers_and_forwards(extrinsics_np)
    camera_center = np.median(frame_centers, axis=0)
    camera_forward = np.mean(frame_forwards, axis=0)
    camera_forward /= max(np.linalg.norm(camera_forward), 1e-12)

    rotations_world_to_camera = extrinsics_np[:, :3, :3]
    gravity_world = np.einsum(
        "nij,nj->ni", rotations_world_to_camera.transpose(0, 2, 1), gravity_camera
    )
    gravity_normal, gravity_deviations = gravity_consensus(gravity_world)

    floor_normal = None
    floor_offset = None
    floor_support = None
    floor_gravity_angle = None
    floor_threshold = None
    floor_sample_count = None
    if floor_polygon is not None:
        masks = np.stack(
            [polygon_mask(floor_polygon, valid.shape[1], valid.shape[2]) & valid[index] for index in range(len(valid))]
        )
        floor_normal, floor_offset, floor_inliers, floor_threshold, floor_sample_count = fit_plane_ransac(
            world_points[masks], rng
        )
        floor_center = floor_inliers.mean(axis=0)
        if (camera_center - floor_center) @ floor_normal < 0:
            floor_normal = -floor_normal
            floor_offset = -floor_offset
        if gravity_normal @ floor_normal < 0:
            gravity_normal = -gravity_normal
        floor_gravity_angle = math.degrees(
            math.acos(float(np.clip(gravity_normal @ floor_normal, -1.0, 1.0)))
        )
        floor_support = len(floor_inliers) / max(floor_sample_count, 1)
        save_floor_mask_review(output / "floor_polygon_review.png", rectified_paths[0], floor_polygon)

    if args.level_method == "floor":
        if floor_normal is None:
            raise ValueError("--level-method floor requires --floor-polygon")
        level_normal = floor_normal
        level_source = "floor-plane RANSAC"
    elif args.level_method == "gravity":
        level_normal = gravity_normal
        level_source = "GeoCalib gravity"
    elif floor_normal is not None and floor_support is not None and floor_support >= 0.10 and floor_gravity_angle is not None and floor_gravity_angle > 35.0:
        level_normal = floor_normal
        level_source = "floor-plane RANSAC (gravity cross-check failed)"
    else:
        level_normal = gravity_normal
        level_source = "GeoCalib gravity"

    selected_points = world_points[valid]
    selected_colors = image_np[valid]
    level_rotation = rotation_to_z(level_normal)
    selected_points = (level_rotation @ (selected_points - camera_center).T).T
    leveled_forward = level_rotation @ camera_forward
    yaw_rotation = rotation_forward_to_up(leveled_forward)
    selected_points = (yaw_rotation @ selected_points.T).T
    leveled_forward = yaw_rotation @ leveled_forward

    radial = np.linalg.norm(selected_points[:, :2], axis=1)
    radial_limit = float(np.percentile(radial, 99.5))
    spatially_valid = radial <= radial_limit
    selected_points = selected_points[spatially_valid]
    selected_colors = selected_colors[spatially_valid]
    if len(selected_points) > args.max_points:
        keep = rng.choice(len(selected_points), args.max_points, replace=False)
        selected_points, selected_colors = selected_points[keep], selected_colors[keep]

    frame_horizontal_fovs = 2.0 * np.arctan(
        images.shape[-1] / (2.0 * intrinsics_np[:, 0, 0])
    )
    horizontal_fov = float(np.median(frame_horizontal_fovs))
    center_spread = float(np.max(np.linalg.norm(frame_centers - camera_center, axis=1)))
    scene_scale = max(float(np.median(radial)), 1e-12)

    write_binary_ply(output / "camera_relative_top_aligned.ply", selected_points, selected_colors)
    save_top_view(output / "top_view.png", selected_points, selected_colors, horizontal_fov, camera_id)
    save_review(
        output / "top_view_review.png",
        image_paths[0],
        rectified_paths[0],
        selected_points,
        selected_colors,
        horizontal_fov,
        camera_id,
        level_source,
        floor_support,
        floor_gravity_angle,
    )
    np.savez_compressed(
        output / "reconstruction_arrays.npz",
        extrinsics=extrinsics_np,
        intrinsics=intrinsics_np,
        frame_centers=frame_centers,
        frame_forwards=frame_forwards,
        gravity_camera=gravity_camera,
        gravity_world=gravity_world,
        level_normal_world=level_normal,
        floor_normal_world=floor_normal if floor_normal is not None else np.empty(0),
        camera_forward_top_aligned=leveled_forward,
    )

    metrics = {
        "status": "camera-relative reconstruction; metric validation pending",
        "camera_id": camera_id,
        "frame_count": len(image_paths),
        "input_files": [str(path) for path in image_paths],
        "input_sha256": {path.name: sha256(path) for path in image_paths},
        "geocalib_seconds": geocalib_seconds,
        "vggt_seconds": vggt_seconds,
        "peak_vram_bytes": peak_vram_bytes,
        "confidence_percentile": args.confidence_percentile,
        "confidence_threshold": confidence_threshold,
        "retained_point_count": int(len(selected_points)),
        "level_method_requested": args.level_method,
        "level_source": level_source,
        "floor_polygon_provided": floor_polygon is not None,
        "floor_support_ratio": floor_support,
        "floor_gravity_angle_degrees": floor_gravity_angle,
        "floor_plane_offset": floor_offset,
        "floor_plane_threshold": floor_threshold,
        "floor_fit_sample_count": floor_sample_count,
        "gravity_deviation_degrees_by_frame": gravity_deviations.tolist(),
        "static_camera_center_spread_scene_ratio": center_spread / scene_scale,
        "horizontal_fov_degrees": math.degrees(horizontal_fov),
        "vggt_checkpoint_expected_sha256": EXPECTED_VGGT_SHA256,
        "coordinate_note": "Camera is at (0,0); predicted forward direction points upward.",
        "scale_note": "Scale is arbitrary until registered to measured control points or a floor plan.",
        "validation_note": "Do not use for operational multi-camera tracking without metric validation.",
    }
    (output / "reconstruction_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    manifest = {
        "fingerprint": fingerprint,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "configuration": configuration,
        "outputs": [
            "geocalib_calibration.json",
            "undistortion_review.png",
            "top_view.png",
            "top_view_review.png",
            "camera_relative_top_aligned.ply",
            "reconstruction_arrays.npz",
            "reconstruction_metrics.json",
        ],
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    marker.write_text("complete\n", encoding="ascii")
    print(json.dumps({"status": "complete", "output": str(output), "metrics": metrics}, indent=2))


def main() -> None:
    args = parse_args()
    os.environ.setdefault("MPLCONFIGDIR", str(PACKAGE_ROOT / "cache/matplotlib"))
    if args.self_test:
        self_test(args)
        return
    run(args)


if __name__ == "__main__":
    main()
