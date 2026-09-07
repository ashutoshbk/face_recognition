"""Shared projection and robust calibration; coordinates are explicit."""

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def transform(matrix, points):
    p = np.asarray(points, dtype=float)
    one = p.ndim == 1
    p = p.reshape(-1, 2)
    q = np.column_stack((p, np.ones(len(p)))) @ np.asarray(matrix, dtype=float).T
    valid = np.isfinite(q).all(axis=1) & (np.abs(q[:, 2]) > 1e-9)
    if not valid.all():
        raise ValueError("Point outside finite projection domain")
    q = q[:, :2] / q[:, 2, None]
    return q[0] if one else q


def fit_homography(anchors, threshold_px=15.0):
    fit = [a for a in anchors if a.get("role", "fit") == "fit"]
    if len(fit) < 4:
        raise ValueError("At least four fit correspondences are required")
    source = np.asarray([a["image_px"] for a in fit], dtype=float)
    target = np.asarray([a["map_px"] for a in fit], dtype=float)
    for points in (source, target):
        if (
            not np.isfinite(points).all()
            or np.linalg.matrix_rank(points - points.mean(0)) < 2
        ):
            raise ValueError(
                "Landmarks must be finite and spread across two dimensions"
            )
    cv2.setRNGSeed(42)
    matrix, mask = cv2.findHomography(
        source, target, cv2.RANSAC, threshold_px, maxIters=5000, confidence=0.999
    )
    if matrix is None or mask is None or int(mask.sum()) < 4:
        raise ValueError("A supported floor homography could not be fitted")
    inliers = mask.ravel().astype(bool)
    matrix, _ = cv2.findHomography(source[inliers], target[inliers], 0)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-12:
        raise ValueError("Degenerate floor homography")
    errors = np.linalg.norm(transform(matrix, source) - target, axis=1)
    return matrix.tolist(), {
        "inlier_count": int(inliers.sum()),
        "fit_count": len(fit),
        "outlier_indices": np.flatnonzero(~inliers).tolist(),
        "errors_px": errors.tolist(),
    }


def calibration_report(camera, ppm, gate=None):
    gate = {
        "minimum_holdouts": 2,
        "median_error_m": 0.25,
        "maximum_error_m": 0.50,
        **(gate or {}),
    }
    anchors = camera.get("anchors", [])
    checks = [a for a in anchors if a.get("role") == "check"]
    fit = [a for a in anchors if a.get("role", "fit") == "fit"]
    h = camera.get("image_to_floor_h")
    errors, reasons = [], []
    if not ppm or ppm <= 0:
        reasons.append("Metric floor scale is missing")
    if h is None:
        reasons.append("Projection matrix is missing")
    if len(checks) < gate["minimum_holdouts"]:
        reasons.append("Add at least two independent holdout correspondences")
    if h is not None:
        try:
            matrix = np.asarray(h, dtype=float)
            if (
                matrix.shape != (3, 3)
                or not np.isfinite(matrix).all()
                or abs(np.linalg.det(matrix)) < 1e-12
            ):
                raise ValueError("Invalid projection matrix")
            for a in checks:
                errors.append(
                    float(np.linalg.norm(transform(h, a["image_px"]) - a["map_px"]))
                )
                if any(
                    np.linalg.norm(np.array(a["image_px"]) - b["image_px"]) < 3
                    for b in fit
                ):
                    reasons.append("Holdout duplicates a fit point")
            if (
                len(checks) >= 2
                and np.linalg.norm(
                    np.ptp(np.array([a["image_px"] for a in checks]), axis=0)
                )
                < min(camera["image_size_wh"]) * 0.1
            ):
                reasons.append("Holdout points are too close together")
        except (ValueError, TypeError, IndexError):
            reasons.append("Invalid projection or anchor")
    median = float(np.median(errors)) if errors else None
    maximum = max(errors) if errors else None
    if errors and ppm:
        if (
            median / ppm > gate["median_error_m"]
            or maximum / ppm > gate["maximum_error_m"]
        ):
            reasons.append("Holdout error exceeds acceptance limits")
    return {
        "ready": not reasons,
        "reasons": sorted(set(reasons)),
        "fit_count": len(fit),
        "holdout_count": len(checks),
        "median_px": median,
        "maximum_px": maximum,
        "median_m": median / ppm if median is not None and ppm else None,
        "maximum_m": maximum / ppm if maximum is not None and ppm else None,
    }


def fit_layer(camera, threshold_px=15.0):
    anchors = [a for a in camera.get("anchors", []) if a.get("role", "fit") == "fit"]
    if len(anchors) < 2 or not camera.get("layer_to_image_h"):
        raise ValueError("Two fit points and a local camera layer are required")
    source = transform(
        np.linalg.inv(np.asarray(camera["layer_to_image_h"])),
        [a["image_px"] for a in anchors],
    )
    target = np.asarray([a["map_px"] for a in anchors], float)
    cv2.setRNGSeed(42)
    a, mask = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold_px,
        maxIters=5000,
        confidence=0.999,
    )
    if a is None or mask is None or mask.sum() < 2:
        raise ValueError("Local layer fit failed")
    h = np.vstack((a, [0, 0, 1])) @ np.linalg.inv(
        np.asarray(camera["layer_to_image_h"])
    )
    errors = np.linalg.norm(
        transform(h, [a["image_px"] for a in anchors]) - target, axis=1
    )
    return h.tolist(), {
        "method": "local_layer_similarity",
        "inlier_count": int(mask.sum()),
        "fit_count": len(anchors),
        "outlier_indices": np.flatnonzero(~mask.ravel().astype(bool)).tolist(),
        "errors_px": errors.tolist(),
    }


def calibration_fingerprint(camera, floor):
    return digest(
        {
            "camera": {
                k: camera.get(k)
                for k in (
                    "image_size_wh",
                    "image_to_floor_h",
                    "distortion",
                    "reference_sha256",
                    "anchors",
                    "valid_floor_polygon",
                )
            },
            "floor": {
                k: floor.get(k) for k in ("size_wh", "pixels_per_metre", "asset_sha256")
            },
        }
    )


def validate_site(site, require_approved=True):
    if site.get("schema_version") != "1.0" or not site.get("site_id"):
        raise ValueError("Site schema_version 1.0 and nonempty site_id are required")
    floor = site["floor"]
    size = np.asarray(floor["size_wh"], dtype=float)
    if size.shape != (2,) or not np.isfinite(size).all() or (size <= 0).any():
        raise ValueError("Invalid floor dimensions")
    ppm = floor.get("pixels_per_metre")
    if ppm is not None and (not np.isfinite(ppm) or ppm <= 0):
        raise ValueError("Invalid metric scale")
    cameras = site.get("cameras", {})
    if require_approved and not cameras:
        raise ValueError("Register at least one camera")
    reports = {}
    for cid, c in cameras.items():
        if not cid or len(c["image_size_wh"]) != 2 or min(c["image_size_wh"]) <= 0:
            raise ValueError("Invalid camera ID or image dimensions")
        distortion = c.get("distortion", {"model": "none"})
        if distortion.get("model") not in ("none", "simple_divisional", "opencv"):
            raise ValueError(f"Unsupported distortion model for {cid}")
        reports[cid] = calibration_report(c, ppm, site.get("acceptance"))
        if require_approved and (
            not reports[cid]["ready"]
            or c.get("approved_fingerprint") != calibration_fingerprint(c, floor)
        ):
            raise ValueError(
                f"Camera {cid} calibration is unapproved or changed: {reports[cid]['reasons']}"
            )
    return reports


def load_site(path):
    site = json.loads(Path(path).read_text())
    validate_site(site, False)
    return site


class Projector:
    def __init__(self, camera, floor):
        self.camera, self.floor = camera, floor

    def undistort(self, point, size_wh):
        source = np.asarray(size_wh, dtype=float)
        expected = np.asarray(self.camera["image_size_wh"], dtype=float)
        if (
            source.shape != (2,)
            or (source <= 0).any()
            or abs((source[0] / source[1]) / (expected[0] / expected[1]) - 1) > 0.005
        ):
            raise ValueError(
                "Frame aspect ratio differs from calibration; recalibrate changed crop or lens"
            )
        p = np.asarray(point, dtype=float) * expected / source
        d = self.camera.get("distortion", {"model": "none"})
        if d["model"] == "simple_divisional":
            f, center = np.asarray(d["focal_xy"]), np.asarray(d["principal_xy"])
            normalized = (p - center) / f
            denominator = 1 + d["k1"] * float(normalized @ normalized)
            if denominator <= 1e-6:
                raise ValueError("Pixel outside distortion domain")
            p = normalized / denominator * f + center
        elif d["model"] == "opencv":
            k = np.asarray(d["K"], dtype=float)
            p = cv2.undistortPoints(
                p.reshape(1, 1, 2),
                k,
                np.asarray(d["coefficients"], dtype=float),
                P=np.asarray(d.get("rectified_K", k)),
            ).reshape(2)
        return p

    def project(self, point, size_wh):
        rectified = self.undistort(point, size_wh)
        p = transform(self.camera["image_to_floor_h"], rectified)
        if not (
            0 <= p[0] < self.floor["size_wh"][0]
            and 0 <= p[1] < self.floor["size_wh"][1]
        ):
            raise ValueError("Projection outside floor plan")
        polygon = self.camera.get("valid_floor_polygon") or self.floor.get(
            "walkable_polygon"
        )
        if (
            polygon
            and cv2.pointPolygonTest(np.asarray(polygon, np.float32), tuple(p), False)
            < 0
        ):
            raise ValueError("Projection outside calibrated floor region")
        return p

    def gravity_foot(self, shoulder, hip, ratio, size_wh):
        gravity = self.camera.get("gravity_camera")
        d = self.camera.get("distortion", {})
        if gravity is None or d.get("model") != "simple_divisional":
            return None
        a = self.undistort(shoulder, size_wh)
        b = self.undistort(hip, size_wh)
        g = np.asarray(gravity, float)
        f = np.asarray(d["focal_xy"], float)
        center = np.asarray(d["principal_xy"], float)
        k = np.array([[f[0], 0, center[0]], [0, f[1], center[1]], [0, 0, 1.0]])
        v = k @ g
        direction = b * g[2] - v[:2]
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm < 1e-8:
            return None
        rect = b + (ratio - 1) * np.linalg.norm(b - a) * direction / norm
        normalized = (rect - center) / f
        discriminant = 1 - 4 * d["k1"] * float(normalized @ normalized)
        if discriminant <= 0:
            return None
        raw = (2 / (1 + np.sqrt(discriminant))) * normalized * f + center
        return raw * np.asarray(size_wh) / np.asarray(self.camera["image_size_wh"])
