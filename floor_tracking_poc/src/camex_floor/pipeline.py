"""Persistent site projects and isolated calibration/inference jobs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import fcntl
import json
from pathlib import Path
import re
import shutil
import subprocess
import threading
import time
import uuid

import cv2
import numpy as np

from .geometry import (
    calibration_fingerprint,
    calibration_report,
    digest,
    fit_homography,
    fit_layer,
    validate_site,
)
from .server import atomic_json
from .vggt_import import import_vggt


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def token(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,96}", value):
        raise ValueError("Invalid identifier")
    return value


def now():
    return datetime.now(timezone.utc).isoformat()


def read_reference_frame(video, seconds):
    """Seek and verify the decoded time before a frame can become calibration input."""
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError("Cannot open the video for reference extraction")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if seconds > 0 and not cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000):
            raise ValueError(
                "Video decoder could not seek to the requested reference time"
            )
        ok, frame = cap.read()
        if not ok:
            raise ValueError("Cannot extract the selected frame")
        actual = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000
        tolerance = max(1.5 / fps, 0.02) if np.isfinite(fps) and fps > 0 else 0.1
        if not np.isfinite(actual) or actual < 0 or abs(actual - seconds) > tolerance:
            raise ValueError(
                "Video decoder did not confirm the requested frame time; use a seekable video export"
            )
        index = float(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
        return frame, {
            "requested_time_s": seconds,
            "decoded_time_s": actual,
            "frame_index": int(round(index))
            if np.isfinite(index) and index >= 0
            else None,
            "timestamp_error_s": actual - seconds,
            "verification_tolerance_s": tolerance,
            "backend": cap.getBackendName(),
        }
    finally:
        cap.release()


def global_pose(camera, floor):
    ppm = floor.get("pixels_per_metre")
    anchors = [a for a in camera.get("anchors", []) if a.get("role", "fit") == "fit"]
    if not ppm or len(anchors) < 4:
        return {
            "status": "metric pose unavailable; scale and four floor landmarks required"
        }
    d = camera.get("distortion", {})
    f = d.get("focal_xy")
    center = d.get("principal_xy")
    if not f or not center:
        return {"status": "intrinsics unavailable"}
    k = np.array([[f[0], 0, center[0]], [0, f[1], center[1]], [0, 0, 1]], float)
    obj = np.array(
        [[a["map_px"][0] / ppm, -a["map_px"][1] / ppm, 0] for a in anchors], float
    )
    pixels = np.array([a["image_px"] for a in anchors], float)
    try:
        ok, r, t, inliers = cv2.solvePnPRansac(
            obj,
            pixels,
            k,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
            reprojectionError=12,
            iterationsCount=500,
        )
        if not ok:
            return {"status": "pose fit failed"}
        rot = cv2.Rodrigues(r)[0]
        position = (-rot.T @ t).ravel()
        projected = cv2.projectPoints(obj, r, t, k, None)[0].reshape(-1, 2)
        error = np.linalg.norm(projected - pixels, axis=1)
        return {
            "status": "estimated; review reprojection and height",
            "world_axes": "x right, y up from plan top-left, z above floor",
            "world_to_camera_R": rot.tolist(),
            "world_to_camera_t": t.ravel().tolist(),
            "camera_position_m": position.tolist(),
            "median_reprojection_px": float(np.median(error)),
            "max_reprojection_px": float(error.max()),
            "positive_height": bool(position[2] > 0),
        }
    except cv2.error as e:
        return {"status": "pose fit failed", "reason": str(e).splitlines()[0]}


class Projects:
    def __init__(self, root, workspace=None):
        self.root = Path(root).resolve()
        self.workspace = Path(workspace or self.root / "workspaces").resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        (self.workspace / "uploads").mkdir(exist_ok=True)
        self.lock = threading.RLock()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.futures = {}
        for path in self.workspace.glob("*/jobs/*/status.json"):
            data = json.loads(path.read_text())
            if data.get("state") in ("queued", "running"):
                data.update(
                    state="interrupted",
                    message="Application restarted; retry this stage",
                )
                atomic_json(path, data)

    def project_dir(self, pid):
        path = self.workspace / token(pid)
        if not (path / "site.json").is_file():
            raise ValueError("Unknown project")
        return path

    def load(self, pid):
        return json.loads((self.project_dir(pid) / "site.json").read_text())

    def save(self, pid, site):
        site["revision"] = site.get("revision", 0) + 1
        site["updated_at"] = now()
        atomic_json(self.project_dir(pid) / "site.json", site)

    def list(self):
        return [
            {
                "id": p.parent.name,
                "name": d.get("name", d["site_id"]),
                "camera_count": len(d["cameras"]),
                "state": d.get("state", "draft"),
            }
            for p in sorted(self.workspace.glob("*/site.json"))
            for d in [json.loads(p.read_text())]
        ]

    def info(self, pid):
        d = self.load(pid)
        d["verification"] = validate_site(d, False)
        d["jobs"] = [
            self.job_info(pid, p.parent.name)
            for p in sorted(
                self.project_dir(pid).glob("jobs/*/status.json"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        ][:12]
        d["engine"] = self.engine_info()
        return d

    def engine_info(self):
        p = self.root / "runtime.json"
        if not p.exists():
            return {"ready": False, "message": "Run install.sh with the model bundle"}
        d = json.loads(p.read_text())
        return {
            "ready": d.get("mode") == "full",
            "image": d.get("gpu_image"),
            "message": "Verified GPU engine"
            if d.get("mode") == "full"
            else "UI-only development mode",
        }

    def register_upload(self, path, name):
        if Path(path).parent != self.workspace / "uploads":
            raise ValueError("Invalid upload location")
        info = {
            "filename": Path(name).name[:160],
            "sha256": file_hash(path),
            "bytes": Path(path).stat().st_size,
            "path": Path(path).name,
        }
        atomic_json(Path(path).with_suffix(".json"), info)
        return {
            "upload_id": Path(path).stem,
            **{k: v for k, v in info.items() if k != "path"},
        }

    def upload(self, uid):
        meta = self.workspace / "uploads" / (token(uid) + ".json")
        data = json.loads(meta.read_text())
        return meta.parent / data["path"], data

    def verify_models(self, runtime):
        manifest = json.loads((self.root / "models_manifest.json").read_text())[
            "models"
        ]
        cache_path = self.root / ".cache/model_verification.json"
        cache_path.parent.mkdir(exist_ok=True)
        cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
        for name, spec in manifest.items():
            path = Path(runtime["models_dir"]) / spec["filename"]
            if not path.is_file():
                raise ValueError(f"Model asset missing: {name}")
            st = path.stat()
            stamp = [st.st_size, st.st_mtime_ns, st.st_ctime_ns]
            expected = runtime.get("model_hashes", {}).get(name)
            cached = cache.get(str(path), {})
            if cached.get("stamp") != stamp or cached.get("sha256") != expected:
                actual = file_hash(path)
                if actual != expected:
                    raise ValueError(
                        f"Model checksum changed: {name}; restore the bundle and rerun installation"
                    )
                cache[str(path)] = {"stamp": stamp, "sha256": actual}
        atomic_json(cache_path, cache)

    def create(self, name, upload_id, page=0):
        source, meta = self.upload(upload_id)
        pid = uuid.uuid4().hex[:16]
        folder = self.workspace / pid
        (folder / "media").mkdir(parents=True)
        if meta["filename"].lower().endswith(".pdf"):
            import fitz

            with fitz.open(source) as pdf:
                if not 0 <= int(page) < len(pdf):
                    raise ValueError("Invalid drawing page")
                pix = pdf[int(page)].get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                pix.save(str(folder / "media/floor.png"))
        else:
            im = cv2.imread(str(source))
            if im is None:
                raise ValueError("Upload a floor-plan image or PDF")
            cv2.imwrite(str(folder / "media/floor.png"), im)
        im = cv2.imread(str(folder / "media/floor.png"))
        h, w = im.shape[:2]
        site = {
            "schema_version": "1.0",
            "project_id": pid,
            "site_id": pid,
            "name": name or "New site",
            "revision": 0,
            "state": "draft",
            "created_at": now(),
            "floor": {
                "asset": "media/floor.png",
                "asset_sha256": file_hash(folder / "media/floor.png"),
                "size_wh": [w, h],
                "pixels_per_metre": None,
            },
            "cameras": {},
            "run_settings": {
                "start_s": 0,
                "duration_s": 120,
                "fps": 10,
                "inference_width": 1280,
            },
        }
        atomic_json(folder / "site.json", site)
        return self.info(pid)

    def add_camera(self, pid, cid, label, uid):
        cid = token(cid)
        with self.lock:
            site = self.load(pid)
            if cid in site["cameras"]:
                raise ValueError("Camera ID already registered")
            if len(site["cameras"]) >= 20:
                raise ValueError("This release supports up to 20 cameras per project")
            source, meta = self.upload(uid)
            cap = cv2.VideoCapture(str(source))
            ok, frame = cap.read()
            fps = cap.get(cv2.CAP_PROP_FPS)
            count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            cap.release()
            if not ok or fps <= 0 or count <= 0:
                raise ValueError("Unable to decode the uploaded video")
            folder = self.project_dir(pid)
            video = f"media/{uuid.uuid4().hex}.mp4"
            shutil.copy2(source, folder / video)
            raw = f"media/{uuid.uuid4().hex}.png"
            cv2.imwrite(str(folder / raw), frame)
            site["cameras"][cid] = {
                "label": label or cid,
                "video": video,
                "video_sha256": meta["sha256"],
                "video_name": meta["filename"],
                "video_duration_s": count / fps,
                "video_fps": fps,
                "video_start_seconds": 0,
                "reference_time_s": 0,
                "raw_reference": raw,
                "reference": raw,
                "reference_sha256": file_hash(folder / raw),
                "image_size_wh": [frame.shape[1], frame.shape[0]],
                "distortion": {"model": "none"},
                "anchors": [],
                "image_to_floor_h": None,
                "calibration_state": "not_calibrated",
                "enabled": True,
                "locked": False,
                "visible": True,
            }
            site["state"] = "draft"
            self.save(pid, site)
            return self.info(pid)

    def snapshot(self, pid, cid, seconds):
        with self.lock:
            site = self.load(pid)
            c = site["cameras"][cid]
            if c.get("locked"):
                raise ValueError("Unlock this camera before replacing its reference")
            seconds = float(seconds)
            if not np.isfinite(seconds) or not 0 <= seconds < c["video_duration_s"]:
                raise ValueError("Reference time is outside the video")
            if any(j["state"] in ("queued", "running") for j in self.info(pid)["jobs"]):
                raise ValueError(
                    "Wait for the current project job before changing the reference"
                )
            folder = self.project_dir(pid)
            frame, capture = read_reference_frame(folder / c["video"], seconds)
            raw = f"media/{uuid.uuid4().hex}.png"
            if not cv2.imwrite(str(folder / raw), frame):
                raise ValueError("Unable to save the extracted reference image")
            for k in [
                "point_layer",
                "all_point_layer",
                "layer_to_image_h",
                "calibration_job",
                "calibration_fingerprint",
                "approved_fingerprint",
                "floor_polygon",
                "global_pose",
                "fit_audit",
                "local_geometry_status",
                "camera_to_local_rotation",
                "gravity_camera",
                "camera_to_floor_R",
                "camera_to_floor_t",
                "local_floor_normal",
                "local_floor_offset",
                "local_pose_units",
                "floor_gravity_disagreement",
                "local_plane_source",
                "calibration_source",
            ]:
                c.pop(k, None)
            c.update(
                raw_reference=raw,
                reference=raw,
                reference_time_s=seconds,
                reference_actual_time_s=capture["decoded_time_s"],
                reference_capture=capture,
                reference_sha256=file_hash(folder / raw),
                image_size_wh=[frame.shape[1], frame.shape[0]],
                distortion={"model": "none"},
                anchors=[],
                image_to_floor_h=None,
                calibration_state="not_calibrated",
            )
            site["state"] = "draft"
            self.save(pid, site)
            return self.info(pid)

    def remove_camera(self, pid, cid, revision):
        """Detach a camera, retaining its media, prior exports and recovery snapshot."""
        cid = token(cid)
        with self.lock:
            site = self.load(pid)
            if revision != site["revision"]:
                raise ValueError("Project changed; reload before removing a camera")
            if cid not in site["cameras"]:
                raise ValueError("Camera is not registered in this project")
            if site["cameras"][cid].get("locked"):
                raise ValueError("Unlock this camera before removing it")
            folder = self.project_dir(pid)
            if any(
                json.loads(path.read_text()).get("state") in ("queued", "running")
                for path in folder.glob("jobs/*/status.json")
            ):
                raise ValueError(
                    "Wait for the current project job before removing a camera"
                )
            finished = folder / "finished_site.json"
            archive = folder / "camera_archive" / (uuid.uuid4().hex + ".json")
            archive.parent.mkdir(exist_ok=True)
            atomic_json(
                archive,
                {
                    "camera_id": cid,
                    "removed_at": now(),
                    "site_before_removal": site,
                    "finished_site_before_removal": json.loads(finished.read_text())
                    if finished.exists()
                    else None,
                },
            )
            del site["cameras"][cid]
            site["state"] = "draft"
            # Invalidate the exported snapshot, not the other cameras' saved approvals.
            if finished.exists():
                atomic_json(
                    finished,
                    {
                        **site,
                        "requires_finish": True,
                        "cameras": {
                            key: {
                                k: v
                                for k, v in c.items()
                                if k != "approved_fingerprint"
                            }
                            for key, c in site["cameras"].items()
                        },
                    },
                )
            self.save(pid, site)
            return self.info(pid)

    def edit(self, pid, cid, body):
        with self.lock:
            site = self.load(pid)
            if body.get("revision") != site["revision"]:
                raise ValueError("Project changed; reload before saving")
            c = site["cameras"][cid]
            action = body.get("action", "save")
            if c.get("locked") and action not in ("unlock", "visibility"):
                raise ValueError("Unlock this camera before editing")
            if action in ("lock", "unlock"):
                c["locked"] = action == "lock"
                self.save(pid, site)
                return self.info(pid)
            if action == "visibility":
                c["visible"] = not c.get("visible", True)
                self.save(pid, site)
                return self.info(pid)
            if "anchors" in body:
                c["anchors"] = body["anchors"]
            if "pixels_per_metre" in body:
                ppm = body["pixels_per_metre"]
                if ppm is not None and (not np.isfinite(ppm) or ppm <= 0):
                    raise ValueError("Scale must be positive")
                site["floor"]["pixels_per_metre"] = ppm
                for other in site["cameras"].values():
                    other.pop("approved_fingerprint", None)
            if action in ("fit", "fit_layer"):
                tolerance = (
                    site["floor"].get("pixels_per_metre")
                    or np.linalg.norm(site["floor"]["size_wh"]) / 20
                ) * 0.6
                h, audit = (
                    fit_homography(c["anchors"], tolerance)
                    if action == "fit"
                    else fit_layer(c, tolerance)
                )
                c["image_to_floor_h"] = h
                c["fit_audit"] = audit
            elif action == "transform":
                h = np.asarray(c["image_to_floor_h"], float)
                a = np.radians(float(body.get("rotation", 0)))
                s = float(body.get("scale", 1))
                dx = float(body.get("dx", 0))
                dy = float(body.get("dy", 0))
                if (
                    h.shape != (3, 3)
                    or not 0.1 <= s <= 10
                    or not np.isfinite([a, dx, dy]).all()
                ):
                    raise ValueError("Invalid transform")
                center = np.asarray(site["floor"]["size_wh"]) / 2
                linear = s * np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
                t = np.eye(3)
                t[:2, :2] = linear
                t[:2, 2] = center - linear @ center + [dx, dy]
                c["image_to_floor_h"] = (t @ h).tolist()
            elif action == "floor_polygon":
                polygon = np.asarray(body.get("polygon"), float)
                if (
                    polygon.ndim != 2
                    or polygon.shape[1] != 2
                    or len(polygon) < 3
                    or not np.isfinite(polygon).all()
                    or (polygon < 0).any()
                    or (polygon > 1).any()
                ):
                    raise ValueError("Select at least three valid floor polygon points")
                c["floor_polygon"] = polygon.tolist()
            c.pop("approved_fingerprint", None)
            site["state"] = "draft"
            c["global_pose"] = global_pose(c, site["floor"])
            if action == "approve":
                report = calibration_report(
                    c, site["floor"].get("pixels_per_metre"), site.get("acceptance")
                )
                if not report["ready"]:
                    raise ValueError("; ".join(report["reasons"]))
                c["approved_fingerprint"] = calibration_fingerprint(c, site["floor"])
            validate_site(site, False)
            self.save(pid, site)
            return self.info(pid)

    def finish(self, pid, preview=False):
        with self.lock:
            site = self.load(pid)
            if not site["cameras"] or any(
                c.get("image_to_floor_h") is None for c in site["cameras"].values()
            ):
                raise ValueError("Calibrate and align every registered camera")
            if not preview:
                validate_site(site, True)
            site["state"] = "preview" if preview else "approved"
            self.save(pid, site)
            atomic_json(self.project_dir(pid) / "finished_site.json", site)
            return self.info(pid)

    def job_info(self, pid, jid):
        path = self.project_dir(pid) / "jobs" / token(jid)
        d = json.loads((path / "status.json").read_text())
        if (path / "progress.json").exists():
            try:
                d["progress"] = json.loads((path / "progress.json").read_text())
            except json.JSONDecodeError:
                pass
        d["log_tail"] = ""
        for name in ("worker.log", "render.log"):
            if (path / name).exists():
                with (path / name).open("rb") as f:
                    f.seek(max(0, (path / name).stat().st_size - 4000))
                    d["log_tail"] += f"\n{name}:\n" + f.read().decode(errors="replace")
        return d

    def enqueue(self, pid, kind, cid=None, settings=None):
        with self.lock:
            site = self.load(pid)
            settings = settings or {}
            runtime = json.loads((self.root / "runtime.json").read_text())
            if runtime.get("mode") != "full":
                raise ValueError("Full GPU engine is not installed")
            self.verify_models(runtime)
            check = subprocess.run(
                [
                    "docker",
                    "image",
                    "inspect",
                    runtime.get("gpu_image_id") or runtime["gpu_image"],
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            if check.returncode:
                raise ValueError("The installed GPU image is missing; rerun install.sh")
            if any(j["state"] in ("queued", "running") for j in self.info(pid)["jobs"]):
                raise ValueError("A project job is already active")
            folder = self.project_dir(pid)
            if kind == "calibrate":
                c = site["cameras"][cid]
                source = {
                    "camera_id": cid,
                    "reference_sha256": file_hash(folder / c["raw_reference"]),
                    "floor_polygon": c.get("floor_polygon"),
                    "level_method": settings.get("level_method", "auto"),
                }
                if source["level_method"] not in ("auto", "floor", "gravity"):
                    raise ValueError("Unknown leveling method")
            else:
                if site.get("state") not in ("approved", "preview"):
                    raise ValueError("Finish calibration before running MCT")
                fps = int(settings.get("fps", 10))
                duration = float(settings.get("duration_s", 120))
                start = float(settings.get("start_s", 0))
                width = int(settings.get("inference_width", 1280))
                detector_size = int(settings.get("detector_size", 416))
                if detector_size not in (416, 640):
                    raise ValueError("Choose balanced or detailed detection")
                if (
                    not 1 <= fps <= 30
                    or not 0 < duration <= 86400
                    or start < 0
                    or not 320 <= width <= 4096
                ):
                    raise ValueError("Invalid run settings")
                offsets = settings.get("offsets", {})
                for camera_id, c in site["cameras"].items():
                    offset = float(
                        offsets.get(camera_id, c.get("video_start_seconds", 0))
                    )
                    if (
                        offset < 0
                        or start + offset + duration > c["video_duration_s"] + 0.11
                    ):
                        raise ValueError(
                            f"Video {camera_id} does not cover the requested interval"
                        )
                    c["video_start_seconds"] = offset + start
                source = {
                    "site": site,
                    "settings": {
                        "fps": fps,
                        "duration_s": duration,
                        "start_s": start,
                        "inference_width": width,
                        "detector_size": detector_size,
                    },
                }
                site.setdefault("runtime", {})["reject_trajectory_jumps"] = 1
            fingerprint = digest(
                {
                    "source": source,
                    "models": runtime.get("model_hashes"),
                    "image": runtime.get("gpu_image_id"),
                    "worker_code": file_hash(self.root / "gpu/worker.py"),
                    "importer_code": file_hash(
                        Path(__file__).parent / "vggt_import.py"
                    ),
                }
            )
            for p in folder.glob("jobs/*/status.json"):
                old = json.loads(p.read_text())
                if (
                    old.get("state") == "complete"
                    and old.get("fingerprint") == fingerprint
                ):
                    if (
                        kind == "calibrate"
                        and site["cameras"][cid].get("calibration_fingerprint")
                        != fingerprint
                    ):
                        self.accept_calibration(pid, old["id"], old)
                    return {**old, "reused": True}
            jid = uuid.uuid4().hex[:16]
            job_dir = folder / "jobs" / jid
            job_dir.mkdir(parents=True)
            status = {
                "id": jid,
                "kind": kind,
                "camera_id": cid,
                "state": "queued",
                "created_at": now(),
                "fingerprint": fingerprint,
                "source": source,
                "message": "Waiting for GPU worker",
            }
            atomic_json(job_dir / "status.json", status)
            if kind == "calibrate":
                shutil.copy2(folder / c["raw_reference"], job_dir / "reference.png")
                payload = {"camera_id": cid, **source}
                if source.get("floor_polygon"):
                    atomic_json(
                        job_dir / "floor_polygon.json",
                        {"polygon": source["floor_polygon"]},
                    )
            else:
                atomic_json(job_dir / "site.json", site)
                payload = {
                    "site_id": site["site_id"],
                    "preview": site["state"] == "preview",
                    **source["settings"],
                }
                inference_fingerprint = digest(
                    {
                        "inputs": {
                            k: {
                                "sha256": c["video_sha256"],
                                "offset": c["video_start_seconds"],
                            }
                            for k, c in site["cameras"].items()
                        },
                        "settings": source["settings"],
                        "models": runtime.get("model_hashes"),
                        "image": runtime.get("gpu_image_id"),
                    }
                )
                status["inference_fingerprint"] = inference_fingerprint
                for previous in folder.glob("jobs/*/status.json"):
                    old = json.loads(previous.read_text())
                    old_dir = previous.parent
                    if (
                        old_dir != job_dir
                        and old.get("inference_fingerprint") == inference_fingerprint
                        and (old_dir / "observations.jsonl").is_file()
                        and (old_dir / "inference_metrics.json").is_file()
                        and old.get("observations_sha256")
                        == file_hash(old_dir / "observations.jsonl")
                    ):
                        for name in (
                            "observations.jsonl",
                            "inference_metrics.json",
                            "inference_configuration.json",
                        ):
                            if (old_dir / name).is_file():
                                shutil.copy2(old_dir / name, job_dir / name)
                        status["inference_reused_from"] = old["id"]
                        break
                atomic_json(job_dir / "status.json", status)
            atomic_json(job_dir / "job.json", payload)
            payload["model_hashes"] = runtime.get("model_hashes", {})
            atomic_json(job_dir / "job.json", payload)
            self.futures[jid] = self.executor.submit(self.execute, pid, jid, runtime)
            return status

    def execute(self, pid, jid, runtime):
        folder = self.project_dir(pid)
        job_dir = folder / "jobs" / jid
        status = json.loads((job_dir / "status.json").read_text())
        kind = status["kind"]
        status.update(
            state="running", started_at=now(), message="Loading isolated GPU engine"
        )
        atomic_json(job_dir / "status.json", status)
        command = [
            "docker",
            "run",
            "--rm",
            "--gpus",
            "all",
            "--network",
            "none",
            "--shm-size",
            "2g",
            "-v",
            f"{job_dir}:/job",
            "-v",
            f"{Path(runtime['models_dir']).resolve()}:/models:ro",
        ]
        if kind == "infer":
            command += ["-v", f"{folder}:/site:ro"]
        command += [
            runtime.get("gpu_image_id") or runtime["gpu_image"],
            "python",
            "/opt/camex/gpu/worker.py",
            kind,
            "--job",
            "/job/job.json",
        ]
        started = time.perf_counter()
        try:
            if not status.get("inference_reused_from"):
                with (self.workspace / ".gpu.lock").open("a") as gpu_lock:
                    fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                    with (job_dir / "worker.log").open("w") as log:
                        result = subprocess.run(
                            command, stdout=log, stderr=subprocess.STDOUT
                        )
                if result.returncode:
                    raise RuntimeError(
                        "GPU stage failed; inspect the job log and retry"
                    )
            if kind == "calibrate":
                self.accept_calibration(pid, jid, status)
            else:
                status["observations_sha256"] = file_hash(
                    job_dir / "observations.jsonl"
                )
                status["message"] = "Rendering floor tracking video"
                atomic_json(job_dir / "status.json", status)
                atomic_json(
                    job_dir / "progress.json",
                    {
                        "stage": "Rendering floor map",
                        "completed_s": 0,
                        "total_s": status["source"]["settings"]["duration_s"],
                        "fraction": 0,
                    },
                )
                site = json.loads((job_dir / "site.json").read_text())
                for c in site["cameras"].values():
                    c["video"] = str(Path("/site") / c["video"])
                site["floor"]["asset"] = str(Path("/site") / site["floor"]["asset"])
                atomic_json(job_dir / "render_site.json", site)
                render = [
                    "docker",
                    "run",
                    "--rm",
                    "--network",
                    "none",
                    "-v",
                    f"{job_dir}:/job",
                    "-v",
                    f"{folder}:/site:ro",
                    runtime.get("gpu_image_id") or runtime["gpu_image"],
                    "python",
                    "-m",
                    "camex_floor.cli",
                    "replay",
                    "--site",
                    "/job/render_site.json",
                    "--events",
                    "/job/observations.jsonl",
                    "--output",
                    "/job/output",
                    "--fps",
                    str(status["source"]["settings"]["fps"]),
                ]
                render += [
                    "--progress-file",
                    "/job/progress.json",
                    "--expected-duration",
                    str(status["source"]["settings"]["duration_s"]),
                ]
                if status["source"]["site"]["state"] == "preview":
                    render.append("--allow-draft")
                with (job_dir / "render.log").open("w") as log:
                    result = subprocess.run(
                        render, stdout=log, stderr=subprocess.STDOUT
                    )
                if result.returncode:
                    raise RuntimeError("Rendering failed; inspect render.log")
            status.update(
                state="complete",
                completed_at=now(),
                elapsed_s=time.perf_counter() - started,
                message="Calibration ready for alignment"
                if kind == "calibrate"
                else "Video and tracking results ready",
            )
        except Exception as error:
            status.update(
                state="failed",
                message=str(error),
                elapsed_s=time.perf_counter() - started,
            )
        atomic_json(job_dir / "status.json", status)

    def accept_calibration(self, pid, jid, status):
        with self.lock:
            folder = self.project_dir(pid)
            job_dir = folder / "jobs" / jid
            site = self.load(pid)
            cid = status["camera_id"]
            old = site["cameras"][cid]
            if (
                file_hash(folder / old["raw_reference"])
                != status["source"]["reference_sha256"]
            ):
                raise ValueError(
                    "Reference changed while calibration ran; output was not applied"
                )
            import_path = job_dir / "import_site.json"
            atomic_json(import_path, {**site, "cameras": {}})
            import_vggt(
                import_path, cid, job_dir / "calibration", allow_provisional=True
            )
            new = json.loads(import_path.read_text())["cameras"][cid]
            for key in ("reference", "point_layer", "all_point_layer"):
                new[key] = str((job_dir / new[key]).relative_to(folder))
            for key in (
                "label",
                "video",
                "video_sha256",
                "video_name",
                "video_duration_s",
                "video_fps",
                "video_start_seconds",
                "reference_time_s",
                "raw_reference",
                "enabled",
                "visible",
                "locked",
            ):
                new[key] = old[key]
            for key in ("reference_actual_time_s", "reference_capture"):
                if key in old:
                    new[key] = old[key]
            if old.get("floor_polygon"):
                new["floor_polygon"] = old["floor_polygon"]
            new["calibration_job"] = jid
            new["calibration_state"] = "calibrated"
            new["calibration_fingerprint"] = status["fingerprint"]
            site["cameras"][cid] = new
            site["state"] = "draft"
            self.save(pid, site)

    def asset(self, pid, kind, cid=None, jid=None):
        folder = self.project_dir(pid)
        site = self.load(pid)
        if kind == "floor":
            return folder / site["floor"]["asset"]
        if jid:
            job = folder / "jobs" / token(jid)
            allowed = {
                "demo": "output/demo.mp4",
                "events": "output/results.jsonl",
                "manifest": "output/manifest.json",
                "telemetry": "inference_metrics.json",
            }
            if kind in allowed:
                return job / allowed[kind]
            raise ValueError("Unknown result asset")
        c = site["cameras"][cid]
        if kind in (
            "video",
            "raw_reference",
            "reference",
            "point_layer",
            "all_point_layer",
        ):
            return folder / c[kind]
        allowed = {
            "calibration_review": "top_view_review.png",
            "undistortion_review": "undistortion_review.png",
            "point_cloud": "camera_relative_top_aligned.ply",
        }
        if kind in allowed:
            return (
                folder / "jobs" / c["calibration_job"] / "calibration" / allowed[kind]
            )
        raise ValueError("Unknown asset")
