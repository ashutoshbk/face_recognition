"""Isolated GPU job entry point. Mounts only job data and authorized model assets."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib


def main():
    p = argparse.ArgumentParser()
    p.add_argument("kind", choices=["doctor", "calibrate", "infer"])
    p.add_argument("--job", type=Path)
    a = p.parse_args()
    if a.kind == "doctor":
        import torch, cv2, numpy
        from geocalib import GeoCalib
        from vggt.models.vggt import VGGT
        from camex_mct.service import TrackingEngine
        from ultralytics import YOLO

        encoders = subprocess.check_output(
            [os.environ.get("CAMEX_FFMPEG", "ffmpeg"), "-hide_banner", "-encoders"],
            stderr=subprocess.STDOUT,
            text=True,
        )
        if "libx264 " not in encoders:
            raise RuntimeError("The packaged FFmpeg encoder is unavailable")

        print(
            json.dumps(
                {
                    "torch": torch.__version__,
                    "cuda": torch.cuda.is_available(),
                    "opencv": cv2.__version__,
                    "numpy": numpy.__version__,
                    "calibration_imports": bool(GeoCalib and VGGT),
                    "tracking_imports": bool(TrackingEngine and YOLO),
                }
            )
        )
        if not torch.cuda.is_available():
            raise SystemExit("CUDA is required for the full pipeline")
        return
    job = json.loads(a.job.read_text())
    assets = {
        "geocalib": "geocalib_distorted.tar",
        "vggt": "vggt_commercial.safetensors",
        "detector": "yolox_tiny.pth",
        "reid": "osnet_ibn_x1_0_msmt17_combineall.pth",
        "pose": "yolo11n-pose.pt",
    }
    required = (
        ("geocalib", "vggt") if a.kind == "calibrate" else ("detector", "reid", "pose")
    )
    for name in required:
        h = hashlib.sha256()
        with (Path("/models") / assets[name]).open("rb") as f:
            for chunk in iter(lambda: f.read(4 * 1024 * 1024), b""):
                h.update(chunk)
        if h.hexdigest() != job.get("model_hashes", {}).get(name):
            raise RuntimeError(f"Model integrity check failed: {name}")
    import torch

    free, _ = torch.cuda.mem_get_info()
    minimum = 8 * 1024**3 if a.kind == "calibrate" else 1024**3
    if free < minimum:
        raise RuntimeError(
            "Insufficient free GPU memory; retry when competing workloads are idle"
        )
    for folder in ["/job/.cache/ultralytics", "/job/.cache/matplotlib", "/job/tmp"]:
        Path(folder).mkdir(parents=True, exist_ok=True)
    os.environ["TMPDIR"] = "/job/tmp"
    if a.kind == "calibrate":
        command = [
            sys.executable,
            "/opt/camex/engines/calibration/app/calibrate.py",
            "--input",
            "/job/reference.png",
            "--output",
            "/job/calibration",
            "--camera-id",
            job["camera_id"],
            "--checkpoint",
            "/models/vggt_commercial.safetensors",
            "--geocalib-weights",
            "/models/geocalib_distorted.tar",
            "--level-method",
            job.get("level_method", "auto"),
            "--seed",
            "42",
        ]
        if job.get("floor_polygon"):
            command += ["--floor-polygon", "/job/floor_polygon.json"]
        subprocess.run(command, check=True)
    else:
        from inference import run

        run(job)


if __name__ == "__main__":
    main()
