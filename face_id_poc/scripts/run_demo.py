"""Location-independent enrollment + demo launcher. No site data in the code."""

import argparse
import hashlib
import json
import os
import math
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "camex-faceid:0.1.1"


def kit_path(root, value):
    value = Path(value)
    if value.is_absolute():
        raise ValueError("Kit paths must be relative")
    resolved = (root / value).resolve()
    if root not in resolved.parents:
        raise ValueError("Kit path escapes its directory")
    return resolved


def verify_kit(root):
    manifest = root / "FILE_MANIFEST.json"
    if not manifest.is_file():
        raise ValueError("Test kit FILE_MANIFEST.json is missing")
    for row in json.loads(manifest.read_text())["files"]:
        path = kit_path(root, row["path"])
        h = hashlib.sha256()
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1048576), b""):
                h.update(block)
        if path.stat().st_size != row["size"] or h.hexdigest() != row["sha256"]:
            raise ValueError(f"Kit checksum mismatch: {row['path']}")


def resolve_inputs(args):
    metadata = {}
    if args.kit:
        root = Path(args.kit).expanduser().resolve(strict=True)
        verify_kit(root)
        metadata = json.loads((root / "demo.json").read_text())
        metadata = dict(metadata)
        for key in ("video", "photos", "models"):
            metadata[key] = kit_path(root, metadata[key])
    values = {}
    for key in ("video", "photos", "models"):
        value = getattr(args, key) or metadata.get(key)
        if not value:
            raise ValueError(f"Supply --{key} or a complete --kit")
        values[key] = Path(value).expanduser().resolve(strict=True)
    if not values["video"].is_file():
        raise ValueError(
            "--video must be a local video file; live frames use the Python API"
        )
    if not values["photos"].is_dir() or not values["models"].is_dir():
        raise ValueError("--photos and --models must be directories")
    values["camera_id"] = args.camera_id or metadata.get("camera_id", "camera_01")
    values["clock_start"] = args.clock_start or metadata.get("clock_start", "00:00:00")
    values["config"] = Path(args.config).expanduser().resolve(strict=True)
    values["output"] = Path(args.output).expanduser().resolve()
    return values


def mount(path, destination, readonly=True):
    if any(c in str(path) for c in (",", "\n", "\r")):
        raise ValueError("Docker mount paths cannot contain commas or newlines")
    spec = f"type=bind,source={path},target={destination}"
    return ["--mount", spec + (",readonly" if readonly else "")]


def commands(backend, data, seconds):
    if not math.isfinite(seconds) or seconds < 0:
        raise ValueError("--seconds cannot be negative")
    if backend == "docker":
        prefix = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--gpus",
            "all",
            "--network",
            "none",
        ]
        if hasattr(os, "getuid"):
            prefix += ["--user", f"{os.getuid()}:{os.getgid()}"]
        prefix += mount(data["models"], "/models") + mount(data["photos"], "/photos")
        prefix += mount(data["video"], "/input-video") + mount(
            data["config"], "/run-config.json"
        )
        prefix += mount(data["output"], "/results", False)
        prefix += [IMAGE, "--models", "/models", "--config", "/run-config.json"]
        photos, video, gallery, output = (
            "/photos",
            "/input-video",
            "/results/gallery",
            "/results/demo",
        )
    else:
        binary = ROOT / f".venv-{backend}/bin/camex-faceid"
        if not binary.is_file():
            raise ValueError(f"Run bash install.sh --backend {backend} first")
        provider = (
            "CPUExecutionProvider" if backend == "cpu" else "CUDAExecutionProvider"
        )
        prefix = [
            str(binary),
            "--provider",
            provider,
            "--models",
            str(data["models"]),
            "--config",
            str(data["config"]),
        ]
        photos, video = str(data["photos"]), str(data["video"])
        gallery, output = str(data["output"] / "gallery"), str(data["output"] / "demo")
    return [
        prefix + ["enroll", "--photos", photos, "--output", gallery],
        prefix
        + [
            "demo",
            "--video",
            video,
            "--gallery",
            gallery,
            "--output",
            output,
            "--seconds",
            str(seconds),
            "--camera-id",
            data["camera_id"],
            "--clock-start",
            data["clock_start"],
        ],
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["docker", "cpu", "cuda"], default="docker")
    p.add_argument(
        "--kit", help="Extracted reproduction kit folder; checksums are verified"
    )
    for key in ("video", "photos", "models"):
        p.add_argument(f"--{key}")
    p.add_argument("--config", default=str(ROOT / "configs/default.json"))
    p.add_argument(
        "--output",
        required=True,
        help="Writable output folder outside the kit/source assets",
    )
    p.add_argument("--camera-id")
    p.add_argument("--clock-start")
    p.add_argument("--seconds", type=float, default=0, help="0 processes the full file")
    args = p.parse_args()
    data = resolve_inputs(args)
    if args.kit:
        kit = Path(args.kit).expanduser().resolve()
        if data["output"] == kit or kit in data["output"].parents:
            raise ValueError(
                "Choose an output directory outside the immutable test kit"
            )
    for key in ("video", "photos", "models"):
        if data["output"] == data[key] or data["output"] in data[key].parents:
            raise ValueError("Output cannot contain source assets")
    steps = commands(args.backend, data, args.seconds)
    data["output"].mkdir(parents=True, exist_ok=True)
    for step in steps:
        subprocess.run(step, check=True, cwd=ROOT)
    print(f"Video and results: {data['output'] / 'demo'}")


if __name__ == "__main__":
    main()
