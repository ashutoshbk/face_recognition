"""Build and verify the declared GPU engine without touching another environment."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models-dir", type=Path, default=ROOT / "models")
    p.add_argument("--ui-only", action="store_true")
    a = p.parse_args()
    config = {
        "schema_version": "1.0",
        "backend": "docker",
        "models_dir": str(a.models_dir.resolve()),
        "gpu_image": "camex-spatial-pipeline:0.3.1",
        "mode": "ui_only" if a.ui_only else "full",
    }
    if not a.ui_only:
        if not shutil.which("docker"):
            raise SystemExit(
                "Docker with NVIDIA GPU support is required for the full engine."
            )
        subprocess.run(["docker", "info"], check=True, stdout=subprocess.DEVNULL)
        manifest = json.loads((ROOT / "models_manifest.json").read_text())
        missing = []
        for name, spec in manifest["models"].items():
            path = a.models_dir / spec["filename"]
            if not path.is_file():
                missing.append(spec["filename"])
                continue
            print("Verifying model:", name, flush=True)
            if sha(path) != spec["sha256"]:
                raise SystemExit(f"Model checksum failed: {name}")
        if missing:
            raise SystemExit(
                "Place the evaluated model bundle in --models-dir. Missing: "
                + ", ".join(missing)
            )
        subprocess.run(
            [
                "docker",
                "build",
                "-f",
                str(ROOT / "gpu/Dockerfile"),
                "-t",
                config["gpu_image"],
                str(ROOT),
            ],
            check=True,
        )
        config["gpu_image_id"] = subprocess.check_output(
            ["docker", "image", "inspect", config["gpu_image"], "--format", "{{.Id}}"],
            text=True,
        ).strip()
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--gpus",
                "all",
                "--network",
                "none",
                config["gpu_image"],
                "python",
                "/opt/camex/gpu/worker.py",
                "doctor",
            ],
            check=True,
        )
        config["model_hashes"] = {k: v["sha256"] for k, v in manifest["models"].items()}
        config["engine_source_hash"] = hashlib.sha256(
            "".join(
                sha(p) for p in sorted((ROOT / "gpu").glob("*")) if p.is_file()
            ).encode()
        ).hexdigest()
    (ROOT / "runtime.json").write_text(json.dumps(config, indent=2) + "\n")
    print(
        "Full engine verified."
        if not a.ui_only
        else "UI-only development installation; GPU jobs are unavailable."
    )


if __name__ == "__main__":
    main()
