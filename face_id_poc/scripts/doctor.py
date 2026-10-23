"""Check prerequisites without installing drivers, dependencies or models."""

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

from run_demo import ROOT, IMAGE, mount


def inspect_environment(models, provider):
    import cv2
    import onnxruntime as ort
    from camex_faceid.models import Networks

    env = os.environ.copy()
    env.pop("LD_LIBRARY_PATH", None)
    env.pop("LD_PRELOAD", None)
    candidates = [
        os.environ.get("FFMPEG_BIN"),
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
    ]
    encoder = next(
        (
            c
            for c in candidates
            if c
            and Path(c).is_file()
            and "libx264"
            in subprocess.run(
                [c, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                env=env,
            ).stdout
        ),
        None,
    )
    if not encoder:
        raise RuntimeError("Install FFmpeg with libx264 before running the demo")
    result = {
        "opencv": cv2.__version__,
        "onnxruntime": ort.__version__,
        "ffmpeg": encoder,
        "requested_provider": provider,
    }
    if models:
        result["verified_model_providers"] = Networks(models, provider).providers
        result["model_checksums"] = "passed"
    else:
        result["compiled_providers"] = ort.get_available_providers()
        result["note"] = (
            "Supply --models to test actual model/session initialization; this list alone does not prove GPU readiness"
        )
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["docker", "cpu", "cuda"], default="docker")
    p.add_argument("--models")
    p.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    a = p.parse_args()
    if a.inside:
        inspect_environment(
            a.models,
            "CPUExecutionProvider" if a.backend == "cpu" else "CUDAExecutionProvider",
        )
        return
    models = Path(a.models).expanduser().resolve(strict=True) if a.models else None
    if a.backend == "docker":
        if not shutil.which("docker"):
            raise RuntimeError(
                "Docker is missing; run setup on a Docker/NVIDIA host or choose CPU"
            )
        subprocess.run(
            ["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"], check=True
        )
        command = [
            "docker",
            "run",
            "--rm",
            "--pull",
            "never",
            "--gpus",
            "all",
            "--network",
            "none",
            "--entrypoint",
            "python",
        ]
        if models:
            command += mount(models, "/models")
        command += [IMAGE, "scripts/doctor.py", "--inside", "--backend", "cuda"]
        if models:
            command += ["--models", "/models"]
    else:
        python = ROOT / f".venv-{a.backend}/bin/python"
        if not python.is_file():
            raise RuntimeError(f"Run bash install.sh --backend {a.backend} first")
        command = [
            str(python),
            str(Path(__file__).resolve()),
            "--inside",
            "--backend",
            a.backend,
        ]
        if models:
            command += ["--models", str(models)]
    subprocess.run(command, check=True, cwd=ROOT)


if __name__ == "__main__":
    main()
