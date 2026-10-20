#!/usr/bin/env python3
"""Build with a minimal face_id_poc-shaped context, never private runtime data."""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


def copy_source(source, destination):
    if source.is_symlink():
        raise ValueError(f"Source symlinks are not allowed: {source.name}")
    if source.is_dir():
        destination.mkdir(parents=True, exist_ok=True)
        for child in source.iterdir():
            if child.name.startswith(".") or child.name in {"__pycache__", "node_modules"}:
                continue
            if child.suffix in {".pyc", ".onnx", ".mp4", ".sqlite", ".zip"}:
                continue
            copy_source(child, destination / child.name)
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="camex-office-demo:cuda12.2-ort1.18.0")
    args = parser.parse_args()
    office = Path(__file__).resolve().parents[1]
    root = office.parent
    if not (office / "app.py").is_file():
        parser.error("office_demo/app.py is not ready; finish the backend before building")
    selected = [office / "Dockerfile", *office.glob("requirements*.txt"),
                *office.glob("*.py"), office / "static", office / "scripts",
                root / "research_demo" / "vision", root / "research_demo" / "core.py"]
    with tempfile.TemporaryDirectory(prefix="camex-office-build-") as temp:
        context = Path(temp)
        for source in selected:
            copy_source(source, context / source.relative_to(root))
        shutil.copy2(office / ".dockerignore", context / ".dockerignore")
        subprocess.run(["docker", "build", "--platform", "linux/amd64", "--file",
                        "office_demo/Dockerfile", "--tag", args.tag, str(context)], check=True)


if __name__ == "__main__":
    main()
