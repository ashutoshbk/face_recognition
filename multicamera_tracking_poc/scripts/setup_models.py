from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

MODELS = {
    "yolox_tiny.pth": {
        "url": "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth",
        "sha256": "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0",
    },
    "osnet_ibn_x1_0_msmt17_combineall.pth": {
        "url": "https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth",
        "sha256": "2aca54945043301c82fe1a08bb22b045adb0c210673cb8319f0d4f44778f0566",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path, expected: str) -> None:
    if destination.exists():
        actual = sha256(destination)
        if actual != expected:
            raise RuntimeError(
                f"checksum mismatch for existing {destination}: {actual} != {expected}"
            )
        print(f"reusing verified model: {destination}")
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "camex-mct-setup/0.1"})
    with urllib.request.urlopen(request) as response, temporary.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    actual = sha256(temporary)
    if actual != expected:
        failed = destination.with_suffix(destination.suffix + ".checksum_failed")
        temporary.replace(failed)
        raise RuntimeError(f"checksum mismatch for {destination}: {actual} != {expected}")
    temporary.replace(destination)
    print(f"downloaded and verified: {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and export pinned model assets")
    parser.add_argument("--root", default=os.environ.get("CAMEX_MCT_ROOT"))
    parser.add_argument("--batch-sizes", nargs="*", type=int, default=[3, 20])
    parser.add_argument("--download-only", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve() if args.root else Path(__file__).resolve().parents[1]
    model_dir = root / "models"
    for filename, metadata in MODELS.items():
        download(metadata["url"], model_dir / filename, metadata["sha256"])

    if args.download_only:
        return
    third_party = Path(os.environ.get("CAMEX_MCT_THIRD_PARTY", root / "third_party")).resolve()
    experiment = third_party / "ByteTrack/exps/default/yolox_tiny.py"
    checkpoint = model_dir / "yolox_tiny.pth"
    exporter = root / "scripts/export_yolox_torchscript.py"
    for batch_size in args.batch_sizes:
        output = model_dir / f"yolox_tiny_b{batch_size}_416.torchscript.pt"
        subprocess.run(
            [
                sys.executable,
                str(exporter),
                "--experiment",
                str(experiment),
                "--checkpoint",
                str(checkpoint),
                "--input-size",
                "416",
                "--batch-size",
                str(batch_size),
                "--output",
                str(output),
            ],
            check=True,
        )


if __name__ == "__main__":
    main()
