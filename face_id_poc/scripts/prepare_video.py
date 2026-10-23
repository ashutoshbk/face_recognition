"""Create a separate indexed, constant-rate working clip without changing raw data."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--start", type=float, default=0)
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--fps", type=int, default=10)
    a = p.parse_args()
    source = Path(a.input).resolve()
    out = Path(a.output).resolve()
    if a.start < 0 or a.seconds <= 0 or a.fps <= 0:
        raise ValueError("Invalid time or FPS")
    if source == out:
        raise ValueError("Output must not overwrite the raw input")
    settings = {
        "source_sha256": digest(source),
        "start_seconds": a.start,
        "duration_seconds": a.seconds,
        "fps": a.fps,
    }
    marker = out.with_suffix(".source.json")
    if marker.exists() and out.exists():
        previous = json.loads(marker.read_text())
        if previous["settings"] == settings and previous["output_sha256"] == digest(
            out
        ):
            print("Completed clip reused")
            return
    if out.exists():
        raise FileExistsError("Choose a new output path")
    out.parent.mkdir(parents=True, exist_ok=True)
    indexed = out.with_suffix(".indexed.mkv")
    work_marker = out.with_suffix(".prepare.json")
    if work_marker.exists():
        if json.loads(work_marker.read_text()) != settings:
            raise ValueError("Existing preparation belongs to different input/settings")
    else:
        for artifact in [
            indexed,
            out.with_suffix(".indexed.partial.mkv"),
            out.with_suffix(".partial.mp4"),
        ]:
            if artifact.exists():
                raise FileExistsError(
                    "Unclaimed preparation file exists; choose a new output"
                )
        with work_marker.open("x") as stream:
            json.dump(settings, stream)
    if not indexed.exists():
        tmp = out.with_suffix(".indexed.partial.mkv")
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-nostdin",
                "-y",
                "-fflags",
                "+genpts",
                "-i",
                str(source),
                "-map",
                "0:v:0",
                "-c",
                "copy",
                "-avoid_negative_ts",
                "make_zero",
                str(tmp),
            ],
            check=True,
        )
        tmp.replace(indexed)
    tmp = out.with_suffix(".partial.mp4")
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-ss",
            str(a.start),
            "-i",
            str(indexed),
            "-t",
            str(a.seconds),
            "-vf",
            f"fps={a.fps}",
            "-c:v",
            "libx264",
            "-threads",
            "4",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-an",
            "-movflags",
            "+faststart",
            str(tmp),
        ],
        check=True,
    )
    tmp.replace(out)
    marker.write_text(
        json.dumps({"settings": settings, "output_sha256": digest(out)}, indent=2)
    )
    print(str(out))


if __name__ == "__main__":
    main()
