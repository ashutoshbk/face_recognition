#!/usr/bin/env python3
"""Prepare private, one-minute video-only fixtures without inference or enrollment."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

SOURCE = Path("/Volumes/SamsungT7/AINLab_Office_3D_Tracking_Visualization/inputs/videos")
CAMERAS = ("camera_a", "camera_b", "camera_c")
DURATION = 60.0


def probe(path):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
        "format=start_time,duration:stream=codec_name,start_time,duration,avg_frame_rate,time_base",
        "-of", "json", str(path)], check=True, capture_output=True, text=True)
    value = json.loads(result.stdout)
    if len(value.get("streams", [])) != 1:
        raise ValueError(f"Expected a video stream in {path.name}")
    stream = value["streams"][0]
    if stream.get("codec_name") not in {"h264", "hevc"}:
        raise ValueError(f"Unsupported RTSP copy codec in {path.name}")
    for field in ("start_time", "duration"):
        number = float(stream[field])
        if not math.isfinite(number):
            raise ValueError(f"Invalid {field} in {path.name}")
    return value


def digest(path):
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def keyframe_at(path, timestamp):
    result = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0", "-skip_frame", "nokey",
        "-show_frames", "-show_entries", "frame=best_effort_timestamp_time", "-of", "json",
        str(path)], check=True, capture_output=True, text=True)
    points = [float(frame["best_effort_timestamp_time"])
              for frame in json.loads(result.stdout).get("frames", [])
              if "best_effort_timestamp_time" in frame]
    matching = [point for point in points if abs(point - timestamp) <= 0.001]
    if not matching:
        raise ValueError(f"{path.name}: requested start {timestamp:.3f}s is not a keyframe. "
                         "Use --start-offset 0 or a common GOP boundary; stream copy cannot "
                         "make an arbitrary frame-exact cut without re-encoding.")
    return matching[0]


def write_json(path, data):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(data, stream, indent=2, allow_nan=False)
        stream.write("\n")
    path.chmod(0o600)


def camera_rows(base):
    names = [
        ("camera_a", "Reception (recorded fixture)", None),
        ("camera_b", "Workspace 1 (recorded fixture)", None),
        ("camera_c", "Workspace 2 (recorded fixture)", None),
        ("camera_d_replay_a", "REPLAY of camera_a (not a fourth camera)", "camera_a"),
        ("camera_e_replay_b", "REPLAY of camera_b (not a fifth camera)", "camera_b"),
    ]
    return [{"id": name, "name": label, "url": f"{base}/{name}",
             "fixture": True, "replay_of": replay,
             "file": f"{replay or name}.mp4"} for name, label, replay in names]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=SOURCE)
    parser.add_argument("--start-offset", type=float, default=0.0,
                        help="Common source-PTS offset in seconds; must align with each source's GOP")
    parser.add_argument("--output", type=Path, help="New private output directory; never overwritten")
    parser.add_argument("--check-only", action="store_true", help="Probe only; no copies or writes")
    parser.add_argument("--transport-profile", choices=["copy", "closed-gop"], default="copy")
    parser.add_argument("--encoder", choices=["libx264", "h264_nvenc"], default="libx264")
    args = parser.parse_args()
    os.umask(0o077)
    if not math.isfinite(args.start_offset) or args.start_offset < 0:
        parser.error("--start-offset must be a finite, nonnegative number")
    if not shutil.which("ffprobe") or (not args.check_only and not shutil.which("ffmpeg")):
        parser.error("Install FFmpeg/ffprobe before preparing fixtures")
    sources = {name: args.source_dir.resolve() / f"{name}.mp4" for name in CAMERAS}
    metadata, starts = {}, {}
    for name, path in sources.items():
        if not path.is_file() or path.is_symlink():
            parser.error(f"Missing regular source file: {path}")
        metadata[name] = probe(path)
        stream = metadata[name]["streams"][0]
        start = float(stream["start_time"])
        # Preserve the source phase relative to the common clip timeline.
        if not 0 <= start <= 0.1 or start + float(stream["duration"]) < args.start_offset + DURATION:
            parser.error(f"{name} does not cover the requested common 60-second source interval")
        try:
            starts[name] = keyframe_at(path, args.start_offset + start) if args.start_offset and args.transport_profile == "copy" else args.start_offset+start
        except ValueError as error:
            parser.error(str(error))
    if args.check_only:
        print(json.dumps({"source_interval_seconds": [args.start_offset, args.start_offset + DURATION],
                          "sources": metadata,
                          "camera_clock_or_ntp_validated": False}, indent=2))
        return
    if args.output is None:
        parser.error("--output is required unless --check-only is used")
    output = args.output.expanduser().absolute()
    if output.exists() or output.is_symlink():
        parser.error("Output exists; use a new directory to preserve the previous fixture")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".office-fixture-", dir=output.parent) as temp:
        stage = Path(temp)
        records = []
        for name, source in sources.items():
            destination = stage / f"{name}.mp4"
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-n"]
            if args.transport_profile == "copy":
                command += ["-copyts"]
                if args.start_offset:
                    command += ["-seek_timestamp", "1", "-ss", str(starts[name])]
                command += ["-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
                           "-to", str(args.start_offset + DURATION), "-c", "copy",
                           "-output_ts_offset", str(-args.start_offset), "-avoid_negative_ts", "disabled"]
            else:
                if args.start_offset:
                    command += ["-ss", str(args.start_offset)]
                command += ["-i", str(source), "-map", "0:v:0", "-an", "-sn", "-dn",
                            "-t", str(DURATION), "-vf", "fps=20,setpts=PTS-STARTPTS",
                            "-c:v", args.encoder, "-g", "20", "-bf", "0", "-pix_fmt", "yuv420p"]
                command += ["-preset", "veryfast", "-crf", "15", "-x264-params", "scenecut=0:keyint=20:min-keyint=20"] if args.encoder == "libx264" else ["-preset", "fast", "-rc", "constqp", "-qp", "18", "-forced-idr", "1"]
            command += ["-map_metadata", "-1", "-movflags", "+faststart", str(destination)]
            subprocess.run(command, check=True)
            actual = probe(destination)
            src_stream, dst_stream = metadata[name]["streams"][0], actual["streams"][0]
            if args.transport_profile == "copy" and abs(float(src_stream["start_time"]) - float(dst_stream["start_time"])) > 0.001:
                raise ValueError(f"Timestamp offset was not preserved for {name}")
            end = float(dst_stream["start_time"]) + float(dst_stream["duration"])
            if not DURATION - 0.15 <= end <= DURATION + 0.25:
                raise ValueError(f"Unexpected stream-copy boundary for {name}: {end}")
            destination.chmod(0o600)
            records.append({"camera": name, "source": str(source), "source_bytes": source.stat().st_size,
                            "source_sha256": digest(source), "source_probe": metadata[name],
                            "file": destination.name, "sha256": digest(destination),
                            "fixture_probe": actual, "command": command[:-1] + [destination.name]})
        write_json(stage / "manifest.json", {
            "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
            "source_interval_seconds": [args.start_offset, args.start_offset + DURATION],
            "output_timestamp_origin_source_seconds": args.start_offset,
            "unique_sources": 3, "published_streams": 5,
            "transport_profile": args.transport_profile, "transcoded": args.transport_profile != "copy",
            "encoder": args.encoder if args.transport_profile != "copy" else None,
            "alignment": "shared clip-relative interval; one-second closed GOP and no B-frames" if args.transport_profile != "copy" else "shared clip-relative PTS, preserved source start offsets; stream-copy boundaries",
            "camera_clock_or_ntp_validated": False, "live_capture": False,
            "identity_inference_performed": False, "staff_enrollment_performed": False,
            "old_tracking_results_used": False, "sources": records})
        for target, base in (("host", "rtsp://127.0.0.1:18555"), ("docker", "rtsp://mediamtx:8554")):
            write_json(stage / f"cameras.{target}.json", {"schema_version": 1, "cameras": camera_rows(base)})
        # Rename only after every clip and metadata file has passed verification.
        stage.rename(output)
    print(f"Prepared 3 private clips / 5 labelled streams in {output}; no enrollment or inference.")


if __name__ == "__main__":
    main()
