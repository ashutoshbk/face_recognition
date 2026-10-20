#!/usr/bin/env python3
"""Supervise five near-simultaneous RTSP replays, stopping all if one fails."""
import argparse
import json
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import time
from urllib.parse import urlsplit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("fixture_dir", type=Path)
    parser.add_argument("--rtsp-base", default="rtsp://127.0.0.1:18555")
    args = parser.parse_args()
    os.umask(0o077)
    base = urlsplit(args.rtsp_base)
    if base.scheme != "rtsp" or not base.hostname or base.username or base.password or base.query or base.fragment:
        parser.error("Use a credential-free RTSP base for the isolated fixture server")
    if not shutil.which("ffmpeg"):
        parser.error("ffmpeg is required")
    root = args.fixture_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest.get("unique_sources") != 3 or manifest.get("old_tracking_results_used") is not False:
        parser.error("Expected a fixture produced by prepare_office_fixture.py")
    streams = [("camera_a", "camera_a"), ("camera_b", "camera_b"), ("camera_c", "camera_c"),
               ("camera_d_replay_a", "camera_a"), ("camera_e_replay_b", "camera_b")]
    for _, source in streams:
        clip = root / f"{source}.mp4"
        if not clip.is_file() or clip.is_symlink():
            parser.error(f"Missing regular fixture clip: {clip.name}")
    deadline = time.monotonic() + 30
    while True:
        try:
            with socket.create_connection((base.hostname, base.port or 554), timeout=1):
                break
        except OSError:
            if time.monotonic() >= deadline:
                parser.error("Fixture RTSP server did not become reachable in 30 seconds")
            time.sleep(0.5)
    processes = []
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        launched = time.monotonic()
        for name, source in streams:
            # No per-camera PTS reset: retain the 0 / 0.05 second source offsets.
            command = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin"]
            if not manifest.get("transcoded", False):
                command += ["-copyts"]
            command += ["-re", "-stream_loop", "-1", "-i", str(root / f"{source}.mp4"),
                       "-map", "0:v:0", "-an", "-c", "copy", "-avoid_negative_ts", "disabled",
                       "-rtsp_transport", "tcp", "-f", "rtsp", f"{args.rtsp_base.rstrip('/')}/{name}"]
            processes.append(subprocess.Popen(command))
        print(f"Publishing 3 unique clips + 2 labelled replays; launch spread {time.monotonic() - launched:.3f}s.", flush=True)
        print("Clip-relative timing only; process scheduling/loop boundaries are not NTP synchronization.", flush=True)
        while not stopping:
            for process in processes:
                if process.poll() is not None:
                    raise RuntimeError("A fixture publisher exited; stopping the entire replay group")
            time.sleep(0.25)
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
        deadline = time.monotonic() + 5
        for process in processes:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


if __name__ == "__main__":
    main()
