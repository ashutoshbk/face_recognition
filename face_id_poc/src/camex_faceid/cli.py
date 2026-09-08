import argparse
import hashlib
import json
import subprocess
import time
import os
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import psutil

from . import __version__
from .gallery import Gallery, enroll
from .models import Networks, MODEL_HASHES, sha256
from .io_utils import claim_directory, FileClock


def source_hash():
    h = hashlib.sha256()
    for path in sorted(Path(__file__).parent.rglob("*.py")):
        h.update(str(path.relative_to(Path(__file__).parent)).encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def gpu_memory():
    try:
        p = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
        )
        # This is all compute-process memory, not attribution to this process alone.
        return sum(
            float(line.split(",")[-1]) for line in p.stdout.splitlines() if line.strip()
        )
    except Exception:
        return None


def render_run(args, net, gallery, config):
    from .pipeline import FaceIDService
    from .render import Renderer

    output = Path(args.output)
    encoder_env = os.environ.copy()
    encoder_env.pop("LD_LIBRARY_PATH", None)
    encoder_env.pop("LD_PRELOAD", None)
    candidates = [
        os.environ.get("FFMPEG_BIN"),
        shutil.which("ffmpeg"),
        "/usr/bin/ffmpeg",
    ]
    ffmpeg = None
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            enc = subprocess.run(
                [candidate, "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                env=encoder_env,
            )
            if "libx264" in enc.stdout:
                ffmpeg = candidate
                break
    if ffmpeg is None:
        raise RuntimeError("An FFmpeg build with libx264 is required; set FFMPEG_BIN")
    fingerprint_data = {
        "video_sha256": sha256(args.video),
        "gallery": gallery.manifest["fingerprint"],
        "models": MODEL_HASHES,
        "config": config,
        "source_sha256": source_hash(),
        "seconds": args.seconds,
        "camera_id": args.camera_id,
        "clock_start": args.clock_start,
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_data, sort_keys=True).encode()
    ).hexdigest()
    done = output / "metrics.json"
    if done.exists():
        previous = json.loads(done.read_text())
        if (
            previous.get("fingerprint") == fingerprint
            and (output / "demo.mp4").exists()
            and sha256(output / "demo.mp4") == previous.get("output_sha256")
        ):
            print(json.dumps({"status": "completed run reused", "output": str(output)}))
            return
        raise ValueError(
            "Different run exists at output path; choose a new output directory"
        )
    run_lock = claim_directory(
        output, fingerprint, protected=[args.video, args.gallery]
    )
    (output / "evidence").mkdir(exist_ok=True)
    capture = cv2.VideoCapture(str(args.video))
    if not capture.isOpened():
        raise ValueError("Cannot open video")
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("Missing video frame rate")
    width, height = int(capture.get(3)), int(capture.get(4))
    output_fps = config["fps"]
    clock = FileClock(fps)
    if output_fps > fps:
        raise ValueError("Processing FPS cannot exceed source FPS")
    service = FaceIDService(net, gallery, config, args.camera_id)
    renderer = Renderer(camera_name=args.camera_id, clock_start=args.clock_start)
    process = subprocess.Popen(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            "1920x1080",
            "-r",
            str(output_fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-threads",
            "3",
            "-preset",
            "veryfast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output / "demo.partial.mp4"),
        ],
        stdin=subprocess.PIPE,
        env=encoder_env,
    )
    latencies = []
    states = Counter()
    names = Counter()
    all_tracks = set()
    events_count = 0
    peak_gpu = None
    peak_rss = 0
    index = 0
    emitted = 0
    next_time = 0.0
    start = time.perf_counter()
    inference_s = 0
    evidence_written = set()
    try:
        with (output / "tracks.jsonl").open("w") as tracks_out, (
            output / "events.jsonl"
        ).open("w") as events_out:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                pts = capture.get(cv2.CAP_PROP_POS_MSEC)
                ts = clock.timestamp(index, pts)
                index += 1
                if args.seconds and ts >= args.seconds * 1000:
                    break
                if ts + 0.01 < next_time:
                    continue
                timestamp_ms = round(ts)
                next_time += 1000 / output_fps
                r = service.process(frame, timestamp_ms)
                latencies.append(r["processing_ms"])
                inference_s += r["processing_ms"] / 1000
                for t in r["tracks"]:
                    states[t["recognition_state"]] += 1
                    all_tracks.add(t["track_id"])
                    if t["person_id"]:
                        names[t["person_id"]] += 1
                tracks_out.write(json.dumps(r, allow_nan=False) + "\n")
                for event in r["events"]:
                    events_count += int(event["event"] == "identity_confirmed")
                    events_out.write(json.dumps(event, allow_nan=False) + "\n")
                    if event["event"] != "identity_confirmed":
                        continue
                    key = (event["track_id"], event["person_id"])
                    e = service.evidence.get(event["track_id"])
                    if e and key not in evidence_written:
                        evidence_written.add(key)
                        stem = f"track_{key[0]}_{event['timestamp_ms']}"
                        cv2.imwrite(
                            str(output / "evidence" / (stem + "_observed.jpg")),
                            e["crop"],
                        )
                        cv2.imwrite(
                            str(output / "evidence" / (stem + "_reference.jpg")),
                            e["reference"],
                        )
                        (output / "evidence" / (stem + ".json")).write_text(
                            json.dumps(event, indent=2)
                        )
                rendered = renderer.render(frame, r, service.evidence)
                if rendered.shape != (1080, 1920, 3):
                    raise ValueError("Invalid renderer dimensions")
                process.stdin.write(rendered.tobytes())
                emitted += 1
                if emitted % 100 == 0 or emitted == 1:
                    mem = gpu_memory()
                    if mem is not None:
                        peak_gpu = max(peak_gpu or 0, mem)
                    peak_rss = max(peak_rss, psutil.Process().memory_info().rss)
                    print(
                        json.dumps(
                            {
                                "frame": emitted,
                                "source_s": timestamp_ms / 1000,
                                "heads": len(r["tracks"]),
                                "confirmed_events": events_count,
                                "inference_fps": round(emitted / inference_s, 2),
                                "gpu_compute_mib": mem,
                            }
                        ),
                        flush=True,
                    )
                if emitted in [1, 100, 500, 1000, 1500, 2000, 2500, 2900]:
                    cv2.imwrite(str(output / f"preview_{emitted:04d}.jpg"), rendered)
        process.stdin.close()
        code = process.wait()
        if code:
            raise RuntimeError(f"Video encoder failed: {code}")
        if emitted == 0:
            raise ValueError("No frames processed")
        (output / "demo.partial.mp4").replace(output / "demo.mp4")
        elapsed = time.perf_counter() - start
        metrics = {
            "schema_version": "1.0",
            "version": __version__,
            "fingerprint": fingerprint,
            "inputs": fingerprint_data,
            "providers": net.providers,
            "source_resolution": [width, height],
            "source_fps": fps,
            "processed_frames": emitted,
            "output_duration_s": emitted / output_fps,
            "output_fps": output_fps,
            "wall_time_s": elapsed,
            "inference_time_s": inference_s,
            "unique_frame_inference_fps": emitted / inference_s,
            "end_to_end_fps_including_render": emitted / elapsed,
            "processing_latency_ms": {
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
            },
            "gpu_all_compute_process_peak_mib": peak_gpu,
            "process_peak_rss_bytes": peak_rss,
            "track_count": len(all_tracks),
            "recognition_track_frame_states": dict(states),
            "predicted_identity_track_frames": dict(names),
            "confirmation_events": events_count,
            "diagnostics": service.diagnostics,
            "recognition_accuracy": None,
            "accuracy_note": "No client-confirmed identity ground truth supplied. Names are model predictions, not verified identifications.",
            "output_sha256": sha256(output / "demo.mp4"),
        }
        done.write_text(json.dumps(metrics, indent=2))
        print(
            json.dumps(
                {
                    "status": "complete",
                    "output": str(output),
                    "frames": emitted,
                    "inference_fps": metrics["unique_frame_inference_fps"],
                }
            ),
            flush=True,
        )
    finally:
        capture.release()
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        if process.poll() is None:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
        run_lock.close()


def main():
    parser = argparse.ArgumentParser(
        description="CAMEX single-camera head tracking and Face ID"
    )
    parser.add_argument(
        "--models",
        required=True,
        help="Folder with the three checksum-verified ONNX models",
    )
    parser.add_argument("--config", default="configs/default.json")
    parser.add_argument(
        "--provider",
        choices=["CUDAExecutionProvider", "CPUExecutionProvider"],
        default="CUDAExecutionProvider",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    e = sub.add_parser("enroll")
    e.add_argument("--photos", required=True)
    e.add_argument("--output", required=True)
    r = sub.add_parser("demo")
    r.add_argument("--video", required=True)
    r.add_argument("--gallery", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--seconds", type=float, default=0)
    r.add_argument("--camera-id", default="camera_01")
    r.add_argument("--clock-start", default="00:00:00")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text())
    cv2.setNumThreads(2)
    net = Networks(args.models, args.provider, config["face_threshold"])
    if args.command == "enroll":
        result = enroll(net, args.photos, args.output, config)
        print(
            json.dumps(
                {"enrolled": len(result["people"]), "rejected": result["rejected"]}
            )
        )
    else:
        render_run(args, net, Gallery(args.gallery), config)


if __name__ == "__main__":
    main()
