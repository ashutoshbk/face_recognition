from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from camex_mct import FrameResult, IdentityAliasEvent, TrackingService
from camex_mct.service import load_config

COLORS = (
    (82, 220, 154),
    (88, 178, 255),
    (238, 122, 96),
    (196, 112, 238),
    (80, 214, 234),
    (238, 188, 77),
)


def identity_color(global_id: int | None) -> tuple[int, int, int]:
    return (170, 170, 170) if global_id is None else COLORS[global_id % len(COLORS)]


def draw_frame(frame: np.ndarray, result: FrameResult, label: str) -> np.ndarray:
    canvas = frame.copy()
    for track in result.tracks:
        color = identity_color(track.global_id)
        x1, y1, x2, y2 = (int(value) for value in track.bbox_xyxy)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 3)
        global_label = "G?" if track.global_id is None else f"G{track.global_id}"
        text = f"L{track.local_id} / {global_label}"
        cv2.rectangle(canvas, (x1, max(0, y1 - 30)), (x1 + 150, y1), color, -1)
        cv2.putText(
            canvas,
            text,
            (x1 + 5, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (15, 15, 15),
            2,
            cv2.LINE_AA,
        )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 42), (24, 24, 24), -1)
    cv2.putText(
        canvas,
        f"{label}  |  {result.camera_id}  |  {len(result.tracks)} tracks",
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return canvas


def render_dashboard(
    frames: dict[str, np.ndarray],
    results: dict[str, FrameResult],
    cameras: list[dict],
    aggregate_fps: float,
) -> np.ndarray:
    width, tile_height, header_height, footer_height = 1920, 360, 64, 56
    tile_width = width // 3
    dashboard = np.full(
        (header_height + tile_height + footer_height, width, 3),
        (24, 22, 20),
        dtype=np.uint8,
    )
    cv2.putText(
        dashboard,
        "CAMEX 3-Camera Live SCT + Global ReID",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    for index, camera in enumerate(cameras):
        camera_id = camera["camera_id"]
        drawn = draw_frame(frames[camera_id], results[camera_id], camera["label"])
        drawn = cv2.resize(drawn, (tile_width, tile_height))
        left = index * tile_width
        dashboard[header_height : header_height + tile_height, left : left + tile_width] = drawn
    footer_text = (
        f"Input 3 x 10 FPS | effective {aggregate_fps:05.1f} FPS | "
        "local IDs are camera-specific | global IDs are site-wide"
    )
    cv2.putText(
        dashboard,
        footer_text,
        (20, header_height + tile_height + 36),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (210, 220, 225),
        2,
        cv2.LINE_AA,
    )
    return dashboard


def main() -> None:
    parser = argparse.ArgumentParser(description="Three-camera SCT/MCT demo")
    parser.add_argument("--config", default="configs/demo_3cam.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="outputs/demo_3cam")
    parser.add_argument("--duration", type=int, default=60)
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    sources = config["demo_sources"]
    camera_labels = {
        item["camera_id"]: item.get("label", item["camera_id"]) for item in config["cameras"]
    }

    captures: dict[str, cv2.VideoCapture] = {}
    for source in sources:
        path = data_dir / source["filename"]
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            raise RuntimeError(f"unable to open demo input: {path}")
        captures[source["camera_id"]] = capture

    target_fps = int(config["runtime"]["target_fps"])
    total_batches = args.duration * target_fps
    video_path = output_dir / "camex_3cam_sct_mct_demo.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        target_fps,
        (1920, 480),
    )
    if not writer.isOpened():
        raise RuntimeError(f"unable to create dashboard video: {video_path}")

    events_path = output_dir / "track_events.jsonl"
    identity_updates_path = output_dir / "identity_updates.jsonl"
    metrics_path = output_dir / "demo_metrics.json"
    identity_updates: list[IdentityAliasEvent] = []
    service = TrackingService(config, on_identity_update=identity_updates.append)
    service.start()
    started = time.perf_counter()
    processed_frames = 0
    try:
        with events_path.open("w", encoding="utf-8") as events_file:
            for frame_index in range(total_batches):
                timestamp_ms = frame_index * 1000 // target_fps
                frames: dict[str, np.ndarray] = {}
                for source in sources:
                    camera_id = source["camera_id"]
                    ok, frame = captures[camera_id].read()
                    if not ok:
                        raise RuntimeError(f"demo input ended early: {camera_id}")
                    frames[camera_id] = frame
                    service.submit_frame(camera_id, frame, timestamp_ms)

                results: dict[str, FrameResult] = {}
                deadline = time.monotonic() + 30
                while len(results) < len(sources):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("timed out waiting for tracking results")
                    result = service.poll_result(timeout=remaining)
                    if result.timestamp_ms == timestamp_ms:
                        results[result.camera_id] = result

                for source in sources:
                    result = results[source["camera_id"]]
                    events_file.write(json.dumps(result.to_dict(), separators=(",", ":")) + "\n")
                processed_frames += len(sources)
                elapsed = time.perf_counter() - started
                writer.write(
                    render_dashboard(
                        frames,
                        results,
                        [
                            {
                                "camera_id": source["camera_id"],
                                "label": camera_labels[source["camera_id"]],
                            }
                            for source in sources
                        ],
                        processed_frames / max(elapsed, 1e-9),
                    )
                )
    finally:
        service.close()
        writer.release()
        for capture in captures.values():
            capture.release()

    elapsed = time.perf_counter() - started
    identity_updates_path.write_text(
        "".join(
            json.dumps(update.to_dict(), separators=(",", ":")) + "\n"
            for update in identity_updates
        ),
        encoding="utf-8",
    )
    metrics = {
        "status": "passed",
        "camera_count": len(sources),
        "duration_seconds": args.duration,
        "frames_processed": processed_frames,
        "aggregate_effective_fps": processed_frames / elapsed,
        "wall_duration_seconds": elapsed,
        "video": str(video_path),
        "events": str(events_path),
        "identity_updates": str(identity_updates_path),
        **service.summary(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
