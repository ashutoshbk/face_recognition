"""Replay recorded observations through the exact live service and render a review."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import cv2
import numpy as np

from .geometry import digest, load_site
from .runtime import FloorTrackingService
from .server import atomic_json


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def batches(path):
    timestamp = None
    current = []
    with Path(path).open() as f:
        for line in f:
            p = json.loads(line)
            if timestamp is not None and p["timestamp_ms"] < timestamp:
                raise ValueError("Replay input must be chronological")
            if current and p["timestamp_ms"] != timestamp:
                yield current
                current = []
            timestamp = p["timestamp_ms"]
            current.append(p)
        if current:
            yield current


def color(gid):
    values = [
        (197, 146, 65),
        (101, 176, 91),
        (68, 165, 231),
        (185, 114, 184),
        (94, 181, 217),
        (188, 161, 105),
    ]
    return values[
        int(hashlib.sha256(str(gid).encode()).hexdigest()[:4], 16) % len(values)
    ]


def label(img, text, x, y, col=(50, 66, 73), scale=0.48):
    width, height = cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0]
    x = max(0, min(int(x), img.shape[1] - width - 10))
    y = max(height + 7, min(int(y), img.shape[0] - 4))
    cv2.rectangle(img, (x, y - height - 6), (x + width + 8, y + 4), col, -1)
    cv2.putText(
        img,
        str(text),
        (x + 4, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        (250, 252, 252),
        1,
        cv2.LINE_AA,
    )


class Mosaic:
    def __init__(self, site, root, output, fps, width=1920, height=1080):
        self.site = site
        self.root = Path(root)
        self.fps = fps
        self.width = width
        self.height = height
        self.floor = cv2.imread(str(self.root / site["floor"]["asset"]))
        if self.floor is None:
            raise ValueError("Floor asset missing")
        self.ids = list(site["cameras"])
        self.captures = {}
        self.last_frames = {}
        self.indices = {}
        for cid, c in site["cameras"].items():
            if c.get("video"):
                cap = cv2.VideoCapture(str(self.root / c["video"]))
                if not cap.isOpened():
                    raise ValueError(f"Cannot open video {cid}")
                self.captures[cid] = cap
                self.indices[cid] = -1
        self.proc = subprocess.Popen(
            [
                os.environ.get("CAMEX_FFMPEG", "ffmpeg"),
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-f",
                "rawvideo",
                "-pixel_format",
                "bgr24",
                "-video_size",
                f"{width}x{height}",
                "-framerate",
                str(fps),
                "-i",
                "pipe:0",
                "-an",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output),
            ],
            stdin=subprocess.PIPE,
        )

    def frame(self, result, relative_seconds):
        canvas = np.full((self.height, self.width, 3), (242, 245, 246), np.uint8)
        label(canvas, "CAMEX | GLOBAL FLOOR TRACKING", 18, 30, scale=0.66)
        label(
            canvas,
            f"{relative_seconds:05.1f}s | {len(result['tracks'])} people | {len(self.ids)} cameras",
            self.width - 460,
            30,
            scale=0.55,
        )
        if result["preview"]:
            label(
                canvas, "DRAFT CALIBRATION", self.width // 2 - 130, 30, (61, 123, 170)
            )
        panel_width = int(self.width * 0.34)
        cols = 1 if len(self.ids) <= 3 else 2
        rows = int(np.ceil(len(self.ids) / cols))
        cellw = panel_width // cols
        cellh = (self.height - 65) // rows
        for i, cid in enumerate(self.ids):
            c = self.site["cameras"][cid]
            x = (i % cols) * cellw + 10
            y = 55 + (i // cols) * cellh
            cap = self.captures.get(cid)
            if cap:
                desired = int(
                    round(
                        (relative_seconds + c.get("video_start_seconds", 0))
                        * (cap.get(cv2.CAP_PROP_FPS) or self.fps)
                    )
                )
                while self.indices[cid] < desired:
                    ok, frame = cap.read()
                    if not ok:
                        raise ValueError(f"Video {cid} ended early")
                    self.last_frames[cid] = frame
                    self.indices[cid] += 1
                frame = self.last_frames[cid].copy()
            else:
                frame = cv2.imread(str(self.root / c["reference"]))
            native_h, native_w = frame.shape[:2]
            resize = min((cellw - 20) / native_w, (cellh - 35) / native_h)
            frame = cv2.resize(frame, (int(native_w * resize), int(native_h * resize)))
            fw, fh = frame.shape[1], frame.shape[0]
            # Events are in their declared input frame coordinates; scale once for drawing.
            for obs in result["observations"]:
                if obs["camera_id"] != cid or obs.get("bbox_xyxy") is None:
                    continue
                event_size = obs.get("frame_size_wh", [fw, fh])
                scale = np.array([fw / event_size[0], fh / event_size[1]])
                box = (np.array(obs["bbox_xyxy"]).reshape(2, 2) * scale).astype(int)
                gid = obs.get("global_id")
                col = color(gid) if gid else (130, 140, 144)
                if obs["status"] in ("probation", "unverified_static"):
                    continue
                cv2.rectangle(
                    frame, tuple(box[0]), tuple(box[1]), col, max(2, fw // 500)
                )
                label(
                    frame,
                    f"L{obs['local_id']} | G{gid}"
                    + (" | map held" if obs.get("floor_px") is None else ""),
                    box[0, 0],
                    box[0, 1] - 6,
                    col,
                    0.45,
                )
            s = min((cellw - 20) / fw, (cellh - 35) / fh)
            rw, rh = int(fw * s), int(fh * s)
            tile = cv2.resize(frame, (rw, rh))
            canvas[y + 25 : y + 25 + rh, x : x + rw] = tile
            label(canvas, c.get("label", cid), x, y + 18, scale=0.47)
        available_w = self.width - panel_width - 35
        available_h = self.height - 88
        s = min(available_w / self.floor.shape[1], available_h / self.floor.shape[0])
        w, h = int(self.floor.shape[1] * s), int(self.floor.shape[0] * s)
        x = panel_width + (available_w - w) // 2 + 10
        y = 65 + (available_h - h) // 2
        canvas[y : y + h, x : x + w] = cv2.resize(self.floor, (w, h))
        for t in result["tracks"]:
            col = color(t["global_id"])
            trail = t["trail_px"]
            segments = []
            seg = []
            for sample in trail:
                if seg and sample[0] - seg[-1][0] > 1500:
                    segments.append(seg)
                    seg = []
                seg.append(sample)
            if seg:
                segments.append(seg)
            for seg in segments:
                if len(seg) < 2:
                    continue
                p = np.asarray([z[1:] for z in seg], float)
                if len(p) > 2:
                    p[1:-1] = (p[:-2] + 2 * p[1:-1] + p[2:]) / 4
                points = np.round(p * s + [x, y]).astype(np.int32)
                cv2.polylines(canvas, [points], False, col, 3, cv2.LINE_AA)
            p = tuple(np.round(np.array(t["floor_px"]) * s + [x, y]).astype(int))
            cv2.circle(canvas, p, 9, (40, 54, 61), -1, cv2.LINE_AA)
            cv2.circle(canvas, p, 7, col, -1, cv2.LINE_AA)
            label(canvas, "G" + str(t["global_id"]), p[0] + 12, p[1] - 9, col)
        return canvas

    def close(self):
        self.proc.stdin.close()
        code = self.proc.wait()
        for cap in self.captures.values():
            cap.release()
        if code:
            raise RuntimeError("Video encoding failed")


def replay(
    site_path,
    events,
    output,
    allow_draft=False,
    fps=10,
    realtime=False,
    video=True,
    progress_file=None,
    expected_duration=None,
):
    site = load_site(site_path)
    root = Path(site_path).resolve().parent
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    code_hash = digest(
        {p.name: file_hash(p) for p in Path(__file__).parent.glob("*.py")}
    )
    fingerprint = digest(
        {
            "site": site,
            "events": file_hash(events),
            "fps": fps,
            "draft": allow_draft,
            "code": code_hash,
            "video": video,
        }
    )
    marker = output / "manifest.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if (
            old.get("fingerprint") == fingerprint
            and (output / "results.jsonl").is_file()
            and (not video or (output / "demo.mp4").is_file())
        ):
            return old
        raise ValueError(
            "Output already contains a different run; choose a new output directory"
        )
    service = FloorTrackingService(
        site, allow_draft=allow_draft, session_id=fingerprint[:16]
    )
    mosaic = None
    latencies = []
    updates = []
    ids = set()
    count = 0
    start = time.perf_counter()
    first = None
    try:
        if video:
            mosaic = Mosaic(site, root, output / "demo.part.mp4", fps)
        with (output / "results.part.jsonl").open("w") as sink:
            for group in batches(events):
                if first is None:
                    first = group[0]["timestamp_ms"]
                result = service.process_batch(group)
                sizes = {p["camera_id"]: p["frame_size_wh"] for p in group}
                for obs in result["observations"]:
                    obs["frame_size_wh"] = sizes[obs["camera_id"]]
                latencies.append(result["processing_ms"])
                updates.extend(result["identity_updates"])
                ids.update(t["global_id"] for t in result["tracks"])
                sink.write(json.dumps(result, allow_nan=False) + "\n")
                seconds = (result["timestamp_ms"] - first) / 1000
                if realtime:
                    time.sleep(max(0, seconds - (time.perf_counter() - start)))
                if mosaic:
                    frame = mosaic.frame(result, seconds)
                    mosaic.proc.stdin.write(frame.tobytes())
                    if count in (0, 30, 240, 740, 837, 1146, 1199):
                        cv2.imwrite(str(output / f"frame_{count:04d}.jpg"), frame)
                count += 1
                if count % 100 == 0:
                    if progress_file:
                        atomic_json(
                            progress_file,
                            {
                                "stage": "Rendering floor map",
                                "completed_s": count / fps,
                                "total_s": expected_duration or count / fps,
                                "fraction": min(
                                    1,
                                    (count / fps) / (expected_duration or count / fps),
                                ),
                            },
                        )
                    print(f"Replay {seconds:.1f}s / {count} batches", flush=True)
    finally:
        if mosaic:
            mosaic.close()
    if video:
        (output / "demo.part.mp4").replace(output / "demo.mp4")
    (output / "results.part.jsonl").replace(output / "results.jsonl")
    service.save_state(output / "state.json")
    result = {
        "status": "complete",
        "fingerprint": fingerprint,
        "batches": count,
        "camera_count": len(site["cameras"]),
        "duration_seconds": count / fps,
        "elapsed_seconds": time.perf_counter() - start,
        "floor_latency_p50_ms": float(np.median(latencies)),
        "floor_latency_p95_ms": float(np.percentile(latencies, 95)),
        "identity_updates": updates,
        "raw_floor_ids_seen": len(ids),
        "canonical_floor_ids_seen": len({service.resolve_global_id(g) for g in ids}),
        "metrics": service.metrics,
        "preview": allow_draft,
        "limitations": [
            "Replay includes recorded SCT and pose observations. This measures the floor runtime; detector/ReID/pose GPU throughput is a separate benchmark.",
            "No floor-position or identity ground truth; these are operational and visual checks.",
        ],
    }
    atomic_json(marker, result)
    return result
