"""Fresh batched detection, SCT, ReID and pose from only the current site videos."""

import json
from pathlib import Path
import time

import cv2
import numpy as np
import torch

from camex_floor.adapter import from_mct_result
from camex_floor.server import atomic_json
from camex_floor.detection_support import merge_pose_support, upper_body_agrees
from camex_mct.service import TrackingEngine
from camex_mct.types import FramePacket


def configuration(site, fps, detector_size=416):
    return {
        "site_id": site["site_id"],
        "cameras": list(site["cameras"]),
        "runtime": {"model_batch_size": len(site["cameras"]), "target_fps": fps},
        "detector": {
            "model": "/models/yolox_tiny.pth",
            "experiment": "/opt/camex/third_party/ByteTrack/exps/default/yolox_tiny.py",
            "input_size": detector_size,
            "confidence": 0.25,
            "nms": 0.45,
        },
        "tracker": {
            "track_high_threshold": 0.35,
            "track_low_threshold": 0.10,
            "new_track_threshold": 0.45,
            "track_buffer": 150,
            "match_threshold": 0.80,
            "proximity_threshold": 0.70,
            "appearance_threshold": 0.25,
            "minimum_box_area": 600,
        },
        "reid": {
            "source": "/opt/camex/third_party/deep-person-reid/torchreid/models/osnet.py",
            "model": "/models/osnet_ibn_x1_0_msmt17_combineall.pth",
            "architecture": "osnet_ibn_x1_0",
            "batch_sizes": [1, 8, 16, 32],
            "minimum_crop_width": 12,
            "minimum_crop_height": 30,
        },
        "global_identity": {
            "similarity_threshold": 0.77,
            "ambiguity_margin": 0.05,
            "active_timeout_seconds": 2.0,
            "gallery_retention_seconds": 1800,
            "same_camera_overlap_tolerance_seconds": 0.2,
            "minimum_track_observations": 6,
            "evidence_window_size": 30,
            "reassignment_interval_seconds": 0.5,
            "reassignment_grace_seconds": 15.0,
            "reassignment_threshold": 0.82,
            "reassignment_margin": 0.06,
            "record_reconciliation_enabled": True,
            "record_reconciliation_interval_seconds": 0.5,
            "record_reconciliation_similarity_threshold": 0.72,
            "record_reconciliation_margin": 0.04,
            "record_reconciliation_max_transition_seconds": 10.0,
            "record_reconciliation_min_cameras": 2,
            "record_reconciliation_min_observations": 8,
        },
    }


def pose_evidence(tracks, result):
    boxes = (
        result.boxes.xyxy.detach().cpu().numpy()
        if result.boxes is not None
        else np.zeros((0, 4))
    )
    if result.keypoints is None:
        return [{} for _ in tracks]
    points = result.keypoints.xy.detach().cpu().numpy()
    conf = result.keypoints.conf.detach().cpu().numpy()
    pairs = []
    for i, t in enumerate(tracks):
        a = np.asarray(t["bbox_xyxy"])
        for j, b in enumerate(boxes):
            overlap = np.maximum(
                0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])
            ).prod()
            union = (a[2:] - a[:2]).prod() + (b[2:] - b[:2]).prod() - overlap
            score = overlap / max(1e-9, union)
            if score >= 0.25 and upper_body_agrees(a, points[j], conf[j]):
                pairs.append((score, i, j))
    used_tracks = set()
    used_pose = set()
    out = [{} for _ in tracks]
    for score, i, j in sorted(pairs, reverse=True):
        if i in used_tracks or j in used_pose:
            continue
        used_tracks.add(i)
        used_pose.add(j)
        out[i] = {
            "keypoints_xy": points[j].tolist(),
            "keypoints_conf": conf[j].tolist(),
            "pose_box_iou": float(score),
        }
    return out


class PoseSupportedDetector:
    def __init__(self, detector):
        self.detector = detector
        self.test_size = detector.test_size
        self.pose_by_frame = {}

    def infer(self, frames):
        detections, timing = self.detector.infer(frames)
        for i, frame in enumerate(frames):
            pose = self.pose_by_frame.get(id(frame))
            if pose is None or pose.boxes is None or pose.keypoints is None:
                continue
            detections[i] = merge_pose_support(
                detections[i],
                pose.boxes.xyxy.detach().cpu().numpy(),
                pose.boxes.conf.detach().cpu().numpy(),
                pose.keypoints.conf.detach().cpu().numpy(),
                [frame.shape[1], frame.shape[0]],
                [self.test_size[1], self.test_size[0]],
                pose_keypoints=pose.keypoints.xy.detach().cpu().numpy(),
            )
        return detections, timing

    def close(self):
        self.detector.close()


def run(job):
    from ultralytics import YOLO

    site = json.loads(Path("/job/site.json").read_text())
    fps = job["fps"]
    duration = job["duration_s"]
    width = job["inference_width"]
    torch.manual_seed(42)
    np.random.seed(42)
    cv2.setNumThreads(1)
    config = configuration(site, fps, job.get("detector_size", 416))
    atomic_json("/job/inference_configuration.json", config)
    engine = TrackingEngine(config)
    engine.detector = PoseSupportedDetector(engine.detector)
    model = YOLO("/models/yolo11n-pose.pt")
    captures = {}
    indices = {}
    frames = {}
    for cid, c in site["cameras"].items():
        captures[cid] = cv2.VideoCapture(str(Path("/site") / c["video"]))
        indices[cid] = -1
        if not captures[cid].isOpened():
            raise ValueError(f"Cannot decode video {cid}")
    torch.cuda.reset_peak_memory_stats()
    times = []
    telemetry = []
    count = 0
    start = time.perf_counter()
    try:
        with Path("/job/observations.partial.jsonl").open("w") as sink:
            for tick in range(round(duration * fps)):
                before = time.perf_counter()
                packets = []
                for cid, c in site["cameras"].items():
                    cap = captures[cid]
                    wanted = round(
                        (tick / fps + c.get("video_start_seconds", 0))
                        * (cap.get(cv2.CAP_PROP_FPS) or fps)
                    )
                    while indices[cid] < wanted:
                        ok, frame = cap.read()
                        if not ok:
                            raise ValueError(
                                f"Video {cid} ended before the requested segment"
                            )
                        frames[cid] = frame
                        indices[cid] += 1
                    raw = frames[cid]
                    height = round(raw.shape[0] * width / raw.shape[1])
                    image = cv2.resize(raw, (width, height))
                    packets.append(FramePacket(cid, image, round(tick / fps * 1000)))
                pose_results = model.predict(
                    [p.frame for p in packets],
                    imgsz=960,
                    conf=0.18,
                    iou=0.65,
                    classes=[0],
                    device="cuda",
                    verbose=False,
                )
                engine.detector.pose_by_frame = {
                    id(p.frame): pose for p, pose in zip(packets, pose_results)
                }
                results = engine.process_batch(packets)
                updates = [e.to_dict() for e in engine.drain_identity_updates()]
                for packet, result, pose in zip(
                    packets, results, pose_results, strict=True
                ):
                    output = from_mct_result(result, packet.frame)
                    evidence = pose_evidence(output["tracks"], pose)
                    for track, extra in zip(output["tracks"], evidence):
                        track.update(extra)
                        track["pose_available"] = True
                    # Reuse the track's current smoothed appearance; no full-recording prototype.
                    tracker = engine.trackers[packet.camera_id]
                    by_local = {
                        tracker.raw_to_local.get(int(t.track_id)): t
                        for t in tracker.tracker.tracked_stracks
                    }
                    for track in output["tracks"]:
                        t = by_local.get(track["local_id"])
                        v = getattr(t, "smooth_feat", None)
                        if v is not None:
                            track["embedding"] = np.asarray(v, dtype=float).tolist()
                    output["upstream_identity_updates"] = updates
                    updates = []
                    sink.write(json.dumps(output, separators=(",", ":")) + "\n")
                    count += 1
                elapsed = time.perf_counter() - start
                times.append((time.perf_counter() - before) * 1000)
                if tick % 10 == 0:
                    atomic_json(
                        "/job/progress.json",
                        {
                            "stage": "Fresh tracking and pose",
                            "completed_s": tick / fps,
                            "total_s": duration,
                            "fraction": tick / max(1, round(duration * fps)),
                            "elapsed_s": elapsed,
                        },
                    )
                    telemetry.append(
                        {
                            "video_s": tick / fps,
                            "elapsed_s": elapsed,
                            "allocated_bytes": torch.cuda.memory_allocated(),
                            "reserved_bytes": torch.cuda.memory_reserved(),
                        }
                    )
                if tick % 100 == 0:
                    print(
                        f"Fresh inference {tick / fps:.1f}/{duration:.1f}s elapsed {elapsed:.1f}s",
                        flush=True,
                    )
        Path("/job/observations.partial.jsonl").replace("/job/observations.jsonl")
        atomic_json(
            "/job/inference_metrics.json",
            {
                "source": "raw videos; fresh inference",
                "frames": count,
                "duration_s": duration,
                "fps": fps,
                "elapsed_s": time.perf_counter() - start,
                "batch_p50_ms": float(np.median(times)),
                "batch_p95_ms": float(np.percentile(times, 95)),
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "gpu": torch.cuda.get_device_name(0),
                "torch": torch.__version__,
                "telemetry": telemetry,
            },
        )
    finally:
        engine.close()
        for cap in captures.values():
            cap.release()
