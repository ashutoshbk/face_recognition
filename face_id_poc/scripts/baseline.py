"""Same-model framewise comparison, not an identity-accuracy evaluator."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from camex_faceid.gallery import Gallery
from camex_faceid.models import Networks, sha256


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--models", required=True)
    p.add_argument("--gallery", required=True)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    out = Path(a.output)
    if out.exists():
        raise FileExistsError("Use a new baseline output path")
    n = Networks(a.models)
    g = Gallery(a.gallery)
    c = cv2.VideoCapture(a.video)
    count = 0
    faces_count = 0
    accepted = Counter()
    unique = {}
    latencies = []
    start = time.perf_counter()
    while True:
        ok, frame = c.read()
        if not ok:
            break
        t = time.perf_counter()
        for face in n.faces(frame):
            v, _ = n.embedding(frame, face["landmarks"])
            pid, score, margin = g.match(v)
            faces_count += 1
            for threshold in [0.3, 0.45]:
                key = str(threshold)
                if score >= threshold:
                    accepted[key] += 1
                    unique.setdefault(key, set()).add(pid)
        latencies.append((time.perf_counter() - t) * 1000)
        count += 1
    c.release()
    if count == 0:
        raise ValueError("No decoded frames")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "video_sha256": sha256(a.video),
                "processed_frames": count,
                "face_detections": faces_count,
                "accepted_face_frames_by_threshold": dict(accepted),
                "distinct_predicted_ids_by_threshold": {
                    k: len(v) for k, v in unique.items()
                },
                "latency_ms_p50": float(np.percentile(latencies, 50)),
                "latency_ms_p95": float(np.percentile(latencies, 95)),
                "inference_fps": count / (sum(latencies) / 1000),
                "wall_s": time.perf_counter() - start,
                "recognition_accuracy": None,
                "note": "Same SCRFD-2.5G and ArcFace-R50, full-frame detection, nearest gallery match per face; no head tracking, quality filtering or temporal voting. Not an exact reproduction of the client's buffalo_l default and not ground truth.",
            },
            indent=2,
        )
    )
    print("Baseline finished", count, "frames")


if __name__ == "__main__":
    main()
