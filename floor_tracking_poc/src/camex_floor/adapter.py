"""Adapters to the existing MCT API, with optional current-frame evidence providers."""

import numpy as np
from .detection_support import upper_body_agrees


def from_mct_result(
    result, frame, evidence_provider=None, camera_epoch="0", identity_updates=None
):
    value = result.to_dict() if hasattr(result, "to_dict") else result
    tracks = [dict(t) for t in value["tracks"]]
    if evidence_provider is not None:
        evidence = evidence_provider(frame, tracks)
        if len(evidence) != len(tracks):
            raise ValueError("Evidence count differs from track count")
        for track, extra in zip(tracks, evidence):
            track.update(extra)
    packet = {
        "camera_id": value["camera_id"],
        "camera_epoch": str(camera_epoch),
        "timestamp_ms": int(value["timestamp_ms"]),
        "frame_size_wh": [frame.shape[1], frame.shape[0]],
        "tracks": tracks,
    }
    if identity_updates is not None:
        packet["upstream_identity_updates"] = [
            e.to_dict() if hasattr(e, "to_dict") else dict(e) for e in identity_updates
        ]
    return packet


class PoseEvidence:
    """Optional backend. Caller supplies an installed, authorized Ultralytics model.

    This module does not download weights or import Ultralytics itself.
    """

    def __init__(self, model, device="cpu", image_size=960):
        self.model = model
        self.device = device
        self.image_size = image_size

    def __call__(self, frame, tracks):
        r = self.model.predict(
            frame,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
            conf=0.18,
            classes=[0],
        )[0]
        if r.boxes is None or r.keypoints is None:
            return [{"pose_available": True} for _ in tracks]
        boxes = r.boxes.xyxy.detach().cpu().numpy()
        xy = r.keypoints.xy.detach().cpu().numpy()
        conf = r.keypoints.conf.detach().cpu().numpy()
        pairs = []
        for i, t in enumerate(tracks):
            a = np.array(t["bbox_xyxy"])
            for j, b in enumerate(boxes):
                low = np.maximum(a[:2], b[:2])
                high = np.minimum(a[2:], b[2:])
                intersection = np.maximum(0, high - low).prod()
                union = (
                    np.maximum(0, a[2:] - a[:2]).prod()
                    + np.maximum(0, b[2:] - b[:2]).prod()
                    - intersection
                )
                overlap = intersection / max(1e-9, union)
                if overlap >= 0.25 and upper_body_agrees(a, xy[j], conf[j]):
                    pairs.append((overlap, i, j))
        used_tracks = set()
        used_pose = set()
        out = [{"pose_available": True} for _ in tracks]
        for overlap, i, j in sorted(pairs, reverse=True):
            if i in used_tracks or j in used_pose:
                continue
            used_tracks.add(i)
            used_pose.add(j)
            out[i] = {
                "pose_available": True,
                "keypoints_xy": xy[j].tolist(),
                "keypoints_conf": conf[j].tolist(),
                "pose_box_iou": float(overlap),
            }
        return out
