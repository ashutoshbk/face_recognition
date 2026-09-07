"""Use reliable pose detections to support an independent person detector."""

import numpy as np


def overlap(a, b):
    inter = np.maximum(0, np.minimum(a[2:4], b[2:4]) - np.maximum(a[:2], b[:2])).prod()
    union = (
        np.maximum(0, a[2:4] - a[:2]).prod()
        + np.maximum(0, b[2:4] - b[:2]).prod()
        - inter
    )
    return float(inter / max(1e-9, union))


def upper_body_agrees(box, points, confidence):
    points = np.asarray(points, float)
    confidence = np.asarray(confidence, float)
    if points.shape != (17, 2) or confidence.shape != (17,):
        return False
    indices = [i for i in (0, 5, 6) if confidence[i] >= 0.5]
    if len(indices) < 2:
        return False
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1
    if min(width, height) <= 0:
        return False
    center = points[indices].mean(0)
    return bool(
        x1 - 0.10 * width <= center[0] <= x2 + 0.10 * width
        and y1 - 0.10 * height <= center[1] <= y1 + 0.70 * height
    )


def merge_pose_support(
    detections,
    pose_boxes,
    pose_scores,
    keypoint_scores,
    frame_wh,
    detector_wh,
    minimum_score=0.65,
    minimum_joints=5,
    pose_keypoints=None,
):
    """Input detector boxes are in detector coordinates; pose boxes are in frame pixels."""
    rows = [r.copy() for r in np.asarray(detections, dtype=np.float32).reshape(-1, 5)]
    scale = min(detector_wh[0] / frame_wh[0], detector_wh[1] / frame_wh[1])
    for pose_index, (box, score, joints) in enumerate(
        zip(pose_boxes, pose_scores, keypoint_scores)
    ):
        if (
            score < minimum_score
            or np.count_nonzero(np.asarray(joints) >= 0.5) < minimum_joints
        ):
            continue
        if not np.isfinite(box).all() or min(box[2] - box[0], box[3] - box[1]) <= 0:
            continue
        candidate = np.r_[np.asarray(box) * scale, float(score)].astype(np.float32)
        scores = [overlap(r, candidate) for r in rows]
        if scores and max(scores) >= 0.40:
            index = int(np.argmax(scores))
            agrees = pose_keypoints is None or upper_body_agrees(
                rows[index][:4] / scale, pose_keypoints[pose_index], joints
            )
            if agrees:
                rows[index][4] = max(rows[index][4], candidate[4])
            elif rows[index][4] < 0.55:
                rows[index] = candidate
            else:
                rows.append(candidate)
        else:
            rows.append(candidate)
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)
