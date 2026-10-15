# Adapted from FoundationVision/ByteTrack; see PROVENANCE.md and LICENSE.
import numpy as np
import lap

from cython_bbox import bbox_overlaps as bbox_ious


HEAD_CENTER_GATE = 0.35
HEAD_MAX_SIZE_RATIO = 2.5
HEAD_AMBIGUITY_MARGIN = 0.10
FORBIDDEN_COST = 1e6


def head_geometry_distance(tracks, detections, thresh, use_score=True):
    """Bounded center/scale alternative to IoU, with conservative tie vetoes.

    Center motion comes from Kalman predictions. Size gates use the last real
    measurement, so Kalman scale lag/overshoot cannot relax those gates.
    Returns costs and ambiguous row/column masks without mutating tracks.
    """
    iou_cost = iou_distance(tracks, detections)
    ambiguous_tracks = np.zeros(len(tracks), dtype=bool)
    ambiguous_detections = np.zeros(len(detections), dtype=bool)
    if iou_cost.size == 0:
        return iou_cost, ambiguous_tracks, ambiguous_detections

    predicted = np.asarray([track.tlbr for track in tracks])
    observed_sizes = np.asarray([track.observed_tlwh[2:] for track in tracks])
    boxes = np.asarray([det.tlbr for det in detections])
    sizes = boxes[:, 2:] - boxes[:, :2]
    delta = (predicted[:, None, :2] + predicted[:, None, 2:]) / 2
    delta = delta - (boxes[None, :, :2] + boxes[None, :, 2:]) / 2
    small_size = np.minimum(observed_sizes[:, None, :], sizes[None, :, :])
    large_size = np.maximum(observed_sizes[:, None, :], sizes[None, :, :])
    positive = (small_size > 0).all(axis=2)
    denominator = np.maximum(small_size, np.finfo(float).eps)
    center_distance = np.linalg.norm(delta / denominator, axis=2)
    size_ratio = (large_size / denominator).max(axis=2)
    size_allowed = positive & np.isfinite(center_distance) & (size_ratio <= HEAD_MAX_SIZE_RATIO)

    scale_cost = np.log(np.maximum(size_ratio, 1)) / np.log(HEAD_MAX_SIZE_RATIO)
    geometry_cost = (0.80 * center_distance / HEAD_CENTER_GATE
                     + 0.10 * scale_cost + 0.10 * iou_cost)
    costs = np.where(center_distance <= HEAD_CENTER_GATE,
                     np.minimum(iou_cost, geometry_cost), iou_cost)
    if use_score:
        costs = fuse_score(costs, detections)

    available_tracks = np.array([not track.association_ambiguous for track in tracks])
    available_detections = np.array([not det.association_ambiguous for det in detections])
    allowed = size_allowed & available_tracks[:, None] & available_detections[None, :]
    costs = np.where(allowed, costs, FORBIDDEN_COST)

    # Include viable IoU-only edges in the tie check too. Otherwise disabling
    # the geometric bonus on a collision could still assign an arbitrary ID.
    candidates = allowed & (costs <= thresh)
    distances = np.where(candidates, center_distance, np.inf)
    near_row = candidates & (distances <= distances.min(axis=1)[:, None] + HEAD_AMBIGUITY_MARGIN)
    near_column = candidates & (distances <= distances.min(axis=0)[None, :] + HEAD_AMBIGUITY_MARGIN)
    row_ties = near_row.sum(axis=1) > 1
    column_ties = near_column.sum(axis=0) > 1
    ambiguous_tracks = row_ties | (near_column & column_ties[None, :]).any(axis=1)
    ambiguous_detections = column_ties | (near_row & row_ties[:, None]).any(axis=0)
    costs[ambiguous_tracks, :] = FORBIDDEN_COST
    costs[:, ambiguous_detections] = FORBIDDEN_COST
    return costs, ambiguous_tracks, ambiguous_detections


def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    matches, unmatched_a, unmatched_b = [], [], []
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    for ix, mx in enumerate(x):
        if mx >= 0:
            matches.append([ix, mx])
    unmatched_a = np.where(x < 0)[0]
    unmatched_b = np.where(y < 0)[0]
    matches = np.asarray(matches, dtype=int).reshape(-1, 2)
    return matches, unmatched_a, unmatched_b


def ious(atlbrs, btlbrs):
    """
    Compute cost based on IoU
    :type atlbrs: list[tlbr] | np.ndarray
    :type atlbrs: list[tlbr] | np.ndarray

    :rtype ious np.ndarray
    """
    ious = np.zeros((len(atlbrs), len(btlbrs)), dtype=float)
    if ious.size == 0:
        return ious

    ious = bbox_ious(
        np.ascontiguousarray(atlbrs, dtype=float),
        np.ascontiguousarray(btlbrs, dtype=float)
    )

    return ious


def iou_distance(atracks, btracks):
    """
    Compute cost based on IoU
    :type atracks: list[STrack]
    :type btracks: list[STrack]

    :rtype cost_matrix np.ndarray
    """

    if (len(atracks)>0 and isinstance(atracks[0], np.ndarray)) or (len(btracks) > 0 and isinstance(btracks[0], np.ndarray)):
        atlbrs = atracks
        btlbrs = btracks
    else:
        atlbrs = [track.tlbr for track in atracks]
        btlbrs = [track.tlbr for track in btracks]
    _ious = ious(atlbrs, btlbrs)
    cost_matrix = 1 - _ious

    return cost_matrix

def fuse_score(cost_matrix, detections):
    if cost_matrix.size == 0:
        return cost_matrix
    iou_sim = 1 - cost_matrix
    det_scores = np.array([det.score for det in detections])
    det_scores = np.expand_dims(det_scores, axis=0).repeat(cost_matrix.shape[0], axis=0)
    fuse_sim = iou_sim * det_scores
    fuse_cost = 1 - fuse_sim
    return fuse_cost
