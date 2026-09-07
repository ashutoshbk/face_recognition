"""Host-side integration boundary; decoding, model ownership and batching remain external."""

import json
from pathlib import Path

from camex_floor.adapter import from_mct_result
from camex_floor.runtime import FloorTrackingService


def create_floor_stage(project_directory):
    """Use the finished site's coordinate system, not independently copied matrices."""
    path = Path(project_directory) / "finished_site.json"
    site = json.loads(path.read_text())
    return FloorTrackingService(site)


def process_current_batch(floor_service, camera_results, identity_updates=()):
    """camera_results: (MCT result, exact BGR frame, current-frame evidence provider).

    The evidence provider returns one dictionary per track with pose_available,
    keypoints_xy, keypoints_conf and the current ReID embedding. Do not substitute
    a newer camera frame or compute identity prototypes from future frames.
    """
    packets = []
    for result, frame, evidence_provider in camera_results:
        packets.append(
            from_mct_result(
                result,
                frame,
                evidence_provider,
                identity_updates=identity_updates if not packets else (),
            )
        )
    return floor_service.process_batch(packets)
