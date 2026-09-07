# Integration Contract

## Boundaries

The platform owns live stream credentials, decoding, camera clocks, buffering and storage. This package owns camera geometry and the floor projection/fusion stage. Its included GPU file runner demonstrates fresh detection, independent per-camera SCT, global appearance ReID and pose. The file runner is not a production live stream gateway.

Calibrate using the same resolution, crop, orientation, lens setting and camera pose as the incoming frames. Changing zoom/pose/crop requires updated geometry. Distinguish raw image coordinates from rectified coordinates; the projector applies the declared distortion model.

## Inputs and Outputs

| Interface | Required information |
|---|---|
| `TrackingService` / `TrackingEngine` in `engines/mct/src/camex_mct` | Registered camera ID, exact BGR frame, timestamp; installed detector/tracker/ReID models |
| `from_mct_result` | MCT result, the exact corresponding frame, current-frame evidence provider and drained identity-correction events |
| Evidence per track | `pose_available`, 17 `keypoints_xy`, `keypoints_conf` when available, current ReID `embedding`; do not invent missing joints |
| `FloorTrackingService.process_batch` | Time-aligned camera packets using one saved site coordinate system |
| Returned result | Current floor tracks/positions/trails, observations including withheld positions, identity updates and operational metrics |

`examples/live_floor_adapter.py` demonstrates the CPU integration boundary. Run the GPU components in the included tested container or an equivalently validated isolated environment. The host UI virtual environment does not automatically expose the embedded MCT module as a separately installed application package.

## State and Identity

- Local IDs belong to a camera stream and its epoch. A reconnect/reset must not silently reuse another person's local history.
- Global IDs are hypotheses, not verified personal names. Handle `global_id_alias` events in the UI, database and search index so older fragments resolve consistently.
- Preserve timestamps and source frame dimensions. Do not attach pose/appearance from a later frame to an earlier tracking result.
- Use current/past evidence only. Do not calculate a completed-recording prototype and present retrospective corrections as real-time decisions.
- Position can be withheld when ground contact is unreliable. Do not turn a missing position into `(0, 0)` or assume every box bottom is a foot.
- Use one service/state per site. Camera numeric IDs reused at another site must not merge their identity galleries.

## Acceptance for Live Operation

Benchmark the actual camera count, crowd load, frame resolution, GPU, queueing policy and processing rate. Record per-camera delivery, replacements/drops, latency, memory and sustained behavior. Offline MP4 playback cadence is not a live throughput measurement. Twenty cameras at 10 FPS requires a separate test on the actual target hardware.

Do not use the UI loopback server as a publicly exposed production API without a separate authentication/security design.
