# Multi-Camera Tracking PoC Handoff

The integration package is isolated under `multicamera_tracking_poc/`. The
existing Face ID prototype and staff-photo directory are unchanged.

## Processing Boundary

- The platform owns RTSP connections, decoding, reconnection, credentials, and
  camera-health monitoring.
- The package accepts decoded BGR frames through a nonblocking Python API.
- Person detection is batched across available cameras.
- Each camera maintains an independent SCT state and local IDs.
- One site-wide appearance gallery assigns global IDs across cameras and can
  reconnect a person after disappearance.
- Results are versioned Python objects that serialize directly to JSON.

## Start Here

```bash
cd multicamera_tracking_poc
make setup
make test
make demo DATA_DIR=/private/path/to/synchronized_three_camera_clips
```

For platform integration, start with:

- `examples/platform_integration.py`
- `docs/INTEGRATION.md`
- `docs/EVENT_SCHEMA.md`
- `configs/live_20cam.example.yaml`

## Verified Baseline

- Three synchronized cameras: 60 seconds at 3 x 10 FPS with no submitted-frame
  replacement, stale timestamps, result-queue drops, or callback errors.
- Twenty-feed regression on Titan X Pascal: 71.62 aggregate FPS.
- The required 20 x 10 FPS target must be accepted on the Dell Pro Max Tower T2
  with its NVIDIA RTX 5000 Ada Generation 32 GB GPU and may need TensorRT
  optimization.
- The recordings did not include identity ground truth, so the run validates
  integration and throughput rather than ReID accuracy.

## Recommended Integration Sequence

1. Run unit tests and the supplied three-camera demo unchanged.
2. Connect three live decoder callbacks using `submit_frame()`.
3. Validate local/global ID events through the JSON-compatible result schema.
4. Expand to 20 camera IDs and run the RTX 5000 Ada throughput acceptance test.
5. Tune identity thresholds using labeled client footage.

The exact workstation specification and measurement checklist are in
`multicamera_tracking_poc/docs/TARGET_HARDWARE.md`.
