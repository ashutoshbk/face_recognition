# Handoff: 0.3.1 Integration Preview

This is a separate application/component folder. It does not replace the repository's existing `main.py`, staff-photo directory, or the earlier SCT/ReID pull request. It contains a pinned copy of the MCT engine so its clean installation does not depend on that earlier PR being merged. Future shared-engine updates should be coordinated explicitly.

## Three Deliveries

| Delivery | Contents | Where it belongs |
|---|---|---|
| This Git branch | Application, upstream engine source/locks, installer, tests, text manual, integration example and validation notes | Client repository |
| `CAMEX_Floor_Tracking_Test_Materials_0.3.1.zip` | Raw camera clips, exact floor PNG, illustrated manual, operator videos, reference sample, operator point guide and checksums | Separate private local folder; never Git |
| `CAMEX_Floor_Tracking_Models_0.3.1.zip` | Five evaluated checkpoints, model manifest and provenance notes | Separate model folder; never Git |

The model bundle is a required input, not an optional example. The test-material bundle is only required to reproduce the office walkthrough; a new site supplies its own videos and drawing. Both bundles are provided separately by the project lead. No private host path or previous author's environment is required.

## Install From Git

On the prepared Ubuntu/NVIDIA/Docker workstation:

```bash
git clone https://github.com/ashutoshbk/face_recognition.git
cd face_recognition
git switch codex/camex-floor-tracking-v0.3.1
cd floor_tracking_poc
bash install.sh --models-dir /path/to/CAMEX_Floor_Tracking_Models_0.3.1/models
bash launch.sh
```

Replace `/path/to/...` with the actual extracted model-bundle location. The installer verifies every checkpoint hash and creates isolated CPU/GPU environments. Docker GPU access and the system NVIDIA driver must already work. Initial installation downloads pinned public dependencies; runtime model jobs use disabled container networking.

Open `http://127.0.0.1:8770/` on that workstation. Through remote desktop, use the remote browser. If that port is already occupied, use `bash launch.sh --port 8790` and open the corresponding address. Do not stop another demo.

## Reproduce the Office Example

Read the private kit's `START_HERE.md` and `OPERATOR_POINT_GUIDE.md`. Start a new project using `inputs/floor.png`, then add the three videos from `inputs/videos/` one camera at a time. Do not substitute the uncropped original PDF when following the supplied point-coordinate examples.

Choose the recorded reference times, run actual calibration, select floor regions, then pair camera-image points with matching plan points. Turn Drag layer off before entering pairs. Use the supplied point guide as approximate operator input, not independent ground truth. No calibration matrices, pose records, embeddings or expected IDs are automatically injected by the application.

The recorded example has unverified metric geometry and uses **Save preview**. Real deployment needs a trustworthy drawing scale and independent holdouts; do not fabricate them to remove the draft warning. The pipeline can then produce a fresh two-minute video, tracking events and metrics.

The kit contains both the original visual reference and the output generated during this version's acceptance test. Treat them as review examples, not a requirement to reproduce literal numeric IDs or a claim of ground-truth accuracy.

## New Site

Use the same code and UI with the new site's drawing/videos. Only operator calibration, scale, correspondence points and synchronization settings should differ. Do not reuse office coordinates, chair regions, person IDs or event times.

## Platform Integration

See [INTEGRATION.md](INTEGRATION.md) and `examples/live_floor_adapter.py`. The current fully tested workflow is file-video preparation and processing. Live decoder ownership, frame batching and target-load acceptance remain integration tasks. Do not call the file runner a complete RTSP service.

## Review Status

Read [VALIDATION.md](VALIDATION.md) before making deployment or accuracy claims. The UI and pipeline can be integrated and investigated while quality/metric/site-transfer gates remain open. Model provenance and application validation are separate matters.
