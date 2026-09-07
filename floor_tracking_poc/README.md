# CAMEX Spatial Studio 0.3.1 - Integration Preview

Start with camera videos and a floor drawing. Add and calibrate cameras one by one, place them on the drawing, verify the mapping, then run fresh multi-camera tracking and generate a floor-map video.

All calibration, tracking, pose and rendering engines are supplied by this package's installer and declared model bundle. No previous CAMEX checkout, calibration ZIP, office project, saved observations or author environment is required.

Start with [HANDOFF.md](docs/HANDOFF.md). Read [VALIDATION.md](docs/VALIDATION.md) for measured evidence and remaining gates. This is a functional integration preview, not an accuracy-certified client release.

## Installation

Target: Ubuntu 22.04 or newer, x86-64, NVIDIA driver, Docker with NVIDIA Container Toolkit, Python 3.10-3.12 with venv, and internet during the initial image build. The workstation should have at least 12 GB GPU memory and 32 GB RAM; 32 GB VRAM provides additional room. Keep at least 30 GB free for image layers, environments, models and project output. Actual throughput depends on camera count, image size and crowd density.

Extract the separately supplied `CAMEX_Floor_Tracking_Models_0.3.1.zip`. It contains the five evaluated checkpoints under its `models/` directory. Inclusion is not deployment or redistribution clearance; review `MODEL_PROVENANCE.md`. From this repository's package folder, run:

```bash
cd floor_tracking_poc
bash install.sh --models-dir /path/to/CAMEX_Floor_Tracking_Models_0.3.1/models
bash launch.sh
```

Open **http://127.0.0.1:8770/** on that workstation. Through TeamViewer, open that address in the remote workstation's browser. The server listens on loopback, not the public network.

Replace the example model path with the actual extracted folder. Alternatively place the matching files in this package's ignored `models/` folder and run `bash install.sh`. A Git checkout alone contains no model binaries; the companion bundle is required. No earlier software package is needed.

The installer creates a local Python environment, verifies every model hash, builds a pinned GPU image containing GeoCalib, VGGT, YOLOX, BoT-SORT, OSNet, YOLO Pose and FFmpeg, and runs tests. It does not update the NVIDIA driver, change the existing platform environment, or stop other demos. It will fail clearly if prerequisites/model files are unavailable.

The same GPU image contains rendering dependencies, so the host does not need a separately configured FFmpeg installation. Runtime model jobs run with container networking disabled. Installation may download public source and dependencies; no site media are sent to those services.

For local UI development without an NVIDIA GPU, `bash install.sh --ui-only` installs only the UI and CPU tools. This mode cannot execute the complete pipeline and is not deployment acceptance.

## 1. Create a Site

1. Select **New project**.
2. Enter a project name and upload a floor-plan PNG/JPEG or PDF.
3. For PDF, select the page number.
4. Select **Create site**.

The application stores the drawing in a new project directory. Use a dimensioned drawing when metric positions are required. A conceptual or outdated drawing may support a visual preview but is not a surveyed coordinate system.

## 2. Add the First Camera

1. Select **Add** in the camera list.
2. Enter a unique camera ID, such as `entrance_01`, and a display name.
3. Upload that camera's video.
4. Open its **Video** tab. Inspect the recording and choose a sharp reference frame with useful floor/wall landmarks visible.
5. Use **Use video position** or enter the time, then **Extract reference frame**.

The reference-capture UI update opens **Reference** automatically after extraction and displays the saved frame time. The player retains that time rather than reloading frame zero. Unrelated saves preserve playback and a time being typed. Frame timestamps are quantized to the video's frame interval; the tool records both requested and decoded time and rejects an unverified seek. The **Reference**, **Video** and **Calibration review** tabs remain separate.

Reference selection is independent of the tracking segment. An empty frame later in the recording can be used to calibrate tracking that starts earlier, provided the camera's pose, zoom and crop did not change.

The current release supports up to 20 static cameras per project and uploads up to 20 GB per file. Cameras with changing PTZ/zoom, different floor levels or non-flat ground require appropriate separate geometry; do not treat them as one static floor homography.

## 3. Calibrate That Camera

Select **Calibrate camera**. The job runs:

1. GeoCalib intrinsic/distortion/gravity estimation.
2. Full-frame undistortion.
3. VGGT local reconstruction and pose estimation.
4. Gravity alignment and local point-layer generation.

The UI displays job state and a processing log. Once complete, inspect **Calibration review**, **Undistortion**, and **3D top view**.

If the result needs a floor constraint:

1. Return to **Reference**, which now shows the undistorted image.
2. Select **Select floor region**.
3. Click at least three vertices around a visible patch of actual floor. Avoid table tops, chair backs and people.
4. Select **Save floor region**.
5. Click **Calibrate camera** again. Keep **Gravity + floor check** unless the review justifies another leveling option.

A result without a supported floor plane is marked provisional. The operator must still verify its global projection. The UI can show **Floor points**, **All 3D points**, or **Image ground projection**. Elevated objects can stretch when projected onto a flat floor.

## 4. Place and Verify the Camera

1. Choose the camera's layer and opacity.
2. Use **Drag layer**, rotation delta and scale multiplier for initial placement.
3. Select **Fit point**. Click a fixed floor landmark in the reference, then the corresponding point on the drawing.
4. For a direct floor mapping, collect at least four well-separated correspondences and choose **Fit floor mapping**. Six to ten are preferable. RANSAC can reject inconsistent correspondences, but four points alone cannot determine which one is incorrect.
5. Alternatively use **Fit local layer** to fit the local 3D-derived layer's translation, rotation and scale. This requires at least two separated correspondences and useful local geometry.
6. Use **Known distance** and **Measure on plan** to establish pixels per metre. Check it against another dimension.
7. Select **Holdout point** and collect at least two new, separated correspondences not used in fitting. Inspect their residuals.
8. Select **Approve camera** only when the independent checks pass.

Default approval requires a verified scale and holdout median <= 0.25 m and maximum <= 0.50 m. These are engineering acceptance limits, not a claimed accuracy. Broader image coverage requires more distributed checks. Do not generate holdouts from the fitted transform itself.

Camera parameters include estimated intrinsics, local rotation/floor geometry, and a global-pose estimate when metric landmarks are supplied. The output explicitly distinguishes a floor homography from a validated full physical pose. Learned geometry and approximate plan landmarks still need review.

Use **Save**, **Lock camera** and **Hide camera** as needed. Changing camera geometry invalidates its approval. A new reference frame clears correspondence data that depended on the old rectification.

## 5. Add the Other Cameras

Repeat **Add -> Reference -> Calibrate -> Inspect -> Align -> Verify -> Save** for each camera. All cameras use the same code and validation rules. Existing cameras remain available while another is edited.

Then select **Finish calibration**. The app validates every registered camera and saves a finished project. **Save preview** is available for visual investigation when scale or independent checks are still pending; preview outputs are marked as draft.

You can close the browser, reopen the application, select the saved project and use **MCT runner**. No copying matrix values or editing scripts is required.

### Remove a Camera (8 September UI Update)

Select the camera, unlock it if needed, then choose **Remove camera** at the bottom of the left panel. Confirm the project and camera in the dialog. **Cancel** makes no change. Removal is blocked while that project's job is running.

This removes the camera from the active project and future runs, not from disk. Its videos, calibration files and previous results remain, with a recovery snapshot under `workspaces/<project-id>/camera_archive/`. The UI does not yet provide automatic restore. The finished export is invalidated; finish calibration or save preview again before running the revised project. Other cameras' saved calibration is preserved.

New projects start with no cameras. Switching projects clears the previous images, overlays, logs and output. After installing an interface update, save any pending point edits and refresh the browser once.

These controls are included in version 0.3.1. Earlier 0.3.0 review archives are not a substitute for this version.

## 6. Synchronize and Run MCT

1. Open **MCT runner** for the saved project.
2. Select segment start, duration, FPS, inference width and detection detail. **Detailed (640)** gives the detector more input detail; **Balanced (416)** reduces its compute cost. Neither is a guaranteed accuracy setting. Benchmark the chosen profile on your views and hardware.
3. Set each camera's source offset if the videos start at different real-world times.
4. Select **Preview synchronization** and inspect the corresponding views. Recording timestamps alone may not establish precise synchronization.
5. Select **Run MCT and render**.

The pipeline decodes the selected frames, runs batched person detection, independent SCT per camera, appearance ReID, batched pose, reliable ground-contact estimation, floor projection, global fusion and smoothing. Appearance vectors come from the current tracking history, not a completed future recording. No precomputed office observations or identity corrections are required.

The finished job provides a video, tracking events and inference metrics. The mosaic adapts to the configured camera count. It displays local/global IDs, a common floor map and smoothed trajectories.

Ground contacts prefer visible ankles. Torso estimates use calibrated gravity when available; recent valid contacts have a short expiry. If pose ran but cannot support a floor contact, the box bottom is not automatically treated as a foot. Unsupported static detections remain unverified; uncertain floor positions may be withheld. Strong pose evidence can supplement small-person detections, but upper-body agreement is required so a chair cannot simply borrow a nearby person's joints. Implausible position jumps start a new trail segment instead of drawing a long unsupported connecting line. These decisions are auditable and can reduce recall in difficult views. Spatial identity merges require appearance and temporal evidence as well as proximity. Initial IDs can change as evidence accumulates.

## Persistence, Retries and Caches

Project data live under `workspaces/<project-id>/`. The site configuration, source videos, reference frames and all job artifacts are kept together.

- Unchanged completed jobs can be reused, with a reuse indication.
- Changed reference images, model/image versions and relevant calibration inputs invalidate the corresponding job fingerprint.
- Incomplete jobs are not treated as complete. Failed stages have logs and can be retried.
- Neural observations may be reused for a rendering retry only when their inference fingerprint and saved checksum match.
- Model jobs share a process lock to avoid overlapping GPU work from two instances using the same workspace.
- A restarted application marks unfinished jobs interrupted. Retry explicitly; do not assume interrupted work finished successfully.

Preserve source videos. The app writes new job directories rather than overwriting previous results. Keep the complete workspace directory when moving a calibrated site, including the media and calibration assets referenced by its JSON.

## Integration Boundary

The included `engines/mct/src/camex_mct` exposes the existing `TrackingService` and frame-result schema. `src/camex_floor` exposes floor projection/fusion and `from_mct_result`. The file runner is one input adapter; the client platform can supply decoded live frames and corresponding timestamps through the same core stages. `examples/live_floor_adapter.py` shows the boundary. Supply current pose evidence, current ReID embeddings and drained identity-correction events, not only boxes and IDs, to retain the full floor-fusion behavior. The example is an adapter, not a complete live decoder or scheduling implementation.

Use an isolated GPU worker with this tested dependency set. Keep camera credentials and stream ownership in the platform. Preserve the exact frame associated with each timestamp, and apply identity-correction events consistently to the database, UI and search index. Full 20-camera live throughput requires a separate target-hardware test; playback FPS of an offline video is not a throughput measurement.

## Troubleshooting

- **Model missing/hash mismatch:** restore the evaluated bundle matching `models_manifest.json`; do not substitute an unrelated checkpoint.
- **GPU unavailable:** repair the driver/Container Toolkit with the machine administrator. The installer does not change system drivers.
- **Insufficient free GPU memory:** retry in an agreed idle window; do not terminate another team's workload.
- **Distorted or tilted reconstruction:** choose a clearer frame, inspect undistortion, then select an actual floor region and recalibrate.
- **Large landmark residual:** correct the correspondence, verify the drawing, or use direct floor fitting. More smoothing cannot fix calibration error.
- **Finish rejected:** read each camera's holdout/scale status. Use preview only for explicitly provisional work.
- **Run rejected:** ensure every video covers the selected segment plus offset and that calibration has been finished or saved as preview.
- **Inference/render failure:** inspect the job's worker and render logs. A failed job is not a successful demonstration.

## Distribution

Keep source, models and site data separate. Do not commit model binaries, videos, faces, embeddings, calibration outputs, credentials or workspaces to Git. Model manifests and applicable notices accompany the private handoff. The model bundle does not grant additional model/data usage rights; see `MODEL_PROVENANCE.md` before the intended deployment.

For the release's actual pass/fail status and remaining data/hardware checks, read the accompanying acceptance report. A working operator workflow does not by itself establish quantitative identity accuracy or surveyed localization.
