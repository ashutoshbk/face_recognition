# CAMEX Spatial Studio

## Operator Manual - Version 0.3.1

**From camera videos and a floor plan to a multi-camera floor-tracking video**

For the person performing calibration and the colleague reproducing the workflow at a new site. This manual covers the 0.3.1 integration-preview interface and its latest capture/removal fixes.

**The essential sequence**

Create a project -> add one camera -> choose a frame -> calibrate and inspect -> set floor scale -> align and verify -> approve or retain draft -> repeat for each camera -> finish/save preview -> synchronize videos -> run MCT -> review and download.

You do **not** need to use every button. Most first-time operators can follow the numbered steps, using **Fit floor mapping** when they have recognizable floor landmarks. **Fit local layer** is an alternative for a useful reconstructed top-view layer, not an additional mandatory step.

**Current release status:** This is a functional review candidate. The office demonstration is not proof of equal accuracy at another site. Difficult detections and identity handoffs, independent metric calibration, second-site transfer and target-hardware performance still require validation. A **DRAFT CALIBRATION** video is a preview, not an approved physical-position result.

### Contents

1. Prepare the inputs and open the tool
2. Understand the screen and create a project
3. Add a camera and choose its reference frame
4. Run calibration and inspect the result
5. Refine the visible floor and choose a leveling method
6. Set the floor plan's real-world scale
7. Position the layer by dragging, rotating and scaling
8. Fit camera-to-plan point correspondences
9. Verify with independent holdout points
10. Approve, lock, add the next camera and finish
11. Synchronize videos and configure the MCT runner
12. Run processing and retrieve the output
13. Read the tracking video and diagnose problems
14. Button reference: navigation and left panel
15. Button reference: camera inspector and runner
16. Follow the recorded example
17. Troubleshooting, backup and handoff checklist

This guide explains the UI, not the setup of the client's live-stream decoder. Screenshots are examples from the recorded office workflow; your camera positions and point choices must come from your own environment.

<!-- pagebreak -->

## 1. Prepare the Inputs and Open the Tool

### Have these items ready

| Item | What to prepare |
|---|---|
| Floor plan | PNG, JPEG or a PDF page showing the floor you will track. Prefer an up-to-date, dimensioned drawing. |
| Camera videos | One decodable video per fixed physical camera. H.264 MP4 is the safest choice for both processing and browser playback. |
| Common time interval | Recordings that cover the same real period, with enough time for the demonstration. |
| Measured dimension | At least one trustworthy distance on the drawing, in metres. Check a second dimension too. |
| Floor landmarks | Corners, tile intersections, door-threshold endpoints or column bases visible in both the camera image and plan. |
| Computing environment | Prepared Ubuntu/NVIDIA/Docker workstation, GitHub source and the separately supplied model bundle. |

For a first test, use two or three cameras and a 30-second segment. After reviewing it, generate the longer video. More cameras are a capacity setting, not a guarantee of real-time performance.

**Supported geometry:** static camera pose and zoom, a single approximately flat floor per mapping. PTZ movement, multiple levels, stairs and uneven ground need additional geometry; one flat-floor mapping does not solve them automatically.

### Open an existing installation

1. On the workstation running the tool, open its browser.
2. Open the address printed by the launcher.
3. Look for **Verified GPU engine** in the top-right corner.
4. Use the project dropdown to choose an existing project, or **New project** for a different environment.

The normal launch address is `http://127.0.0.1:8770/`. The existing review installation uses `http://127.0.0.1:8781/`. Through TeamViewer, open this address in the **remote workstation's browser**. On another computer, localhost refers to that computer; access requires an existing approved tunnel or remote-desktop session.

### Install only when necessary

After checking out the source branch and extracting the model bundle, enter `face_recognition/floor_tracking_poc` in Terminal and run:

```bash
bash install.sh --models-dir /path/to/CAMEX_Floor_Tracking_Models_0.3.1/models
bash launch.sh
```

Replace the example path with the actual extracted model folder. An integrator should prepare Docker GPU access, the NVIDIA driver and Python prerequisites first. The installer creates isolated environments; it does not install drivers or stop another demo. Initial dependency installation needs internet. Runtime containers do not upload site media.

**Do not run the installer again just to open a saved project.** Use the launcher. Keep the launcher/server running while working in the browser.

<!-- pagebreak -->

## 2. Understand the Screen and Create a Project

The illustrated example is supplied in the separate private test-material bundle.
### Four areas to recognize

| Screen area | Purpose |
|---|---|
| Top bar | Select/create a project; switch between **Calibration** and **MCT runner**. |
| Left panel | Select the camera; set the drawing scale; choose and position overlays; lock/hide a camera layer. |
| Centre | The shared floor plan. Click matching landmarks here, or drag the selected camera layer. |
| Right panel | The selected camera's image/video, calibration, fit/check points, parameters and logs. |

The **camera title on the right** tells you which camera your next action will affect. Select a camera in the list before editing. Other cameras' overlays are deliberately much fainter than the selected camera's overlay.

### Create the new site

1. Click **New project** at the top. On an empty screen, **Create project** opens the same form.
2. Enter a descriptive **Project name**, such as `Building B - Ground Floor`.
3. Choose your **Floor plan** file.
4. For a PDF, set **PDF page (first page is 1)**. This field is irrelevant for a PNG/JPEG.
5. Click **Create site**. **Cancel** closes the form without creating the project.
6. Confirm the correct drawing is visible in the centre.

**Expected result:** the project is created with no calibrated cameras yet. Create a separate project for a new building or floor; do not move office camera layers onto an unrelated site's drawing.

Use a desktop display for precise point placement. A narrow screen can display the tool, but requires scrolling and makes accurate clicking harder.

<!-- pagebreak -->

## 3. Add a Camera and Choose Its Reference Frame

### Register one physical camera

1. In the left **CAMERAS** section, click **Add**.
2. Enter a unique **Camera ID**, for example `entrance_01`. Use letters, numbers, underscores or hyphens; no spaces.
3. Enter a readable **Display name**, for example `Main entrance`.
4. Select the camera's **Video file**, then submit the Add camera form.
5. Wait for upload and decoding to finish. Do not repeatedly click the submit button.
6. Select the new camera in the list and check its title on the right.

The first decoded frame is selected initially. The file picker accepts several extensions, but this does not mean every CCTV codec is supported. A video that plays in VLC may still fail in the browser. Use a compatible export; changing the filename extension does not convert it.

### Choose a useful calibration frame

1. On the right, open **Video**.
2. Play or scrub to a sharp frame. Pause it.
3. Choose a view with visible floor and permanent architectural landmarks. Fewer people and less motion blur are preferable.
4. Click **Use video position**. This copies the current player time into **Reference time (seconds)**; it does not extract the frame yet.
5. Alternatively, type a time directly into **Reference time (seconds)**.
6. Click **Extract reference frame**.
7. **Reference** opens automatically with the saved image and its captured timestamp. Returning to Video retains this position rather than jumping to frame zero.

**Expected result:** the selected frame is shown, and recalibration is required. A small difference between requested and captured time is normal because frames are discrete. Unlock the camera before capture; replacing existing calibration asks for confirmation. Extraction alone is not calibration.

### Three different kinds of time

| Setting | Meaning |
|---|---|
| Reference time | Chooses the image used for camera calibration. |
| Runner Start | Chooses where the tracking segment begins on the configured common timeline. |
| Camera offset | Aligns the source video with that timeline. |

A reference frame at 60 seconds can calibrate a tracking run starting at 0 seconds, **if the camera did not move or change zoom/crop**.

**Important:** extracting a new reference clears the camera's fit points and mapping and invalidates its calibration. Do this before alignment, not as a harmless refresh after finishing. Save a backup before intentionally rebuilding an already useful camera.

<!-- pagebreak -->

## 4. Run Calibration and Inspect the Result

### First calibration pass

1. Select the camera you just added.
2. Leave the leveling selector at **Gravity + floor check** initially.
3. Click **Calibrate camera** once.
4. Wait for the job to complete. The GPU action buttons are disabled while a project job is active.
5. Read **Processing log** if progress appears slow or a failure is reported. Model loading can take time before a measured progress fraction is available.

### What the tool is doing

| Stage | Easy-English meaning |
|---|---|
| Intrinsics and distortion | Estimates how the lens turns light rays into pixels, including lens curvature. |
| Undistortion | Produces a rectified image to reduce that curvature. |
| VGGT reconstruction | Estimates local 3D structure and camera geometry from the selected image. |
| Gravity and floor alignment | Tries to make the reconstructed floor horizontal for a useful top view. |
| Global alignment | Not automatic at this stage. You will place or fit this local result on your uploaded floor plan. |

### Inspect the output

1. Open **Calibration review**.
2. Click **Undistortion** to inspect the lens-correction review.
3. Click **3D top view** to inspect the reconstructed scene from above.
4. Return to **Reference** when you want to select floor regions or point correspondences.

After calibration, **Reference is the rectified image** used for point fitting. Click the image displayed there. Do not copy raw-video pixel coordinates into rectified-image correspondences, and do not undistort the image a second time yourself.

**Expected result:** a reconstruction and usable reference image. **Ready to align** means local geometry is available, not that the camera is correctly positioned on the building plan. **Provisional geometry** means there is unresolved uncertainty, not necessarily that processing failed.

A local 3D shape can be distorted, incomplete or arbitrarily scaled. Correct-looking furniture is not a substitute for floor-point verification. If the camera result is tilted or stretched, proceed to floor-region refinement before global fitting.

<!-- pagebreak -->

## 5. Refine the Floor and Choose a Leveling Method

This step tells the reconstruction which visible surface is actually the floor. It is **not** a people-detection region, alarm zone or area to ignore during tracking.

### Select a real floor patch

1. Complete the initial calibration first.
2. Select the camera and open **Reference**.
3. Unlock the camera if necessary.
4. Click **Select floor region**.
5. Click at least three vertices around a visible, flat patch of floor. The tool draws a filled polygon.
6. Avoid desks, chair seats, people's clothing, walls, glass reflections and stairs.
7. Use **Remove last point** to remove the most recently selected polygon vertex if needed.
8. Click **Save floor region**.
9. Click **Calibrate camera** again and wait for completion.
10. Inspect **Calibration review** again before fitting the camera to the plan.

Use one reasonable polygon around a clear patch. Do not drag a rectangle or double-click expecting a different drawing tool; the current UI uses a sequence of vertex clicks. Clicking **Select floor region** again toggles selection mode; beginning a new selection starts a new polygon.

### Leveling choices

| Choice | When to use it |
|---|---|
| **Gravity + floor check** | Default starting point. Estimates gravity and checks selected floor evidence when available. Conflicting geometry may remain provisional. |
| **Gravity** | Inspect this alternative when the floor patch is weak but the gravity-based result looks more plausible. It is still an estimate. |
| **Selected floor plane** | Advanced alternative when a trustworthy floor patch is available and the plane estimate is more plausible than gravity. A bad patch can produce a bad result. |

Do not cycle modes until furniture merely looks attractive and call that validation. The final choice must also project independent floor landmarks correctly.

**Important:** a newly computed calibration replaces that camera's local layer and mapping; expect to repeat global alignment and point verification. Finish the reference/level/floor-refinement decisions **before** spending time on the final alignment. An unchanged calibration request may be reused from cache instead of recomputed.

**Lock is not a backup.** Reference replacement is blocked while locked. Still avoid recalibration on a finished camera unless you intend to rebuild its mapping; preserve a backup of useful work.

<!-- pagebreak -->

## 6. Set the Floor Plan's Real-World Scale

Scale tells the tool how many drawing pixels correspond to one real metre. It is shared by all cameras in the project.

### Recommended method: measure a known dimension

1. Add at least one camera and select it. It must be **unlocked** because this version saves scale through the selected camera's editing controls.
2. Identify two exact endpoints of a trustworthy dimension on the plan.
3. Enter that real distance in **Known distance (metres)**.
4. Click **Measure on plan**.
5. Click the first endpoint on the centre floor plan.
6. Click the second endpoint on the floor plan.
7. The tool calculates and saves **Pixels / metre** after the second click. Confirm the bottom status says the project was saved.
8. Check another known dimension in a different part of the drawing. A uniformly scaled drawing should give a consistent result.

### Example

The plan labels a span as `3000 mm`, which is **3 metres**. Enter `3`, not `3000`.

If the two endpoints are 240 pixels apart, the scale is:

```text
Pixels per metre = 240 / 3 = 80
```

At 80 pixels/metre, an error of 16 plan pixels corresponds to 0.20 metres. Use this only when the scale and correspondence are trustworthy.

### Manual entry

If you already know the correct pixels/metre for this exact uploaded drawing, enter it in **Pixels / metre** and click **Save scale**. Merely typing a number does not save it.

### Do not confuse these controls

| Control | What it changes |
|---|---|
| **Known distance / Pixels per metre** | The whole plan's relation to physical metres. |
| **Scale multiplier** | The selected camera layer's size on that plan. |

Changing the floor scale invalidates approvals for **all cameras**. Set it early, before camera approval. Recheck geometry after changing it.

If the drawing is not reliably dimensioned, you may continue as a **preview**, but do not claim that its distances or positions are accurate in metres. Do not enter an arbitrary scale simply to remove an approval warning.

<!-- pagebreak -->

## 7. Position the Layer by Dragging, Rotating and Scaling

Think of this as placing a puzzle piece on the plan. It provides initial alignment; landmark checks establish whether the alignment is actually correct.

### Choose what is visible

| Layer choice | Meaning and useful application |
|---|---|
| **Floor points** | Top-down points near the estimated floor. Usually the clearest starting layer. It can be sparse. |
| **All 3D points** | Includes walls, furniture and other reconstructed surfaces. Useful context, but do not mistake furniture surfaces for floor contacts. |
| **Image ground projection** | Projects image colour onto the assumed floor. Floor areas can look smooth; elevated objects may stretch dramatically. |
| **Hidden** | Hides the overlays so you can see drawing landmarks clearly. It does not remove calibration. |

**Opacity** changes visibility, not geometry or inference. It applies to the displayed overlays; unselected cameras are shown much fainter. Select the camera you want to inspect before judging its brightness.

### Move the selected camera layer

1. Select the camera in the left list and click **Unlock camera** if needed.
2. Choose a visible layer and adjust **Opacity**.
3. Click **Drag layer** to turn drag mode on.
4. Drag on the centre plan. The selected layer's translation is applied and saved after you release the pointer; continuous animated movement is not required.
5. Click **Drag layer** again to turn it off.

**Important:** while drag mode is on, clicking the plan moves the layer instead of completing a point pair. Look at the mode text above the plan: it should say **Camera alignment** or **Independent holdout check** when you are pairing points, not **Drag selected layer**.

### Rotate and resize

1. Enter a **Rotation delta** in degrees and a **Scale multiplier**.
2. Click **Apply adjustment**.
3. Use a small correction, inspect it, then adjust again if needed.

Examples: rotation `10` is clockwise on the displayed plan; `-10` is counterclockwise. Multiplier `1.10` enlarges the layer by 10%; `0.90` reduces it by 10%; `1` preserves its current size.

These are **relative changes**, not absolute camera yaw or final scale. Pressing Apply twice repeats the change. Reset the fields to rotation `0` and scale `1` when finished. The adjustment uses the centre of the plan as its pivot, so rotation/scaling can also move the layer; drag again if necessary.

Manual transforms change the actual saved mapping, not just a decorative image. They invalidate approval and may change the point residuals. There is no general transform-undo button in this version. **Remove last point** does not undo a drag or rotation.

<!-- pagebreak -->

## 8. Fit Camera-to-Plan Point Correspondences

A correspondence is one physical floor landmark clicked in **both** the camera reference image and the floor plan.

### Add a fit pair

1. Select the camera and unlock it.
2. Open **Reference**. Exit floor-region selection if it is active.
3. Turn **Drag layer** off. Finish any active plan measurement first.
4. Select **Fit point**. Choose **Hidden** in the layer dropdown if overlays obscure the drawing.
5. Click a floor landmark in the right-hand reference image **first**.
6. Click the same physical landmark on the centre floor plan **second**.
7. A numbered pair appears in both views and in the point list.
8. Repeat, distributing points across the useful floor area instead of clustering them in one corner.

If the first image click was wrong but you have not clicked the plan yet, click the correct image location again; it replaces the pending image point. **Remove last point** removes the last completed pair, not just an unfinished first click.

Good choices include floor/wall junctions, floor tile corners, column bases and door thresholds. A desk corner, a person's shoulder or the top of a chair is not a floor point. Do not pair a furniture symbol on the plan with the elevated top of that object in the image.

### Choose one fitting method

| Method | What it solves | Minimum and recommendation |
|---|---|---|
| **Fit floor mapping** | Directly maps the rectified camera image's floor plane to the drawing. It does not require the reconstructed furniture shape to be perfect. | At least 4 well-distributed fit pairs; prefer 6-10 when possible. Avoid points all on one line. |
| **Fit local layer** | Adjusts the existing reconstructed layer by translation, rotation and uniform scale to match the pairs. | At least 2 separated fit pairs and an available local layer; prefer 3 or more. |

**Recommended decision:** use **Fit floor mapping** when clear floor correspondences are available, particularly if the reconstruction is distorted. Use **Fit local layer** when its top-view shape is useful and needs mainly placement/rotation/scale. Do not press both as if they were two required processing stages: the later fit replaces the current mapping.

### Save and inspect

1. Click the chosen fit button. It saves the current pairs and fitted mapping.
2. Inspect the overlay and the point residuals in pixels.
3. Read **Inliers**, for example `6/7`. An inlier is a pair accepted by the robust fit; it is not a percentage of tracking accuracy.
4. To fix the latest mistaken pair, use **Remove last point**, add it again, and refit. There is no per-row delete/move control; removing an older pair requires removing the newer pairs after it too.
5. Click **Save** before leaving, particularly after adding/removing pairs without fitting.

New pairs are initially held in the browser. **Save** or a successful fit commits them. Camera/tab changes attempt to save dirty points, but project selection is not a substitute for saving. With exactly four fit points, almost zero residual can simply mean the transform fits those four clicks exactly. Independent checks are still required.

<!-- pagebreak -->

## 9. Verify With Independent Holdout Points

This is the step that answers: **Does a floor point I click in the camera project to the correct place on the drawing?**

### Collect checks not used for fitting

1. Finish the fit before adding check points.
2. Select **Holdout point**. Turn drag mode off and leave floor-region selection closed.
3. Choose a new floor landmark that was **not** one of the fit points.
4. Click its true location in the reference image.
5. Click its independently known location on the plan. Do not place the plan click where the current projection suggests merely to obtain zero error.
6. Add at least two well-separated holdout pairs. Use more across the walking area when possible, especially near entrances and camera handoffs.
7. Click **Save** before judging the verification status; the summary is based on saved server-side data.

Fit points are shown in green/teal; holdouts are amber. The line from the clicked plan point toward the calculated projection shows the discrepancy. The point list reports its distance in **plan pixels**. This discrepancy should be small because the physical landmark matches, not because you adjusted the truth to match the model.

### Default approval checks

| Requirement | Meaning |
|---|---|
| Valid mapping | No missing or singular camera-to-floor transform. |
| Saved positive scale | The operator has supplied the correct drawing scale. The software cannot independently survey the building. |
| At least 2 separated holdouts | Checks must not duplicate fit points or be almost coincident. |
| Holdout median error <= 0.25 m | Typical check-point error stays inside the configured limit. |
| Holdout maximum error <= 0.50 m | No saved check point exceeds this limit. |

**Example:** at 80 pixels/metre, errors of 12 and 20 pixels are 0.15 and 0.25 m. Their median is 0.20 m and maximum is 0.25 m. These meet the numerical limits, assuming independent, correctly placed points and a correct scale.

### If checks fail

Inspect mistaken clicks, floor-vs-elevated landmarks, drawing accuracy, floor scale and undistortion. Correct the cause and refit. If you convert a failing holdout into a fit point, it is no longer independent: reserve new checks before claiming validation.

Do not delete difficult valid check points, loosen the scale or suppress walking areas just to make approval pass. More smoothing will not fix a wrong geometric projection.

**Holdout checks passed** is not yet **Camera approved**. Approval is a separate action in the next step. Passing two points does not establish uniform accuracy over the entire camera view or prove a surveyed 3D camera pose.

<!-- pagebreak -->

## 10. Approve, Lock, Add the Next Camera and Finish

### Complete one camera

1. Save its final fit and holdout points.
2. Check the residuals and physical plausibility across the walking area.
3. If verification is satisfactory, click **Approve camera**.
4. Confirm **Approved** appears for that camera.
5. Click **Lock camera** to protect ordinary alignment edits. Locking does not itself approve anything.

**Save and approval are different.** A normal Save or another calibration edit returns the camera to draft and can clear its approval. Use **Approve camera** after the last edit, then **Lock camera**. Do not press Save again just as a final ritual after approval. Changing the project scale requires reapproval of all cameras.

### Add the next camera

Click **Add**, choose the next physical camera's video, and repeat Steps 3-9 for that camera. Each camera must align to the **same uploaded plan and its coordinate system**, not to a separately rotated copy of the plan.

For every camera, record its ID, video filename, chosen reference time, fit method, number of fit/check pairs and any unresolved warnings. Low visual overlap between cameras does not remove the need for time synchronization and reliable cross-view identity evidence.

### Choose the correct way to finish

| Action | What happens | When to use it |
|---|---|---|
| **Finish calibration** | Checks every registered camera's approval, saves the finished project and opens the runner. | All camera checks and approvals are valid. |
| **Save preview** | Saves an explicitly provisional project and opens the runner. Every camera still needs a mapping. | Investigating the visual result while metric checks remain pending. |

**Save preview does not turn a failed calibration into an approved one.** Preview output is marked **DRAFT CALIBRATION**. Use that route honestly for review, as in the recorded office example.

### Close and reopen

1. Finish calibration or save the preview before closing the browser.
2. Keep the server running if a job is active.
3. Reopen the tool and select the project from the top dropdown.
4. Select **Calibration** to inspect/edit, or **MCT runner** to process videos.
5. Recheck the runner settings before each run. This version does not reliably restore every previously entered run field after a reload.

To remove a camera, select it, **Unlock camera**, then click **Remove camera** and confirm its name/project. Its source files, calibration and past runs are retained; new runs exclude it. Removal is blocked during an active job. Finish/save preview again afterward. **Hide camera** remains display-only, not removal. There is no automatic Restore button.

<!-- pagebreak -->

## 11. Synchronize Videos and Configure the MCT Runner

### What the fields mean

| Field | Meaning | First-test example |
|---|---|---:|
| **Start (seconds)** | Start on the common configured playback timeline. Not the time used to extract calibration references. | 0 |
| **Duration (seconds)** | How much common time to process. All videos must cover it. | 30, then 120 |
| **Frames / second** | Requested processing sample rate and resulting replay cadence. Not measured hardware speed. | 10 |
| **Inference width** | Resizes each input frame proportionally for the pipeline. Larger input costs more work and cannot create missing detail. | 1280 |
| **Detection detail** | Internal detector input setting. **Balanced (416)** is the current default; **Detailed (640)** is a heavier alternative. | Balanced (416) |
| **[Camera name] offset (s)** | Seconds to skip in that camera's file relative to the configured common timeline. | Depends on that recording |

The source position used for each camera is:

```text
Source video time = Start + that camera's offset + elapsed output time
```

### Worked synchronization example

Camera A's file starts at 10:00:00. Camera B's starts at 10:00:05. Both should show the period beginning at 10:00:10.

Choose the common origin as 10:00:05. Set **Start = 5**, **A offset = 5**, and **B offset = 0**. The tool starts A at source second 10 and B at source second 5: both represent 10:00:10.

Offsets are nonnegative in this UI. Choose a common origin that makes all offsets nonnegative; do not reverse the offset sign based on intuition. Ensure `Start + offset + Duration` fits every clip.

### Check synchronization

1. Enter Start and each camera's offset.
2. Click **Preview synchronization**.
3. Inspect the preview videos at the selected source positions.
4. Check a shared visible event where possible, and inspect another later event to detect clock drift.
5. Adjust offsets and click Preview again if needed.

The preview creates **separate video players positioned at the selected start**. It does not calculate offsets, guarantee synchronized playback controls, or automatically verify camera clocks. Burned-in clocks can differ from capture time; the office footage showed approximately one-second differences at some moments.

Higher resolution or Detailed detection does not automatically improve global IDs. In the review experiment, Detailed found more small people but fragmented more identities. Evaluate both detection and identity continuity when changing settings.

<!-- pagebreak -->

## 12. Run Processing and Retrieve the Output

### Start a run

1. Ensure the project was finished or saved as preview after its latest calibration changes.
2. Open **MCT runner** and recheck every field, especially offsets and Duration.
3. Start with a short segment if this is a new environment.
4. Click **Run MCT and render** once.
5. Wait for the state to become **complete**. Keep the workstation and server running.

The processing pipeline reads the configured videos, detects people, tracks each camera independently, extracts appearance information for cross-camera ReID, estimates pose/ground contacts, projects reliable contacts onto the plan, fuses observations and renders smoothed trails.

| Phase | What to expect |
|---|---|
| Queued / loading | GPU worker and models are starting. The display may not immediately show a useful fraction. |
| Fresh tracking and pose | Frames from the selected videos are being processed. |
| Rendering floor map | Tracking observations are projected and the output video is encoded. A completed inference phase is not yet a completed video. |
| Complete | Output player and download links are available. |
| Failed / interrupted | Open **Run log**. Investigate before retrying; do not treat a partial file as a successful result. |

### Download the three outputs

| Link | What you receive |
|---|---|
| **Download video** | The MP4 showing camera views and the shared floor map/trails. |
| **Tracking events** | JSONL: one JSON object per output record. Includes tracking/projection observations and identity-correction information for technical review. |
| **Inference metrics** | JSON timings, frame counts and GPU-memory information for that run. |

Browser downloads go to the computer running that browser. Through TeamViewer, clicking Download in the remote browser downloads on the remote workstation, not automatically onto your Mac. Use the approved transfer method when sharing permitted files.

### Cached results and older videos

An identical completed request can return **Completed run reused**. That is expected cache behavior, not evidence of new inference. Changed inputs/settings can invalidate affected stages; a render retry may reuse already verified neural observations.

An older output can remain visible while a new job is queued/running or after a failed attempt. Confirm the latest run says **complete**, its settings are correct and its new output is the one you intend to share. After calibration changes, finish/save preview again and rerun; the old MP4 does not update itself.

The current UI has no dedicated job-cancel button. Do not close terminals, restart the GPU stack or stop another team's demo just to interrupt processing. A 10 FPS output file does not establish 10 FPS real-time throughput on every camera.

<!-- pagebreak -->

## 13. Read the Tracking Video and Diagnose Problems

The illustrated example is supplied in the separate private test-material bundle.
### What the display means

| Display element | Meaning |
|---|---|
| **L** ID | Local track ID within one camera. The same person can have different local IDs in different cameras. |
| **G** ID | Current global identity hypothesis shared across cameras. It is not a person's verified name. |
| Floor marker | A person position the pipeline currently considers reliable enough to publish. |
| Trail | Recent estimated movement. A break can reflect missing evidence or an implausible jump rather than a continuous observed path. |
| Draft calibration label | The site is running as a preview, not independently approved metric calibration. |

Global IDs may initially be separate and later reconciled when evidence accumulates. The platform must apply those correction events consistently. Two different G IDs for the same person are still a problem to investigate; a smooth trail does not prove a correct identity.

### Review the whole result, not just a clear frame

1. Follow one person through an entrance, a camera overlap and an exit.
2. Check that empty chairs do not create published people on the floor.
3. Look at small, seated and occluded people as well as clear full-body detections.
4. Check whether visible foot contacts land on plausible walking/seating locations.
5. Inspect the last part of the video for late ID fragmentation and time drift.
6. Distinguish a missing detection from a detection with an uncertain, withheld floor position.

**Geometry problem:** the same image landmark projects to the wrong place repeatedly. Revisit distortion, fit points, scale and the drawing. **Tracking problem:** the floor mapping is plausible but identity changes between views or over time. Check synchronization, detection coverage and ReID evidence. Do not rotate the whole camera layer merely to conceal a wrong ID.

The recorded office result still contains difficult cases. Use it to understand the workflow, not as a guarantee that every person or handoff is correct.

<!-- pagebreak -->

## 14. Button Reference: Navigation and Left Panel

| Control | What it does / important distinction |
|---|---|
| **Project dropdown** | Opens a saved site. Save pending point edits before changing projects. |
| **New project / Create project** | Opens the form for a new drawing and independent camera collection. |
| **Create site / Cancel** | Creates the project from the form, or closes the form without creating it. |
| **Calibration** | Opens floor alignment and camera-preparation tools. |
| **MCT runner** | Opens video processing controls; does not by itself approve or finish calibration. |
| **Verified GPU engine** | Installation status, not calibration or tracking-accuracy approval. |
| **Add** | Uploads and registers another physical camera's video. |
| **Camera list item** | Selects which camera the right panel and layer adjustments affect. |
| **Known distance (metres)** | Real length of the span you will click on the plan. |
| **Measure on plan** | Starts two-click measurement on the floor plan; the second click calculates and saves scale. Select an unlocked camera first. |
| **Pixels / metre** | Numeric drawing scale; manual typing needs **Save scale**. |
| **Save scale** | Commits manual scale and invalidates previous camera approvals. |
| **Floor points** | Shows points near the estimated floor. |
| **All 3D points** | Shows the full reconstructed top-view layer, including elevated structures. |
| **Image ground projection** | Shows image colour warped onto the assumed floor; furniture stretching is possible. |
| **Hidden** | Hides all displayed calibration overlays through the layer selector. Does not disable cameras. |
| **Opacity** | Changes overlay transparency, not mapping quality. Other cameras remain fainter. |
| **Rotation delta** | Relative on-screen rotation in degrees for the next Apply action. |
| **Scale multiplier** | Relative selected-layer size for the next Apply action; not the floor's metre scale. |
| **Apply adjustment** | Applies/saves the entered relative rotation and scale to the selected mapping. Repeated clicks repeat the adjustment. |
| **Drag layer** | Toggles plan dragging on/off. Turn off before pairing image/plan points. |
| **Lock / Unlock camera** | Prevents/allows ordinary point and transform edits. Not an approval or a backup of calibration jobs. |
| **Hide / Show camera** | Hides/shows just the selected calibration overlay. Does not remove it from MCT. |
| **Remove camera** | Confirms removal from the active project; keeps files/history for recovery. Unlock first. Finish/save preview afterward. |
| **Finish calibration** | Requires all registered camera approvals and saves the finished site. |
| **Save preview** | Allows a provisional run with mappings present; keeps metric acceptance pending. |

**Three common confusions:** Hidden is a display option, Lock is editing protection, and Approved is a calibration-validation state. They are not interchangeable.

<!-- pagebreak -->

## 15. Button Reference: Camera Inspector and Runner

| Control | What it does / important distinction |
|---|---|
| **Reference** | Displays the image where floor regions and point correspondences are clicked. Rectified after calibration. |
| **Video** | Opens source playback for selecting a calibration frame. |
| **Reference time (seconds)** | Time within that source file, not a camera synchronization offset. |
| **Use video position** | Copies player time into the reference-time field only. |
| **Extract reference frame** | Creates a new reference and clears dependent camera mapping/points. |
| **Calibration review** | Opens the calibration visualization. Not the final MCT result. |
| **Undistortion** | Displays the lens-correction review image. |
| **3D top view** | Displays the camera-relative reconstructed top-view review. |
| **Calibrate camera** | Runs/reuses the isolated calibration job. A fresh result requires renewed alignment and verification. |
| **Leveling selector** | Chooses Gravity + floor check, Gravity, or Selected floor plane for the next calibration job. |
| **Select / Save floor region** | Starts polygon selection / saves the polygon. Saving alone does not rerun calibration. |
| **Fit point** | Makes the next image-then-plan pair an input to fitting. |
| **Holdout point** | Makes the next pair an independent check, excluded from fitting. |
| **Remove last point** | Removes the last pair, or last polygon vertex during floor selection. Not general undo. |
| **Fit floor mapping** | Fits a direct floor transform from at least four fit pairs and saves it. |
| **Fit local layer** | Fits the existing local layer's similarity transform from at least two fit pairs. An alternative to direct mapping. |
| **Save** | Saves point edits; does not approve calibration and may clear previous approval. |
| **Approve camera** | Saves/checks the current camera and marks it approved only when the configured checks pass. |
| **Camera parameters** | Expands estimated distortion, local geometry and available pose parameters for technical inspection. |
| **Processing log** | Expands calibration diagnostics. The job panel shows the latest calibration job, not necessarily the currently selected camera's job. |
| **Preview synchronization** | Creates separate source players at Start + offset. Does not synchronize clocks automatically. |
| **Run MCT and render** | Queues processing/rendering or reuses an identical completed job. |
| **Run log** | Expands processing errors and progress details. |
| **Download video / Tracking events / Inference metrics** | Downloads MP4 / JSONL / JSON for the completed result. |

The runner's **Start**, **Duration**, **Frames / second**, **Inference width**, **Detection detail**, and per-camera **offset** fields are explained with examples in Step 11. Recheck them after reopening the page.

<!-- pagebreak -->

## 16. Follow the Recorded Example

Companion recording: `walkthrough/operator_workflow_original.mp4`, approximately **9 minutes 54 seconds**, normal speed. The table below refers to this original recording. The narrated edition is approximately **11 minutes 50 seconds**, with retimed actions and its own chapters/script.

| Time | What is happening | Manual step |
|---|---|---|
| 00:00 | Create the project and upload the drawing. | 2 |
| 00:01 | Add Reception, choose its frame and run initial calibration. | 3-4 |
| 01:22 | Select Reception's visible floor and refine calibration. | 5 |
| 02:11 | Add Reception fit pairs, fit its mapping, save and lock. | 8, 10 |
| 02:19 | Add Workspace 1 and run initial calibration. | 3-4 |
| 03:08 | Refine Workspace 1's floor geometry. | 5 |
| 04:05 | Fit Workspace 1 to the plan and save. | 8, 10 |
| 04:16 | Add Workspace 2 and run calibration. | 3-4 |
| 05:00 | Refine Workspace 2's floor geometry. | 5 |
| 05:45 | Fit Workspace 2's local layer and save. | 8, 10 |
| 05:52 | Strict approval fails; the example explicitly uses Save preview. | 9-10 |
| 05:54 | Reopen the project and inspect synchronization previews. | 11 |
| 05:57 | Run fresh inference and rendering; wait for completion. | 12 |
| 09:50 | Inspect the generated result. | 13 |

### What the recording deliberately does not establish

The example did not establish independent metric scale and holdout checks. **Steps 6 and 9 remain required for an approved site.** The recordings predate the latest removal and reference-player fixes, but the workflow remains the same. Neither video proves tracking accuracy.

### Optional office practice inputs

This test-material bundle contains three raw videos and the exact `inputs/floor.png`, without saved calibration or tracking decisions. Reception uses reference time 60s; both workspaces use 0s. The runner uses Start 0, Duration 120, FPS 10, width 1280, Balanced 416 and zero offsets for this exploratory fixture. Follow `OPERATOR_POINT_GUIDE.md` for the same approximate pairs.

Those frame choices, landmarks and offsets are **office-specific operator inputs**. Do not copy them into a different site. Even for the office, exact synchronization remains a check to perform rather than an assumption to inherit.

Local companion files:

- `walkthrough/operator_workflow_original.mp4`: original operator recording.
- `walkthrough/operator_workflow_narrated.mp4`: narration, captions and new chapter timings.
- `expected/reference_120s.mp4`: the prior visual reference; not ground truth.
- `expected/regenerated_120s_v0.3.1.mp4`: this version's fresh verification output.
- `VALIDATION_NOTES.md`: checks, limitations and pending gates.

The manual's screenshots are examples from this recorded review, not a live assertion about your currently edited calibration.

<!-- pagebreak -->

## 17. Troubleshooting, Backup and Handoff Checklist

### When something does not work

| Symptom | Check first |
|---|---|
| Points cannot be clicked | Select/unlock the correct camera; open Reference; turn drag mode off; exit floor-region selection; finish any plan measurement. |
| Point clicks draw a polygon | Floor-region selection is still active. Save it or toggle Select floor region off. |
| Plan clicks move the layer | Drag mode is on. Toggle Drag layer off. |
| Stuck in Measure known dimension | To cancel an unfinished measurement without changing scale, click Drag layer twice, without dragging the map. End with drag mode off. |
| Scale does not save | Add/select an unlocked camera; use two endpoints or click Save scale after manual entry. Check the status at the bottom. |
| Verification count changes but status seems old | Click Save. The summary comes from saved data. |
| Calibration looks too faint | Select that camera; increase Opacity; compare Floor points and All 3D points. Unselected layers are intentionally faint. |
| Geometry is stretched or tilted | Inspect undistortion, select real floor, reconsider leveling and use valid floor pairs. Elevated objects can stretch in image ground projection. |
| Fit is rejected | Check enough well-spread pairs, correct click order and floor landmarks; avoid collinear pairs. |
| Approval is rejected | Check scale, independent holdouts, separation and residual limits. A processing-complete message is not approval. |
| Fit disappeared after recalibration | A fresh calibration replaces the mapping. Refit and reverify; do refinement before final alignment. |
| Project changed / reload error | Another edit changed its revision. Use one operator at a time. Save only against the current version; do not overwrite someone else's work. |
| MCT cannot start | Finish/save preview after the latest edits; ensure every clip covers Start + offset + Duration. |
| Wrong people merge / one person has two IDs | Check real synchronization and geometry, then detection/ReID evidence. Do not force numeric IDs for the demonstration. |
| Old result is still displayed | Confirm the latest job completed successfully; then select/download its output. |

### Save the work, not only the video

Project data live under the installation's `workspaces/<project-id>/` directory. Back up the **whole project directory** through your approved local process after jobs finish: media, site configuration, calibration assets and job records belong together. Copying just `site.json` or the MP4 is not a reusable project backup. There is no complete-project Export/Import button in this UI version.

Before handing over a new-site demonstration, confirm:

- Correct floor drawing, physical cameras and useful calibration frames.
- Trustworthy scale, independent holdouts and the correct approved/preview label.
- Shared-event synchronization, full source-interval coverage and clear recorded settings.
- Complete video, events and metrics; no failed job presented as success.
- Review of overlap duplicates, late handoffs, chairs, seated people and foot projection.
- The recipient knows which checks remain pending and how to reproduce the run.

Share only approved material through the approved channel. Do not put client videos, identities, embeddings, camera credentials, workspaces or model binaries in the source repository. Coordinate model/deployment permissions separately with the integrator. The current package's test report remains the authority for what has and has not been accepted.
