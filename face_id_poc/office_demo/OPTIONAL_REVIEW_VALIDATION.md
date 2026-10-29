# Optional review validation - 29 October 2026

## Scope

Controlled, consented office evaluation only. Both features are off by default.
This is not a general CCTV watchlist, a liveness system or an automated security
decision. No new models or model-rights assumptions were introduced.

## Automated checks

- Mac Python 3.11 / CPU, supplied SCRFD-2.5G + ArcFace + YOLOX models:
  **158 Python tests passed**, including real-model execution.
- **27 JavaScript tests passed**. Reset now restores all 19 settings, including
  disabled optional collection/notices and cleared consent confirmation.
- Tests cover strict storage quality, opt-in, five best crops, duplicate
  rejection, immediate shorter retention, record/storage limits, expired-photo
  access, manual enrollment, flag consent/expiry, notice cooldown, stale jobs,
  nonblocking queue submission and legacy tracking/enrollment behavior.
- Existing trained model binaries were not changed or committed.

## Ubuntu/NVIDIA checks

An isolated test instance on the local Ubuntu TITAN X (Pascal), using the existing
CUDA 12.2 / ONNX Runtime 1.18.0 runtime, verified model checksums and profiled actual
CUDA nodes for ArcFace, YOLOX and SCRFD. Existing services were left running.

The three authorized synchronized office RTSP replays were started at a reduced
3 analysis FPS per camera for a functional smoke check. This was deliberately
not a throughput test: the shared GPU was already busy and near 83-84 C. The
existing 85 C thermal guard remained enabled. No 10-FPS capacity claim is made.

For deterministic positive UI cases, an explicitly labelled two-person
consented-photo RTSP replay was used. One participant was initially withheld
from the **isolated copy** of the gallery. The walkthrough shows:

1. Opt-in to unknown collection and a saved unknown track.
2. Manual enrollment through the normal consent and quality preview, then both
   test participants being recognized.
3. Explicit review flag with a reason and expiry.
4. A fresh-evidence review notice with reference and observed photos, followed
   by operator acknowledgement.

The static-photo replay is **not** face-recognition accuracy, anti-spoofing or
real-camera generalization evidence. Similarity values in that replay must not
be used as calibrated production thresholds.

## Persistence and isolation

- Settings, enrolled people, review records and flags survived app restart;
  RTSP capture recovered.
- Unknown images remain outside the Qdrant reference collection until explicitly
  enrolled. The separate review store contains no face embeddings.
- A tightly cropped saved face initially failed re-detection at enrollment. The
  final implementation retains additional native surrounding pixels for unknown
  crops; normal enrollment checks remain unchanged. The corrected path passed
  the live preview/commit test.
- Removing a registered reference also removes review notices that depend on
  that reference, avoiding a stale comparison image.
- The original local gallery database was unchanged (20 people / 23 references).
  The client's server was not accessed or altered.
- Desktop and narrow-screen UI checks completed. The walkthrough is silent,
  browser-only, about 56 seconds; it omits pauses between captured chapters.

## Remaining validation

Real office false-match/false-notice rates, prolonged storage/retention testing,
production authentication/encryption and consequential-decision suitability are
not established. Keep the functionality limited to the agreed consented test.

## Follow-up: real office-video workflow

The three existing 2560x1440 office recordings (09:16-09:21) were replayed
together through RTSP on the local Ubuntu GPU host. These are recorded video,
not a connection to the live office cameras. The original gallery and clips
were preserved; the client's server was not accessed.

- The unmodified 20-person / 23-reference gallery completed a 310-second
  full-frame smoke run at approximately 3 analysis FPS per camera. There were
  364 face detections and 68 recognition queries. Counts are repeated frame
  observations, not unique people or accuracy scores. No strict review notice
  passed in this baseline.
- To exercise unknown collection, one consenting participant was deliberately withheld from an
  isolated gallery. The existing head-detail option was enabled, without a
  new model or lower storage checks. One track accumulated five retained
  native video crops, 86-88 pixels wide, quality scores 75.3-78.3.
- This exposed a real enrollment defect: small saved crops could yield zero
  detections after the ordinary detector resize. Enrollment now retries
  zero-face small uploads using two bounded black-padding sizes. Coordinates,
  pixel measurements, detector confidence, single-face checks and reference
  quality checks are unchanged. It does not retry a detected multi-face upload.
- After the correction, four of those five actual video crops passed normal
  enrollment; one remained rejected. On the continued replay, recognition
  and a review notice succeeded (similarity 0.846, margin 0.672, three fresh
  observations). The operator acknowledged that notice through the UI.
- This is **same-recording enrollment and replay matching**, so it establishes
  the functional workflow only. It does not establish independent recognition
  accuracy, liveness or deployment false-alert rates.
- **162 Python tests passed**, including supplied-model CPU execution, and
  **27 JavaScript tests passed**. The added regression checks cover native-size
  coordinates, padding boundaries, invalid landmarks and large-image behavior.
  The deployed GPU source hash matches the reviewed source.

The replacement walkthrough is 58 seconds, silent, and shows actual office-video
inputs, stored video crops, manual enrollment and the fresh-observation notice.
The earlier still-photo walkthrough is retained separately, not substituted for
this video test. Capture was limited to 3 analysis FPS under shared-host load;
this follow-up is not a camera-capacity benchmark.
