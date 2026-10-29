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
