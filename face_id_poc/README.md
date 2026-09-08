# CAMEX Face ID and Head Tracking

Version 0.1.1. Single-camera review PoC: frames in, head tracks and proposed employee
identities out. The OpenCV MP4 renderer is optional. No floor calibration, MCT
package, database, web UI or cloud inference service is required.

## Networks and Scope

**SCRFD-2.5G** detects faces and five landmarks. **ArcFace ResNet50/WebFace600K**
produces 512-D face embeddings matched to enrolled photos. A **custom YOLOX-Tiny
CrowdHuman head/body detector** and ByteTrack-based tracker maintain local tracks.

Read [business limitations and model-use notes](docs/MODELS_AND_LIMITATIONS.md),
[model provenance](MODEL_PROVENANCE.md) and [validation status](docs/VALIDATION.md)
before presenting deployment claims. Unknown is not proof that someone is an outsider.

## 1. Get the Code and Test Kit

For a new checkout (GitHub access to the private repository is required):

```bash
git clone --branch codex/camex-face-id-v0.1.1 https://github.com/ashutoshbk/face_recognition.git
cd face_recognition/face_id_poc
```

For an existing checkout, preserve your work, fetch the new branch, switch to it,
and enter `face_id_poc/`. The branch can be tested before its PR is merged.
Extract the separately supplied `CAMEX_Face_ID_Test_Kit_0.1.1.zip` outside the repository:

```text
CAMEX_Face_ID_Test_Kit_0.1.1/
  START_HERE.md
  demo.json
  SOURCE_VERSION.json
  FILE_MANIFEST.json
  inputs/reception_5min.mp4
  inputs/reception_validation_60s.mp4
  inputs/staff/                       # 20 original reference images
  models/                            # three ONNX files
  reference_outputs/demo.mp4         # latest chair-fixed result
  reference_outputs/metrics.json
  reference_outputs/events.jsonl
```

The kit's `demo.json` defines its sample paths, camera label and clock. The
launcher does not hard-code a site or expect a folder named `package`.

## 2. Install an Isolated Backend

Recommended for the NVIDIA Ubuntu server:

```bash
bash install.sh --backend docker
python3 scripts/doctor.py --backend docker --models /path/to/CAMEX_Face_ID_Test_Kit_0.1.1/models
```

Prerequisites: Docker, Docker GPU support/NVIDIA Container Toolkit, a compatible
NVIDIA driver and internet for the initial build. Setup does not install host
drivers, change host Python, stop other demos or use sudo. Inference containers
run with networking disabled.

For a CPU-only machine, including a Mac:

```bash
bash install.sh --backend cpu --python python3.11
python3 scripts/doctor.py --backend cpu --models /path/to/CAMEX_Face_ID_Test_Kit_0.1.1/models
```

CPU prerequisites: Python 3.10/3.11, FFmpeg with libx264, and a C/C++ compiler for
cython-bbox. Setup creates `.venv-cpu/`; it does not install globally. CPU speed
is not a real-time guarantee. macOS MPS is not used.

`--backend cuda` creates a separate `.venv-cuda/` for native Linux use, requiring
compatible CUDA 11.8/cuDNN 8 libraries already available to ONNX Runtime. Docker
is easier to reproduce. Do not install CPU and GPU ONNX Runtime wheels together.
The Docker image is Linux x86-64, not a Jetson/ARM image. Windows users should use
configured Ubuntu/WSL2 with Docker GPU support; native Windows is not supplied or tested.

## 3. Reproduce the Reception Demo

From `face_id_poc/`:

```bash
python3 scripts/run_demo.py --backend docker \
  --kit /path/to/CAMEX_Face_ID_Test_Kit_0.1.1 \
  --output /path/to/face_id_runs/reception
```

Use `--backend cpu` for the CPU environment. The launcher verifies the kit, enrolls
the photos, and runs fresh tracking/recognition. Only the specified input, model,
config and output paths are mounted, not the whole machine. Keep output outside
the kit. Paths with spaces work; commas/newlines in Docker mount paths are rejected.

For a smoke test, add `--seconds 10` and choose a different output directory.
The supplied clip is 300 seconds, 2560 x 1440 at 10 FPS; output is a 1080p MP4 at
normal playback speed. The displayed camera clock is derived from the configured
start time, not OCR/NTP. Evidence crops are timestamped last-good observations.

Outputs under your selected directory:

- `gallery/`: vectors, reference crops, source hashes and rejected-photo details.
- `demo/demo.mp4`: annotated video with captured/reference face pairs.
- `demo/tracks.jsonl`: observed tracks and identity states per processed frame.
- `demo/events.jsonl`: identity confirmation and revocation events.
- `demo/evidence/`: confirmation images and metadata.
- `demo/metrics.json`: providers, hashes, frame count and performance.

Completed matching runs are reused. Changed settings, photos or videos require
a fresh output directory. Interrupted inference restarts to reconstruct track
state; completed input preparation/enrollment are reused. Concurrent writers to
one run directory are rejected.

## 4. Use Your Own Camera Video and Photos

No code changes are needed for another static camera view:

```bash
python3 scripts/run_demo.py --backend docker \
  --video /path/to/new_site/entrance.mp4 \
  --photos /path/to/new_site/employee_photos \
  --models /path/to/CAMEX_Face_ID_Test_Kit_0.1.1/models \
  --output /path/to/face_id_runs/new_site \
  --camera-id entrance_01 --clock-start 09:00:00
```

Use one clear, single-face photo per person, with a unique filename stem as the
person's label. Do not use duplicate stems with different extensions or put two
people under one label. This preview has no multi-photo enrollment UI. Re-enroll
after changing photos; keep labels consistent when mapping to employee records.

For a long or poorly indexed recording, create a separate working clip:

```bash
python3 scripts/prepare_video.py --input /path/to/raw_video.mp4 \
  --start 0 --seconds 300 --fps 10 --output /path/to/new_site/entrance.mp4
```

FFmpeg is required. This preserves the raw file, creates a seekable intermediate
and records hashes. Check the visible camera clock at the first and last frame.
Changing an extension to `.mp4` does not fix an MPEG program stream or bad timestamps.

Use `--config /path/to/site_config.json` to tune copied settings on separate labeled
validation footage. Do not lower recognition thresholds merely to show more names.
Unknown can reflect absent enrollment or insufficient face evidence.

## 5. Platform Integration

```python
import json
from camex_faceid.models import Networks
from camex_faceid.gallery import Gallery
from camex_faceid.pipeline import FaceIDService

with open("configs/default.json") as stream:
    config = json.load(stream)
networks = Networks("/path/to/models", provider="CUDAExecutionProvider")
gallery = Gallery("/path/to/enrolled_gallery")
service = FaceIDService(networks, gallery, config, camera_id="entrance_01")

result = service.process(bgr_frame, timestamp_ms)
platform.handle_face_id_result(result)
```

One service instance per stream; calls are synchronous and must be serialized.
Supply uint8 BGR images and strictly increasing integer source timestamps in
milliseconds. The platform owns live decoding, scheduling/latest-frame buffering,
storage, access control and alerts. The file-demo launcher is not a live API.

`track_id` is camera-local; nullable `person_id` identifies an enrolled reference.
`similarity` is a cosine score, not a probability or measured accuracy. States
are `no_usable_face`, `pending`, `unknown`, and `identified`. A name may carry its
previous face confirmation; inspect `last_face_confirmed_ms`.

Handle **both** `identity_confirmed` and `identity_revoked`. Ambiguity, conflict,
expiry or loss can revoke a name. Treat `tracks` as the current observed snapshot
and remove absent overlays. See [the event contract](docs/INTEGRATION.md).

## Head-Only Confirmation

The chair fix requires two head scores >=0.65 within 700 ms plus motion evidence
before publishing a new head-only track. An associated face can verify it immediately.
Already verified tracks can continue at lower confidence. Detector thresholds
remain 0.40/0.15; Face ID thresholds are 0.45 similarity and 0.08 top-two margin.
These are tunable starting points, not universal settings.

The separately supplied ONNX files are checksum-verified. Review
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and the model-use notes.
