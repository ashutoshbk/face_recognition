# Office Demo: Local Runtime and Deployment

Standalone office evaluation app on **127.0.0.1:9170**, separate from
`research_demo` and existing services. It adds named-folder enrollment, two
five-photo reference collections, configurable RTSP inputs, head tracking and
progressive recognition. No model downloads or old tracking JSON replay are used.

## Updating the Previous Delivery

Preserve your local changes and the previous runtime. Do not reset the checkout.

```bash
git fetch origin
git switch codex/office-face-enrollment-multicam
cd face_id_poc
```

Use the existing model bundle, with the same verified model hashes. No download
or rebuild of the large synthetic gallery is needed for this office entry point.

## Enrollment and Reference Collections

In **People > Enroll images**, select one person and multiple photos, or select a
parent folder containing one named subfolder per person. ZIP uploads of that
layout are accepted. **Capture photos** uses an explicitly started browser webcam.
Confirm authorized/consenting use, review the proposed import, then commit it.
The folder command uses the same API and checks:

```bash
python3 office_demo/scripts/enroll_folder.py /private/Staff --consent --commit
# Legacy folder containing one named file per person:
python3 office_demo/scripts/enroll_folder.py /private/Staff --flat-names --consent --commit
```

Omit `--commit` for preview only. Exact existing names appear as updates in the UI
preview; verify the person before submitting. Incorrect manual labels cannot be
corrected by image-quality scoring alone.

```text
OFFICE_RUNTIME_DIR/Face_ID/people/<stable-person-id>/<display-name>/
  registered/  up to five manually enrolled JPEG references
  live/        up to five approved live JPEG references
  metadata.json
```

Eligible photos fill empty slots; a better photo replaces the lowest-scoring
active reference when full. Duplicates and unusable faces are rejected. Registered
changes invalidate live references until they are checked again.

Recognition searches registered and active live references in Qdrant. Each new
track observation contributes one person-level vote, not one vote per reference.
The first qualifying name is Tentative. Further observations may confirm, revise
or clear it. Five recent observations are kept per track; quality and similarity
scores are not calibrated identity probabilities or liveness evidence.

Live admission independently checks registered references only: 2/2, 3/3, 3/4 or
4/5 strong agreements, a competitor margin and repeated track observations.
Existing live photos cannot authorize further additions. One registered photo
supports recognition but does not enable automatic live admission. The supplied
office folder has one portrait per person, so empty live slots are expected.

Review is the default. Automatic mode also requires the deployment opt-in
`CAMEX_OFFICE_ALLOW_AUTO_ADMISSION=1` after threshold validation. Both target and
gallery-wide registered revisions are rechecked under the write lock. Failed
filesystem cleanup is retained in a retry worklist, visible in diagnostics.

Camera count is configurable, with a default resource guard of 32 through
`CAMEX_OFFICE_CAMERA_LIMIT`; that is not a throughput claim. GPU mode uses two
bounded tracking workers sharing sessions, per-camera serial trackers and a
separate recognition worker. CPU mode uses one tracking worker.

## Supported GPU Stack

Target: the **NVIDIA TITAN X Pascal, 12 GB, compute capability 6.1** test server,
with Ubuntu x86_64 and driver 535. This is not an RTX 5000 deployment or a claim
of measured throughput. Five displayed streams include two duplicates, not five
independent cameras. Real GPU execution and sustained performance must be checked
on the target server.

| Component | Pin |
| --- | --- |
| Base image | `nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04` |
| Python | Ubuntu 22.04 Python 3.10, isolated venv, `python -m pip` |
| GPU ONNX Runtime | `1.18.0`, official **CUDA 12 / cuDNN 8** Linux x86_64 wheel |
| NumPy | `1.26.4` (keep below 2) |
| Qdrant server / client | `qdrant/qdrant:v1.15.5` / `qdrant-client==1.15.1` |
| Fixture RTSP server | `bluenviron/mediamtx:1.15.3` |

The [ORT compatibility matrix](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html)
lists 1.18.0 with CUDA 12 and cuDNN 8; 1.18.1's CUDA 12 build uses cuDNN 9.
The [official installation instructions](https://onnxruntime.ai/docs/install/)
link the older CUDA 12 wheels separately. `requirements-gpu.txt` therefore pins
the official CPython 3.10 wheel URL and SHA-256, **not** the CUDA 11.8 build of
1.18.0 on PyPI. Do not replace this with the latest ORT/CUDA 13 package.

[NVIDIA's CUDA 12.2.2 release notes](https://docs.nvidia.com/cuda/archive/12.2.2/cuda-toolkit-release-notes/index.html)
list 535.104.05 as the corresponding Linux driver and 525.60.13 as the CUDA 12.x
minor-compatibility minimum. Verify the actual 535 patch release with `nvidia-smi`;
535.104.05 or later in the 535 series is preferred. Do not upgrade the host driver
or install a host CUDA toolkit as part of this demo. Docker Engine, Compose v2+
and the NVIDIA Container Toolkit must already be installed/configured by the
server administrator. No TensorRT, Torch or FP16 requirement is introduced.
Camera decode defaults to FFmpeg software decode. Set `OFFICE_HWDECODE=1` before
Compose startup to request NVDEC; the video driver capability and supported codecs
are required. The selected decoder is shown per camera. Normal operation has
model profiling disabled; use `check_runtime.py --exercise-models` for an isolated
CUDA kernel check instead of measuring performance with profiling enabled.

Qdrant [v1.15.5 is a published release](https://github.com/qdrant/qdrant/releases/tag/v1.15.5).
Its client is pinned to the same minor family, independently of the existing
research environment's 1.19.x client. Never attach this server to an existing
Qdrant storage directory. Do not substitute unpinned `latest` on failure. The
first-round report records the versions actually built and exercised.

## Private Paths and Model Prerequisites

Use a new runtime outside the source checkout. The extracted, already supplied
model directory must contain these three files at its top level:

```text
arcface-r50-w600k-112x112.onnx
scrfd-kps-2.5g-640x640.onnx
yolox-tiny-crowdhuman-416x416.onnx
```

Use the provided `CAMEX_Research_Models_20261014.zip` or the already provided
model folder, preserving its license/research restrictions. Extract it yourself
to a private location; neither build nor startup downloads/extracts models.
Checksums come from `research_demo/vision/networks.py`. Models mount read-only.

From your **face_id_poc** directory (adapt the source path on Ubuntu):

```bash
cd /path/to/face_id_poc
export OFFICE_RUNTIME_DIR="$HOME/camex-office-runtime"
export OFFICE_MODELS_DIR="/absolute/path/to/provided/models"
export OFFICE_FIXTURE_DIR="$OFFICE_RUNTIME_DIR/fixture"
export OFFICE_UID="$(id -u)"
export OFFICE_GID="$(id -g)"
umask 077
mkdir -p "$OFFICE_RUNTIME_DIR"
chmod 700 "$OFFICE_RUNTIME_DIR"
```

Private JSON containing camera addresses must remain mode `600`; directories
should be `700`. All provided scripts use `umask 077`; the app runs under the
configured host UID/GID. Do not use `chmod 777`. Do not commit or publish runtime
data, camera credentials, face images, SQLite databases, Qdrant volumes, model
weights, or entire fixture directories. No `.env` file is required; if you create
one locally, keep it outside version control and `chmod 600` it.

## Build and Start on Ubuntu GPU

Preferred build (installs inside Docker only when you explicitly run it):

```bash
python3 office_demo/scripts/build_image.py
bash office_demo/scripts/office_stack.sh config
bash office_demo/scripts/office_stack.sh up
bash office_demo/scripts/office_stack.sh status
bash office_demo/scripts/office_stack.sh logs
```

The builder stages an allowlisted **face_id_poc-shaped** context: office Python,
static assets, deployment scripts, requirements, and shared `research_demo/vision`
plus `research_demo/core.py`. It copies the office `.dockerignore` to the staging
context root. No models, media, enrollment data, or research runtime are staged.
This also avoids sending the surrounding delivery archive to Docker.
Neither `core.py` nor the shared vision adapters import `research_demo/settings.py`;
that research-service configuration is deliberately not copied or used.

For a reviewed, clean `face_id_poc` context only, the equivalent direct command is:

```bash
docker build --platform linux/amd64 -f office_demo/Dockerfile \
  -t camex-office-demo:cuda12.2-ort1.18.0 .
```

Do **not** run that direct command from `office_demo/` or from the parent delivery
repository. Its context must contain both `office_demo/` and `research_demo/`.
The nested `.dockerignore` is not active in a direct parent-context build; use the
staging builder for private worktrees.

The `camex-office-demo` Compose project starts only its own services and volume:

| Service | Host access | In-container access |
| --- | --- | --- |
| Office app | `http://127.0.0.1:9170` | Uvicorn listens on `0.0.0.0:9170` |
| Qdrant REST | Not published by Compose | App uses `http://qdrant:6333` on the private database network |
| Fixture RTSP, opt-in | `rtsp://127.0.0.1:18555` | App uses `rtsp://mediamtx:8554` |

Qdrant's database network is internal; only the app shares it. The stream network
also lets the app reach authorized physical camera IPs. Qdrant gRPC and other
MediaMTX protocols/admin APIs are not published. Host ports bind to IPv4 loopback,
not the LAN; check the host firewall/Docker policy too. No existing ports, Docker
networks, or research services are stopped/reconfigured. A port conflict is an
error to investigate, not permission to kill another service.

The app gets `CAMEX_OFFICE_HOME=/runtime`, `CAMEX_OFFICE_MODELS=/models`, and
`CAMEX_OFFICE_PROVIDER=CUDAExecutionProvider`. With bridge networking, do not set
`CAMEX_OFFICE_QDRANT_URL=http://127.0.0.1:16343` inside the app container: that
loopback belongs to the app container, not Qdrant. That URL is correct for a
native host process, or Linux-only host networking (not the default here).

After startup, verify dependencies and model hashes; explicitly run the second
command to exercise each model once with blank tensors and inspect real provider
execution. This does not enroll or identify anybody:

```bash
docker compose -p camex-office-demo -f office_demo/docker-compose.yml \
  exec app python scripts/check_runtime.py
docker compose -p camex-office-demo -f office_demo/docker-compose.yml \
  exec app python scripts/check_runtime.py --exercise-models
```

Listing `CUDAExecutionProvider` alone is not proof that CUDA kernels ran. The
exercise command profiles executed nodes and fails if the requested provider
does not execute any. The first-round report contains the observed server result.

The backend's `/api/health` returns a `ready` Boolean covering models, workers and
index readiness; HTTP 200 alone is not readiness. Normal operation disables
profiling. For a diagnostic launch explicitly using `CAMEX_OFFICE_PROFILE=1`,
`GET /api/profile` finalizes the trace once. Do not use profiled timings as normal
capacity results. Prefer the standalone checker for proving CUDA execution.

Stop only this stack, retaining its data:

```bash
bash office_demo/scripts/office_stack.sh down
```

Do not add `--volumes` unless you deliberately intend to destroy this demo's
Qdrant index. Retain the runtime and Qdrant volume together when backing up.

## Prepare the One-Minute Fixture

Preparation requires local FFmpeg/ffprobe. By default it reads only:

```text
/Volumes/SamsungT7/AINLab_Office_3D_Tracking_Visualization/inputs/videos/camera_a.mp4
/Volumes/SamsungT7/AINLab_Office_3D_Tracking_Visualization/inputs/videos/camera_b.mp4
/Volumes/SamsungT7/AINLab_Office_3D_Tracking_Visualization/inputs/videos/camera_c.mp4
```

The read-only probe confirmed all three are H.264, 20 FPS; A starts at PTS
0.000 s, B/C at 0.050 s. A lasts 125.000 s; B/C last 124.950 s. The source
project README describes clip-relative alignment, explicitly not hardware clock
validation. The default fixture preserves those offsets and takes the common first
minute using `-copyts -to 60 -c copy`, video only, without re-encoding. GOP/B-frame
boundaries can extend the last packet slightly; actual timestamps/durations and
source/output hashes are recorded in the manifest. No timestamp/NTP validation,
new calibration, identity inference, or old tracking-result JSON is performed.

```bash
python3 office_demo/scripts/prepare_office_fixture.py --check-only
python3 office_demo/scripts/prepare_office_fixture.py \
  --transport-profile closed-gop --output "$OFFICE_FIXTURE_DIR"
```

`--start-offset 0` is the default on macOS and Ubuntu. A nonzero common offset is
supported only at a keyframe boundary in all three sources; each source's original
0/0.05-second phase is retained, and the output timeline is shifted by the same
common offset. Use `--check-only --start-offset 1` to check before preparing a new
directory. These supplied clips have long GOPs, so **1 second is not a valid
stream-copy boundary** and is rejected, not silently presented as an accurate cut.
Use offset 0 for this fixture. The optional `closed-gop` profile re-encodes a
separate copy with a one-second keyframe interval, no B-frames and normalized
timestamps, for reliable repeated RTSP playback. It retains 2560x1440 at 20 FPS;
it is not the original compressed bitstream. Default encoder is libx264 (CRF 15).
`--encoder h264_nvenc` uses constant QP 18 when the NVIDIA encoder is available.
The first-round interactive demo uses this profile because the stream-copy
fixtures exhibited decoder corruption across joins/loop boundaries. Preserve
the originals for separate quality evaluation. Alignment remains approximate
clip-relative timing, not verified camera-clock/NTP synchronization.

On Ubuntu, provide `--source-dir /private/path/to/the/three/videos` after an
authorized transfer. The output must not exist; preparation refuses overwrite
and publishes the completed directory only after verifying all three copies.
`manifest.json`, `cameras.host.json`, `cameras.docker.json`, and the MP4s are mode
`600`. This script neither finds nor copies staff photos, face galleries, or
`recorded_results.jsonl`.

Start the dedicated RTSP server and supervised publishers:

```bash
bash office_demo/scripts/office_stack.sh fixture-up
bash office_demo/scripts/office_stack.sh logs
```

For a publisher on the host talking to the same dedicated MediaMTX container:

```bash
docker compose -p camex-office-demo -f office_demo/docker-compose.yml \
  --profile fixture up -d mediamtx
bash office_demo/scripts/publish_fixture.sh "$OFFICE_FIXTURE_DIR"
```

Do not run both publisher modes simultaneously. Both use FFmpeg
`-re -stream_loop -1 -c copy -rtsp_transport tcp -f rtsp`. Each group starts its
five processes together, reports launch spread, and stops the group if one
publisher exits. RTSP publishing and repeated GOPs are a best-effort synchronized
replay, not frame-locked live cameras or NTP. All sources are recorded fixtures:

| Stream path | Meaning |
| --- | --- |
| `camera_a` | Reception, unique source A |
| `camera_b` | Workspace 1, unique source B |
| `camera_c` | Workspace 2, unique source C |
| `camera_d_replay_a` | Explicit REPLAY of A, not another viewpoint |
| `camera_e_replay_b` | Explicit REPLAY of B, not another viewpoint |

The JSON files contain camera names/URLs for manual entry or optional API
import. Use **docker** URLs for the container app and **host** URLs for a native
app. They are fixture manifests, not the app's private runtime configuration;
do not overwrite `runtime/config/cameras.json` with them. Camera API import is
optional. No import or consent assertion happens automatically. After confirming
authorization and backend readiness, either enter the feeds through the UI or run:

```bash
python3 office_demo/scripts/import_fixture.py \
  "$OFFICE_FIXTURE_DIR/cameras.docker.json" --authorized
```

This requires a private config, checks `/api/health`, preserves the REPLAY labels,
and adds only camera records through `/api/cameras`; it performs no enrollment.
Cameras stay stopped for review. Append `--start` only to start those five new
cameras, not other configured cameras. The script refuses duplicate fixture
names; after a partial failure inspect the UI before retrying. Never use
`--authorized` as a substitute for obtaining authorization. Stop the fixture
alone with `bash office_demo/scripts/office_stack.sh fixture-down`.

## Enroll Consenting Participants Only

The operator has authorized local staff enrollment and matching. The planned
evaluation uses **20 real portraits, one per person**. These deployment scripts
still do not copy/import portraits or enroll anybody automatically. Use only the
explicitly authorized source images; do not mine these videos, reuse old track
IDs, or expand that authorization to unrelated staff-photo directories.

With one registered portrait per person, ordinary local matching can be evaluated,
but the registered-only admission gate requires at least **two distinct trusted
registered references**. No live candidate can pass that gate with the present
one-portrait seeds, even in review mode; Live Photos should remain empty. Do not
duplicate a portrait to simulate a second seed, bypass the gate, or enable
`CAMEX_OFFICE_ALLOW_AUTO_ADMISSION`. A second independently approved reference per
person and threshold validation are needed before evaluating live admission.

Once authorized, use the app's enrollment dialog:

1. **One person / individual photos:** select Person, enter the participant's
   name, select JPG/PNG images of that person, confirm the consent statement, and
   preview. Inspect accepted/rejected results and explicitly confirm the import
   rather than assuming every file becomes a reference.
2. **One-person folder:** select Folders and choose a directory named for that
   participant containing only their JPG/PNG reference images. Preview the group
   name and results, then confirm.
3. **Bulk folders:** select Folders and choose the common parent of one directory
   per participant. Review each inferred folder/person name before confirming.
   Do not mix multiple people in a single directory or nest unrelated folders.

```text
consented-reference-import/
  Person_A/
    front.jpg
    slight_left.jpg
    slight_right.jpg
  Person_B/
    front.jpg
    alternate_lighting.jpg
```

Each person has **two separate galleries, up to five photos each**: Registered
Photos (trusted enrollment anchors) and Live Photos (camera candidates admitted
through the app's registered-only evidence/review policy). Folder import feeds
Registered Photos, not both galleries. Low-quality/duplicate photos can be
rejected; a full pool only replaces a worse reference. Review live candidates
before admitting them; automatic admission remains locked by default. A gallery
change invalidates relevant evidence, so do not equate replay continuity with
verified identity. Confirm UI/import behavior against the finished backend before
a participant session. Imports accept up to 200 JPG/PNG files and 128 MB total;
larger authorized batches should be split and each preview reviewed.

## Native CPU Development

Use a separate Python **3.10 or 3.11** environment. A C/C++ compiler and matching
Python development headers are needed to build `cython-bbox`; on Ubuntu these
come from `build-essential` and `python3.10-dev`. Do not install CPU and GPU ORT
into the same environment. FFmpeg/ffprobe and a reachable, dedicated Qdrant server
are still needed. These commands install only when explicitly run by you:

```bash
python3.10 -m venv "$HOME/.venvs/camex-office-cpu"
source "$HOME/.venvs/camex-office-cpu/bin/activate"
python -m pip install --index-url https://pypi.org/simple \
  pip==24.3.1 setuptools==75.6.0 wheel==0.45.1 Cython==3.0.11 numpy==1.26.4
python -m pip install --index-url https://pypi.org/simple --no-build-isolation \
  -r office_demo/requirements-cpu.txt
python -m pip check
docker run -d --name camex-office-native-qdrant \
  -p 127.0.0.1:16343:6333 -p 127.0.0.1:16344:6334 \
  -e QDRANT__TELEMETRY_DISABLED=true \
  -v camex-office-native-qdrant:/qdrant/storage qdrant/qdrant:v1.15.5
export CAMEX_OFFICE_HOME="$OFFICE_RUNTIME_DIR"
export CAMEX_OFFICE_MODELS="$OFFICE_MODELS_DIR"
export CAMEX_OFFICE_PROVIDER=CPUExecutionProvider
export CAMEX_OFFICE_QDRANT_URL=http://127.0.0.1:16343
export PYTHONPATH="$PWD:$PWD/research_demo:$PWD/office_demo"
python office_demo/scripts/check_runtime.py
cd office_demo
umask 077
python -m uvicorn app:app --host 127.0.0.1 --port 9170 --workers 1
```

Do not run the native app while the container already owns port 9170. Do not run
multiple app workers against this demo runtime. On macOS use CPU mode only; this
CUDA image targets Linux AMD64, not Apple GPUs.

## Verification Status

See `SMOKE_TEST_RESULTS.md` for the dated first-round measurements and limits.
Short replay tests and synthetic failure tests do not establish real-person
recognition accuracy, liveness, production readiness or another GPU's capacity.
