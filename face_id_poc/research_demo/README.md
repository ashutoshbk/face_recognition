# Local Face Research Demo

Reproduce the Mac research experiment on a MacBook or Ubuntu computer: detect and
track heads, associate faces, register one consenting tester, and search that
template alongside a synthetic DigiFace gallery in Qdrant. Detection/tracking,
recognition and RTSP decoding use separate processes. Recognition is capped at
one job per second; the displayed camera frame and its overlays stay synchronized.

This is a research-only test station, **not** an unrestricted real-person gallery
or unattended CCTV deployment. Read [asset and model restrictions](THIRD_PARTY_NOTICES.md)
before downloading or using the research data. The code does not grant commercial
rights to the dataset, results or pretrained weights.

## What you need

- Python **3.11**, Git and curl. Mac: Command Line Tools and macOS 14 or later for
  the pinned native wheels. Ubuntu: a compiler toolchain and Python development
  headers if a native wheel is unavailable. No CUDA/GPU is required for this build.
- Recommended: 16 GB RAM and 50 GB free space. The eight dataset ZIPs total
  approximately 23.17 GiB. They stay compressed; do not extract all images.
- A browser with camera permission, or one authorized RTSP test feed reachable
  from the computer running the backend. Use a controlled scene with consenting
  participants; multiple heads/faces pause enrollment and recognition.
- The separate `CAMEX_Research_Models_20261014.zip`, or the `models` folder from
  the previously delivered `CAMEX_Face_ID_Test_Kit_0.1.1`. This prerequisite is
  required: the custom checkpoint has no verified public download URL. Obtain
  the companion ZIP from the handoff owner, not a similarly named COCO model.

Mac Apple Silicon was exercised locally. Linux unit-test status is reported in
the PR/CI; physical Ubuntu webcam, client RTSP feeds, Intel Mac and ARM Linux
hardware are not claimed as tested merely because installers are provided.

## 1. Get the code and environment

Repository access is required. Sparse checkout avoids downloading the unrelated
historical staff-photo folder in this repository.

```sh
git clone --filter=blob:none --sparse --branch codex/local-face-research-20261014 https://github.com/ashutoshbk/face_recognition.git
cd face_recognition
git sparse-checkout set face_id_poc/research_demo
cd face_id_poc/research_demo
bash bootstrap.sh
source .venv/bin/activate
```

`bootstrap.sh` expects `python3.11` on PATH, or `PYTHON=/absolute/path/to/python3.11`.
On Ubuntu versions whose default Python differs, install Python 3.11 using your
approved package manager or a managed-Python tool such as uv; do not replace the
system Python. The [uv managed Python guide](https://docs.astral.sh/uv/guides/install-python/)
provides that setup. With uv already installed:

```sh
uv python install 3.11
PYTHON="$(uv python find 3.11)" bash bootstrap.sh
source .venv/bin/activate
```

Optionally set `CAMEX_LAB_HOME` to an external disk **before** the following steps.
Keep the same value in later terminals. Otherwise data is isolated in `.runtime/`.

```sh
export CAMEX_LAB_HOME="/path/to/your/camex-runtime"
python lab.py assets --model-bundle "$HOME/Downloads/CAMEX_Research_Models_20261014.zip" --accept-research-terms
python lab.py doctor
```

The bundle can instead be an existing model directory or a direct HTTPS ZIP URL.
MEGA/Drive sharing pages are not direct ZIP URLs: download them in a browser first.
The importer checks every model hash and installs the correct native Qdrant
binary for macOS/Linux on ARM64 or x86-64. It does not install GPU drivers.

## 2. Quick smoke test

Start here before the full multi-hour run. It needs the first official dataset
archive (about 2.74 GiB), creates a 256-identity gallery, and measures held-out images.
The acceptance flag is for operators who have reviewed the research terms above.

```sh
python lab.py dataset --parts 1 --accept-research-terms
python manage.py start-db
python -m synthetic_bench.pipeline --name quick256 --identities 256 --probes 64 --queries 64
export CAMEX_GALLERY_RUN=quick256
python lab.py start
```

Open **http://127.0.0.1:9160/?mode=self**. The gallery page is
**http://127.0.0.1:9161/**. Counts and timings on this quick run refer to 256
synthetic identities, not the completed 109,619-identity development benchmark.

## 3. Build the full synthetic gallery

Stop the quick-test services before changing gallery selection. Download all
eight parts, then keep the computer powered until the foreground build finishes.

```sh
python lab.py stop
unset CAMEX_GALLERY_RUN
python lab.py dataset --accept-research-terms
python lab.py build
```

Downloads resume through `.part` files and are checked against pinned SHA-256
hashes. `--check-crc` also scans every member. The build reads ZIP members directly,
uses checkpointed SQLite descriptors, loads Qdrant, waits for indexing, then
evaluates held-out probes and starts the UI. Re-running the same command resumes
completed batches; do not change its configuration under the same run name.

The default targets **109,999 fictional identities**, one reference per identity,
not every image in the 1,219,995-image dataset. Image 1 is held out for probes;
gallery candidates are 0, then 2/3 only if detection fails. Failed images are
reported, not silently counted as enrolled. The full M1 Pro CPU run took several
hours. Different machines and workloads require their own timing measurements.

## 4. Register and test yourself

1. Select **My face + synthetic gallery** and **Webcam**, then Start camera.
2. Keep one person in view. After at least three accepted samples, enter your
   label, confirm consent and press Register best 3 samples.
3. Three quality-weighted ArcFace vectors form one template. SQLite stores the
   label/reference photo; a separate one-point Qdrant collection stores the template.
4. Each lookup searches the synthetic index and self collection, then ranks the
   combined candidates. A positive vote requires the self profile to rank first
   and exceed the slider threshold. Three of five time-spaced votes, plus a positive
   latest vote, produce Match; a fresh five-query window takes roughly four seconds.
5. Move/turn your head. Head IDs may continue without a visible face, but face loss,
   ambiguity, stale input or camera stop clears the identity verdict. Late background
   results cannot restore an obsolete match.
6. Assess / Use suggestion examines synthetic-negative separation. It does **not**
   establish real-person false-positive rates, liveness or a deployment threshold.
7. Delete the saved profile with the trash button to re-enroll. Original synthetic
   benchmark results remain unchanged. Do not share your runtime folder.

The UI reports detection/tracking FPS, recognition encoding/lookup/total time,
gallery size and returned candidates. HNSW is approximate: gallery size is not
the exact number of vectors inspected per query.

## 5. Use a controlled RTSP feed

Select **Controlled RTSP test feed**, enter its address in the password field,
confirm authorization and start. The backend decodes one feed into a latest-frame
buffer; the same synchronized detection and recognition pipeline processes it.
The URL stays in memory, is not saved to settings, and native decoder stderr is
suppressed to avoid logging embedded credentials. No video is recorded.

The camera must be reachable from the backend, not merely from your browser.
Start with a supported H.264/TCP feed. A disconnected/stale feed clears capture
state rather than retaining a successful identity result. Real client feeds,
RTSPS servers and unusual codecs need separate site tests.

For Ubuntu without a desktop, forward loopback ports over SSH and open the local
URL in your own browser. Your browser webcam then supplies frames to that backend.
Do not expose these unauthenticated research services directly to the LAN/internet.

```sh
ssh -L 9160:127.0.0.1:9160 -L 9161:127.0.0.1:9161 USER@UBUNTU_HOST
```

## Operations and troubleshooting

```sh
python lab.py status
python lab.py doctor
python lab.py stop
python lab.py start
```

- Default ports: UI 9160, gallery 9161, Qdrant HTTP 16333 / gRPC 16334, all loopback.
  Overrides: `CAMEX_PORT`, `CAMEX_GALLERY_PORT`, `CAMEX_QDRANT_PORT`,
  `CAMEX_QDRANT_GRPC_PORT`. Set all consistently before starting processes.
- Paths: `CAMEX_LAB_HOME`, `CAMEX_MODELS_DIR`, `CAMEX_DATASET_DIR`; gallery run:
  `CAMEX_GALLERY_RUN`. Never commit runtime data or stream credentials.
- Logs are in the runtime `logs/` directory. A missing model or gallery result is
  a setup issue; use doctor before changing thresholds.
- An occupied port is left untouched. Stop this package's owned services or choose
  other ports; the manager will not kill an unrelated application.
- Camera permission denied: use localhost or an SSH tunnel, grant browser permission,
  and close another app using the device.
- Recognition unavailable: check the completed gallery result and worker status in
  `/api/health`. Recognition failure does not move expensive work into tracking.
- Stop a foreground build with Ctrl-C; its current completed batches are retained.
  For a build started from the gallery UI, use Pause and wait for its paused state
  before stopping services or disconnecting storage.

## Tests and evidence

Optional browser checks (Node.js 20+, synthetic fixtures only):

```bash
python lab.py fixtures
npm install
npx playwright install chromium
npm run test:browser
npm run test:gallery
```

The browser test starts an isolated self-profile service on port 9162 and deletes
its temporary Qdrant collection. Set `CAMEX_QA_PORT` if occupied. It preserves the
selected benchmark collection and your normal enrolled profile. Set
`CAMEX_CHROME` to an installed Chrome executable to avoid downloading Chromium.
Generated synthetic screenshots remain local and must not be committed.

The opt-in `python rtsp_smoke.py` test requires FFmpeg, a MediaMTX executable
(`MEDIAMTX_BIN`) and an **empty separate test runtime**. It publishes a synthetic
clip locally, verifies enrollment/search and stream-loss handling, then removes
its test profile. It refuses to run against an existing enrollment.

```sh
python lab.py fixtures   # needs FFmpeg and the first official dataset archive
python lab.py test
```

Without model/data assets, unit tests still run; asset-dependent tests report a
skip rather than claiming model validation. The development run's aggregate
measurements are in `evidence/benchmark_summary.json`. They are not hardware promises.

Delivery checks on 14 October 2026: Mac clean environment **61 passed**, browser
enrollment/search/synchronization checks passed, and local synthetic H.264/TCP
RTSP registration/match/disconnect checks passed. A clean Ubuntu 24.04 CI install
passed **59 tests with 2 asset-dependent skips**. See `evidence/mac_validation.json`
and `evidence/ubuntu_validation.json`. The complete gallery was measured on Mac;
Ubuntu camera/model integration and GPU performance still require separate tests.

The implementation diagram is in [ARCHITECTURE.md](ARCHITECTURE.md). The broader
parent Face ID PoC remains unchanged; this module has its own environment and data.
