# Office Face ID: First-Round Smoke Test

Date: 20 October 2026. Status: functional evaluation, not production acceptance.

## Implemented

- Single-person photo/webcam enrollment and previewed folder/ZIP bulk imports.
- Separate registered and live collections, each bounded to five eligible photos.
- Quality scoring, duplicate rejection and lower-score reference replacement.
- Configurable cameras, independent capture/reconnection and head-track state.
- YOLOX heads, ByteTrack, SCRFD, ArcFace and shared Qdrant reference search.
- Tentative/confirmed/unknown voting over recent observations; asynchronous recognition.
- Registered-only live-admission consensus, stricter margins, anchor revisions and stale-result guards.
- Private local dashboard, per-camera timings, GPU/CPU diagnostics and exportable metrics.

The supplied 20 staff portraits were explicitly authorized for local evaluation,
previewed and enrolled successfully. Repeating the import added zero references
and left all 20 registrations intact. Original input files were not modified.

## Hardware and Inputs

| Item | Tested configuration |
| --- | --- |
| Server | Ubuntu 22.04.5, Ryzen 7 2700, about 24 GB RAM |
| GPU | NVIDIA TITAN X Pascal, 12 GB, driver 535.274.02 |
| Runtime | CUDA 12.2 / cuDNN 8, ONNX Runtime GPU 1.18.0, Python 3.10 |
| Models | Existing YOLOX-Tiny CrowdHuman, SCRFD-2.5G, ArcFace R50 exports; hashes checked |
| Search | Qdrant 1.15.5, exact per-reference cosine scores grouped by person; gRPC inside the private network |
| Footage | Three supplied office views: Reception, Work Space 1, Work Space 2 |
| Connections | Five independent RTSP decoder inputs: three views plus two explicitly labelled replays |
| Video profile | 2560x1440, 20 FPS source, one-minute clips replayed over RTSP/TCP |
| Processing | Capture capped at 10 FPS, analysis target 10 FPS, recognition limited per track |

The interactive fixture is a separately encoded, high-quality H.264 copy with
one-second GOPs and no B-frames. Originals remain unchanged. This is not a
bitrate-preservation or camera-native codec benchmark. The three source clocks
begin at 16:35:00 on the supplied recording date. The replay is not an NTP test.

## Observed Performance

The final measured window was 60 seconds after 7.9 seconds waiting for initial
decoded/analysed frames. The following are measured rates, not the configured target.

| Input | Mean analysis FPS | Minimum reported rolling FPS |
| --- | ---: | ---: |
| Reception | 5.37 | 4.63 |
| Work Space 1 | 5.37 | 4.67 |
| Work Space 2 | 5.34 | 4.27 |
| Reception replay | 5.38 | 4.64 |
| Work Space 1 replay | 5.35 | 4.38 |

- GPU utilization averaged about 38%; CPU utilization about 54%.
- Maximum observed GPU temperature was 77 C. The application stops its camera sources at 85 C.
- Maximum total GPU memory observed was 3,712 MiB. GPU figures include other applications on the host.
- The mean of sampled background-recognition timing readouts was about 154 ms, including encoding and search; it is not a percentile or a recognition accuracy result.
- The capture path reported approximately 10 frames/s; preview polling does not create extra analysed frames.
- A separate stored-vector sanity check returned the expected person for 20/20 enrolled vectors. Median query time was 6.6 ms; the 20-query cold-start p95 was 300 ms. This checks indexing/mapping, not held-out face recognition.

Ten analysed FPS on every input was not achieved in this round. CPU-side image
handling and scheduling remain optimization work. These results are not an RTX
5000 benchmark, a maximum-camera claim or an all-day stability guarantee.

## Transport Corrections

The initial replay was not accepted: long-GOP joins and relay queue overflow
produced visible corruption and reader drift. The retained fixture uses a
closed-GOP copy, and the dedicated relay write queue is increased to 8,192.
Capture is separate from analysis locking, raw-frame reads avoid an intermediate
copy, and rawvideo output has one encoder thread. Rechecked frames were clean,
the repeated Reception clocks agreed, and the relay stopped reporting dropped
frames in the checked final interval. The source originals were not altered.

Normal model profiling is disabled. CUDA was verified separately by executing
all three actual models and inspecting profiled CUDA node events. GPU availability
alone was not used as proof. The final capture run requested NVDEC; software
decode remains the explicit default for installations that have not verified it.

## Functional Checks

- 43 new automated checks passed, plus an explicit real-model execution check.
- 78 existing research-demo regression tests passed; their code was not changed.
- Tests cover consensus counts, competitor margins, stale anchor/track results,
  bounded pools, duplicates, invalid paths, ZIP imports, index-failure recovery,
  cleanup retries and keeping capture independent of the analysis lock.
- A sixth configured test source was accepted. Its intentional connection failure
  did not stop the five working sources; the temporary source was then removed.
- A working source was independently stopped and reconnected while the other four continued.
- Browser checks covered camera/people/diagnostic views, enrollment preview/commit,
  webcam UI behavior, error states and responsive layouts. Real desktop walkthrough
  recording completed without JavaScript page errors; physical webcam testing by
  the operator remains to be done.

## Recognition Limits and Next Review

The office cameras are high/wide views. Many faces fail minimum size, pose or
quality checks and correctly remain Unknown. This first round did not establish
manually labelled true/false identification rates. A temporary Confirmed state
was observed in the smoke data, but it was not independently identity-verified
and is not evidence of recognition accuracy.

Each supplied person currently has only one trusted registered portrait.
Automatic admission to the live collection is therefore intentionally blocked;
no unqualified live references were added. Supply at least two independently
verified registered references (preferably three or more) and validate the
registered-only threshold/margin before enabling automatic updates. Do not
manufacture additional trusted references by accepting the system's own guesses.

The next user review should cover face quality, provisional labels, track
continuity and single-person capture, followed by longer testing and real labelled
acceptance on the team's five Hikvision cameras. Model licensing, liveness,
authentication and production deployment acceptance are separate requirements.

## Reproduction

This is historical first-round evidence, not the current installation guide.
Follow the [Ubuntu/NVIDIA runbook](README.md) for a clean installation using your
own private paths, authorized references and RTSP cameras. Operator-specific
host addresses and prepared datasets are not prerequisites and are intentionally
omitted from this handoff.

See [the latest validation](VALIDATION_20261020.md) for the subsequent
SCRFD-2.5G synchronized replay check and the exact limits of the results.
