# SCRFD-2.5G Office Delivery Validation

Date: 20 October 2026. Application source tested: `e9b645e`.
This report is a documentation-only follow-up to that source revision.

## Tested Environment

- Ubuntu 22.04.5, x86_64; Ryzen 7 2700; approximately 24 GB host RAM.
- NVIDIA TITAN X Pascal, 12 GB; driver 535.274.02.
- Docker image built using the supplied allowlisted builder.
- Python 3.10.12, NumPy 1.26.4, ONNX Runtime 1.18.0, CUDA 12/cuDNN 8;
  Qdrant 1.15.5. This is not a test on the recipient's RTX 5000.
- Normal models: YOLOX-Tiny CrowdHuman, SCRFD-2.5G and ArcFace R50.
  The fresh model directory contained only those three ONNX files.

## Completed Checks

| Check | Result |
| --- | --- |
| Office tests from a clean source archive on Mac | 132 passed; one optional real-model test skipped |
| Office tests using the pinned Ubuntu container dependencies | 128 passed; five skipped (four Node-dependent UI tests and one optional real-model test) |
| Separate real-model test on the Ubuntu GPU | 1 passed; includes CUDA node profiling |
| Runtime doctor | All three model hashes verified; actual CUDA nodes executed for each model |
| Previous research source suite | 78 passed; separate suite, no change to that entry point |
| Fresh empty runtime and fresh Qdrant volume | Healthy, with no automatic enrollment |
| Single-photo enrollment | One participant accepted through preview/commit |
| Named-folder-equivalent bulk enrollment | Nineteen more participants accepted through preview/commit; zero rejected |
| Variable RTSP input count | Five simultaneous connections received frames: three real viewpoints plus two explicitly repeated inputs |
| Isolated camera stop/start | Stopped one input without changing the other camera generations; restarted input recovered |
| App restart persistence | Twenty enrolled references and five camera records retained; cameras restarted in the stopped state |
| Existing review data | Original gallery left unchanged; no automatic live enrollment enabled |
| Static checks | JavaScript syntax and Git diff whitespace checks passed |

The five-input check ran alongside another three-input review app and source
tests. It establishes functional concurrency, not a five-camera throughput
benchmark. Do not use its timing to quote device capacity.

The unit suite separately covers five registered plus five live reference
retention, duplicate rejection, replacement by quality, registered-only admission,
one-photo manual review, independent recognition, bounded head holding, solid-head
and dotted-face overlays, and the query-quality controls. These are algorithm/API
checks, not evidence of real-world identification accuracy.

## Synchronized Replay

The delivery image passed a 330-second check of the existing three-view
09:16-09:21 synchronized replay using the original 23 active references.
All three feeds stayed running at native 2560 x 1440. Camera generations,
settings and gallery references stayed unchanged across the loop.

The test used strictness 50, minimum face width 70 px, full-frame SCRFD-2.5G,
10 FPS analysis target and 1 Hz recognition. Adaptation was explicitly Off.
The five-input test and other test workloads were stopped before this run.

| View | Mean sampled analysis FPS | Face detections | Quality-eligible faces | Named decisions | Confirmed decisions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Reception | 9.46 | 363 | 0 | 0 | 0 |
| Workspace 1 | 9.49 | 787 | 236 | 29 | 25 |
| Workspace 2 | 9.46 | 220 | 31 | 2 | 0 |

These are repeated observations, not distinct people or verified correct
identifications. Reception's detected faces did not pass the quality/size gates
in this window; the system does not identify every visible person. Workspace 1
also exhibited head-tracked identity carry-forward after face evidence ceased.
No stale recognition results, camera errors or automatic gallery writes were
observed. This is a short replay check, not a production capacity guarantee.

## Limits

No claim of calibrated false-accept rate, liveness, correct identification of
every employee, cross-camera global identity, production security, or all-day
stability follows from these checks. The recipient must validate their own
cameras, GPU variant, model rights, consent, thresholds and expected operating
conditions. Model candidates and confirmed states are not ground-truth labels.
Private photos, video, embeddings, camera URLs and raw reports are not included
in this source delivery. Follow the [runbook](README.md) for reproduction.
