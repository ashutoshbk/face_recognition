# Validation Status - 0.3.1 Integration Preview

**Functional reproduction completed; not production-approved.**

## Completed Checks

| Check | Result |
|---|---|
| CPU tests on Mac | 54 passed |
| Fresh Ubuntu virtual environment | 54 passed |
| Packaged GPU container CPU test suite | 54 passed |
| GPU/model doctor | All declared calibration/tracking imports, CUDA and FFmpeg encoder available |
| Raw-input operator workflow | New site, exact floor PNG, three videos, six fresh calibration jobs, point fitting, lock/save, reopen and MCT execution |
| Calibration approval gate | Missing metric scale/holdouts rejected; Save preview used explicitly |
| Full output | 120.000s, 1,200 frames, 1920x1080, 10 FPS; full decode passed |
| Camera packets | 3600, with 0 stale and 0 invalid |
| Identical completed request | Correctly reused the same completed job through the UI |
| Source separation | Only current site inputs/calibration and declared model assets mounted; no old observations/expected IDs used as runtime input |

The clean install used a new environment and the explicit companion model asset set. Runtime containers had networking disabled. The source revision adds no camera-name, person-ID or chair-location exceptions.

## Measured Performance

Hardware: NVIDIA Titan X Pascal, 12 GB, Ubuntu 22.04. The client RTX 5000 Ada target was not tested here.

- Fresh detection/SCT/ReID/pose: 3600 camera frames in 125.546s, **28.67 aggregate FPS**.
- Inference batch latency: p50 100.658ms; p95 115.020ms.
- Framework peak GPU allocation: 2.240 GiB; peak reservation: 2.264 GiB.
- Floor/render pass: 94.519s. Complete job including startup: 238.885s.

The rendered video's 10 FPS is playback cadence. It does not establish 20-camera live processing at 10 FPS per camera.

## Sparse Independent Review

| Sample set | Detected / labeled | Published map positions | Same-person pair matches / pairs | False merged distinct-person pairs |
|---|---:|---:|---:|---:|
| Primary samples | 24/34 | 19 | 34/42 | 0 |
| Additional times | 22/28 | 19 | 56/61 | 0 |

These are sparse office samples, not full-clip precision/recall, IDF1, surveyed localization or a blind new-site test. Pair scores resolve final aliases and depend on which detections were matched. Zero sampled false merges or chair hits does not certify every frame. Operator point selections and new calibration affect floor association; literal ID numbers and byte-identical images are not acceptance criteria.

## Open Gates

- Quality parity with the original visual reference is not established. Some small/occluded people and late handoffs remain difficult.
- The first seated-person overlap joins at 7.1s versus 6.9s in the frozen reference; the predefined +0.1s timing gate remains unpassed.
- Metric scale and independent calibration holdouts were not supplied for this example. Draft is not surveyed localization.
- A different real environment's synchronized videos and matching drawing were unavailable. Synthetic tests/camera renaming are not a substitute.
- Target RTX 5000 Ada live load, bounded-latency/drop behavior and a production soak remain separate tests.
- Model provenance and intended-use clearance are separate from engineering validation.

This draft handoff is for integration and investigation. Unit tests, successful installation and an attractive video must not be presented as an unsupported accuracy percentage.
