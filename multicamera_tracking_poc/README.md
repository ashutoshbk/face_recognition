# CAMEX Multi-Camera Tracking PoC

Minimal integration package for person detection, single-camera tracking (SCT), and site-wide appearance ReID. The client platform keeps ownership of live stream decoding and submits BGR frames to this package.

## What It Does

- Accepts frames from 1-20 registered cameras.
- Uses one batched YOLOX-Tiny detector.
- Maintains an independent BoT-SORT state and local IDs per camera.
- Uses OSNet-IBN appearance features for local association and global ReID.
- Builds identity decisions from multiple observations rather than a single crop.
- Emits global-ID alias events when stronger cross-camera evidence safely corrects an earlier ID.
- Retains an in-memory global identity gallery for 30 minutes by default.
- Emits versioned Python results that serialize directly to JSON.

## Quick Start

Requirements: Linux, or Windows 11 Pro through WSL2, with a compatible NVIDIA
driver, Docker GPU support, Git, and internet access for the first model setup.

```bash
make setup
make test
make demo DATA_DIR=/private/path/to/synchronized_three_camera_clips
```

The private demo directory must contain:

```text
reception.mp4
workspace_1.mp4
workspace_2.mp4
```

Generated output is written under `outputs/demo_3cam/`.

## Platform Integration

```python
from camex_mct import TrackingService

service = TrackingService.from_yaml(
    "configs/live_20cam.example.yaml",
    on_result=platform.handle_tracking_result,
    on_identity_update=platform.handle_identity_alias,
)
service.start()

service.submit_frame("camera_01", bgr_frame, timestamp_ms)

service.close()
```

`submit_frame()` is nonblocking. Each camera has one latest-frame slot; replacing an unprocessed frame avoids latency growth. Cameras may submit independently, and a missing camera does not block the rest of the batch.

An identity-alias event is generated dynamically when a provisional ID is
reconciled. The platform should update cached and stored references from the
source ID to the canonical ID. No identity numbers are hardcoded.

See [Integration](docs/INTEGRATION.md), [Event Schema](docs/EVENT_SCHEMA.md),
[Configuration](docs/CONFIGURATION.md), [Benchmark](docs/BENCHMARK.md),
[Global-ID Reconciliation](docs/GLOBAL_ID_RECONCILIATION.md),
[Target Hardware](docs/TARGET_HARDWARE.md), and
[Limitations](docs/LIMITATIONS.md).

## Acceptance Status

The packaged three-camera path completed a 60-second GPU run at 3 x 10 FPS
with no submitted-frame replacement, stale timestamps, result-queue drops, or
callback errors. A separate 20-feed integration regression confirmed bounded
latest-frame behavior, but the Titan X Pascal reached 71.62 aggregate FPS, not
the required 200 FPS. The 20 x 10 FPS target therefore remains subject to an
acceptance test on the Dell Pro Max Tower T2 with its NVIDIA RTX 5000 Ada
Generation 32 GB GPU and, if needed, TensorRT pipeline optimization. These tests
had no identity ground truth and are not ReID accuracy measurements. See
[Target Hardware](docs/TARGET_HARDWARE.md).

## Data and Models

Client footage, generated events, model files, and credentials are excluded from Git. `make models` downloads the original checkpoints, verifies SHA-256 checksums, and exports local batch-3 and batch-20 TorchScript models.

First-party use is governed by the applicable consultancy agreement; no public
open-source license is granted for this package. Third-party components and
model assets retain their own terms. Review [License and Usage](LICENSE_AND_USAGE.md),
[Model Provenance](MODEL_PROVENANCE.md), and
[Third-Party Notices](THIRD_PARTY_NOTICES.md) before any client demonstration,
redistribution, or production deployment.
