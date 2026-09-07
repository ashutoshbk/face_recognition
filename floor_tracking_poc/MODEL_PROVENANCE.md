# Model and Dependency Provenance

This complete pipeline contains first-party source plus explicitly identified third-party calibration/tracking engines. Model binaries are supplied in the private installation kit under `models/`, with checksums in `models_manifest.json`; they are excluded from the source-only Git archive.

## Models

| Role | Model | Provenance |
|---|---|---|
| Intrinsics, distortion, gravity | GeoCalib distorted checkpoint | GeoCalib commit `97b8968e7798a66bf04fcf791fb535624241bda7`; Apache-2.0 source |
| Local reconstruction | VGGT-1B-Commercial | VGGT commit `a288dd0f14786c93483e45524328726ab7b1b4ce`; commercial checkpoint commit `cc276933efe26fd26af85a0951061038fa5a1952` |
| Person detection | YOLOX-Tiny | YOLOX release 0.1.1rc0; Apache-2.0 source |
| Appearance ReID | OSNet-IBN x1.0, MSMT17-trained | Torchreid commit recorded in `engines/mct/third_party.lock` |
| Human pose | YOLO11n-pose | Ultralytics 8.4.137; installed metadata declares AGPL-3.0 |

The supplied binary hashes are the exact assets used for the engineering evaluation. The kit does not grant new rights to use or redistribute them. VGGT's applicable agreement/AUP is included under `engines/calibration/licenses/`. The previous organizational review requirement for the exact Police HQ/correctional scope remains unresolved by software packaging alone.

The OSNet checkpoint's model metadata and MSMT17 training-data terms require the commercial-use review already recorded in `engines/mct/MODEL_PROVENANCE.md`. Treat the evaluation checkpoint accordingly until the organization has clearance or a compatible approved replacement.

The optional/proprietary integration route for Ultralytics also needs to be established by the organization. The package does not silently relicense the pose engine. Preserve third-party notices when distributing the private installation kit or its container.

## Reproducible Engine

The Docker base is pinned to the digest in `gpu/Dockerfile`. Calibration source is included in `engines/calibration/vendor/`; tracker source is fetched at the immutable commits in `engines/mct/third_party.lock` during installation. The BoT-SORT external-ReID compatibility patch is included and applied during the build.

The image uses a single tested PyTorch 2.3.1 / CUDA 12.1 environment for the neural stages. NumPy, OpenCV, SciPy, pose and other primary dependencies are pinned in `gpu/requirements.txt` and the Dockerfile. The host UI uses its own virtual environment. FFmpeg is supplied inside the image and its libx264 encoder is checked by the doctor command.

No client video, floor drawing, face image, embedding, reference tracking result or calibration fixture is part of this source package or the model assets. The private installation kit contains source and model assets only; optional example footage is distributed separately as raw input.

## First-Party Scope

First-party code is a private CAMEX/Dezzex consultancy deliverable under the applicable agreement. No public open-source license is granted for those materials here. Third-party components retain their own licenses. Legal/model clearance and engineering workflow acceptance are separate decisions.
