# Model Provenance

## YOLOX-Tiny Detector

- Source: Megvii YOLOX release `0.1.1rc0`
- Project license: Apache-2.0
- Integrity: the setup script verifies the pinned SHA-256 digest
- Status: retain the Apache-2.0 license and applicable attribution when
  redistributing an image or artifact containing the model or implementation

## OSNet-IBN Person Re-Identification

- Source: `kaiyangzhou/osnet` on Hugging Face
- Architecture source: `deep-person-reid`, pinned in `third_party.lock`
- Training dataset identified by the filename: MSMT17
- Integrity: the setup script verifies the pinned SHA-256 digest
- Status: evaluation-only pending commercial clearance

The Hugging Face model repository labels its files MIT. The official MSMT17
release agreement, however, limits the dataset to academic, non-commercial use.
This project has not established whether those dataset terms affect commercial
use of the trained checkpoint. Repository visibility and private distribution
do not resolve that question.

Before client-facing or production use, Dezzex must either:

1. obtain written commercial clearance covering this checkpoint; or
2. configure a compatible checkpoint whose model and training-data provenance
   permit the intended commercial use.

The checkpoint path is configurable through `reid.model`; replacing it does not
require changing the service API.

## Excluded Assets

No model binaries, client footage, face images, embeddings, generated events, or
credentials are committed to this PoC directory.
