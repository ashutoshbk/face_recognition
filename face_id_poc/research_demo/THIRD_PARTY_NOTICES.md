# Research assets and rights

This module is an isolated local research demonstrator, not a production camera
identification service. It accepts one explicitly consented test profile and a
separate fictional gallery. Do not use it for unattended public-camera identity
search or import a real-person gallery.

- DigiFace1M: Microsoft R-UDA, non-commercial research only. The restriction also
  covers use of the data or results to improve commercial products/services.
  Review `assets/DIGIFACE_LICENSE.txt` and https://github.com/microsoft/DigiFace1M.
- InsightFace source: MIT; licence retained in `vision/vendor/INSIGHTFACE_LICENSE.txt`.
  Pretrained SCRFD and ArcFace weights have separate research/non-commercial
  restrictions. Source-code licensing does not grant commercial weight rights.
- The supplied YOLOX-Tiny CrowdHuman checkpoint is the existing custom head/body
  model, not the COCO detector. Commercial rights/provenance still require clearance.
  The three exact hashes are in `vision/networks.py` and `assets/model_bundle_manifest.json`.
- ByteTrack attribution and licence are retained under `vision/vendor/bytetrack`.
- Lucide attribution and licence are in `static/LUCIDE_LICENSE.txt`.
- Qdrant is downloaded from its official GitHub release with a pinned checksum;
  its Apache 2.0 licence is fetched beside the local binary.
- ONNX Runtime, OpenCV, NumPy, SciPy, scikit-image, FastAPI, lap and other dependencies
  retain their respective licences.

The code can be shared within this private repository. Model weights, data,
enrollment records, credentials and generated images are not committed. The
model-only companion ZIP contains the same three separately supplied evaluation
weights; its availability is not commercial deployment clearance.

No liveness, production authentication, formal FAR/TAR calibration, encryption
at rest, employee-system integration or security acceptance is claimed here.
