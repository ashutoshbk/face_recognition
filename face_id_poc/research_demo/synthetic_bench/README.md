# Synthetic Gallery Benchmark

Use the parent [reproduction guide](../README.md) for installation, official
DigiFace archive download, model import, quick validation and the full build.
Use [architecture](../ARCHITECTURE.md) for the split and measurement definitions.

This module enrolls one reference per fictional identity, not every image.
Gallery image 0 (fallback 2 or 3) and held-out probe image 1 remain disjoint.
SQLite checkpoints make encoding resumable. Only valid gallery vectors enter
Qdrant. The viewer exposes synthetic probes only, not arbitrary image upload.

`result.json` is written after encoding, Qdrant indexing and benchmarking finish.
Its measured gallery size may be smaller than the identity-folder count because
some images fail detection or alignment. Compare search latency, exact-index
recall and identity rank separately; they are different measurements.

The DigiFace R-UDA and pretrained-weight restrictions apply. Neither synthetic
accuracy nor search latency establishes performance or rights for deployment.
