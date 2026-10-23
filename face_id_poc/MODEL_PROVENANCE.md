# Model Provenance

| File | Provenance | SHA-256 |
| --- | --- | --- |
| `arcface-r50-w600k-112x112.onnx` | ArcFace R50/WebFace600K; byte-identical to inspected buffalo_m recognition weights | `4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43` |
| `scrfd-kps-2.5g-640x640.onnx` | Existing SCRFD-2.5G export with five landmarks, supplied for the original demo | `a2a876b5ad23e66a8b3973112ff137f98414c214b17c31a45955d26e54b58cbf` |
| `yolox-tiny-crowdhuman-416x416.onnx` | Existing custom CrowdHuman head/body checkpoint; not the COCO checkpoint used by MCT | `7880da82285f9cedd92c46c3c8edece6e751ce8558f4be080ea6325d2a19d56e` |

Weights are supplied separately and verified at load time. A similarly named model
cannot be substituted without validating graph conventions, preprocessing, gallery
compatibility and thresholds. Old TensorRT engines are not used or distributed.

SCRFD/alignment Python source is pinned to InsightFace commit
`3e6486942a1be2da0e5b475fac375ea73264bd21`. The adapted tracker retains its license
and provenance under `src/camex_faceid/vendor/bytetrack/`.

Read [business limitations and commercial-use notes](docs/MODELS_AND_LIMITATIONS.md).
