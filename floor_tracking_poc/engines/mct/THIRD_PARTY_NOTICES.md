# Third-Party Notices

This private PoC uses the following public components:

| Component | Source | License | Pinned commit |
|---|---|---|---|
| ByteTrack | https://github.com/FoundationVision/ByteTrack | MIT | `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99` |
| YOLOX | https://github.com/Megvii-BaseDetection/YOLOX | Apache-2.0 | Checkpoint release `0.1.1rc0` |
| BoT-SORT | https://github.com/NirAharon/BoT-SORT | MIT | `251985436d6712aaf682aaaf5f71edb4987224bd` |
| Torchreid / OSNet | https://github.com/KaiyangZhou/deep-person-reid | MIT | `f8cd150fdf77e8d9e1ed143b7f308c2c609ded50` |
| PyTorch | https://pytorch.org/ | See upstream terms | Container version 2.1.0 CUDA 11.8 |
| OpenCV | https://opencv.org/ | Apache 2.0 | Python package 4.8.1.78 |

The tracked `patches/bot-sort-external-reid.patch` makes BoT-SORT's bundled
FastReID adapter optional and removes two unused Matplotlib imports. CAMEX
supplies precomputed OSNet embeddings to BoT-SORT instead; the tracking logic
is otherwise unchanged.

Model checkpoints:

| Model | Original source | SHA-256 |
|---|---|---|
| YOLOX-Tiny | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.pth | `9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0` |
| OSNet-IBN x1.0 MSMT17 | https://huggingface.co/kaiyangzhou/osnet/resolve/main/osnet_ibn_x1_0_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth | `2aca54945043301c82fe1a08bb22b045adb0c210673cb8319f0d4f44778f0566` |

The Hugging Face repository labels the OSNet weights MIT, while the official
MSMT17 dataset release agreement limits the underlying dataset to academic,
non-commercial use. The effect of those dataset terms on the trained checkpoint
has not been legally resolved for this project. Treat that checkpoint as
evaluation-only until Dezzex obtains written commercial clearance or supplies a
commercially cleared compatible replacement. See [MODEL_PROVENANCE.md](MODEL_PROVENANCE.md).

Redistributions must retain applicable upstream license and attribution notices.
