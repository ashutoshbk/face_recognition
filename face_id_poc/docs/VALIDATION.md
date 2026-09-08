# Validation Status

The reference is the five-minute Reception demo with the v0.1.1 chair fix, using
20 supplied enrollment photos. Its three proposed enrolled labels are model
predictions, not client-confirmed identity ground truth.

## Clean-Checkout Handoff Test (2026-09-08)

Source commit `54bbbbf` was transferred as a Git bundle and cloned into a new
directory on the Ubuntu GPU server. The Docker image was built using the supplied
installer. Enrollment used the 20 original photos, not a saved gallery. The launcher
mounted only the supplied inputs/models/config and new output directory, with
networking disabled and no mounted host source code. Subsequent handoff edits
are documentation only; the inference/tracking implementation is unchanged.

| Check | Observed result |
| --- | --- |
| Model-free tests | 113 passed on Mac CPU and 113 in the fresh Linux image |
| Model checksums and active providers | All three passed; CUDA sessions confirmed |
| Enrollment | 20 accepted, 0 rejected |
| Full GPU reproduction | 3,000 frames, 300 seconds, 1080p at 10 FPS |
| Analysis throughput | 20.26 FPS on TITAN X Pascal 12 GB |
| Including rendering/encoding | 9.54 FPS; not a real-time full-render acceptance |
| Sampled peak GPU compute-process memory | 650 MiB; not a guarantee of total device peak |
| Processing latency | p50 25.52 ms, p95 114.27 ms |
| Completed-run rerun | Existing completed inference reused successfully |
| Mac CPU smoke | 100 frames / 10 seconds; 2.36 analysis FPS, 2.09 including rendering |
| Encoded output integrity | All 3,000 video frames decoded successfully |

Against the frozen chair-fixed reference, every head box, track ID, proposed
identity/state, confirmation/revocation decision and event timestamp matched.
There were 77 fragmented head tracks, 14 confirmation events and three proposed
enrolled identities in both runs. Track-state counts and diagnostic counts also
matched. This does not mean 77 different people were present.

The JSON files and MP4 are **not byte-identical**. Processing time changes between
runs. Small floating-point differences were observed: maximum similarity change
0.000532, margin change 0.000759 and face-box coordinate change 0.000245 pixels.
These did not change the tracking/identity decisions. Visual samples included the
reviewed chair case and the start, middle and end of the video.

This is not an RTX 5000 Ada, multi-camera or production-load acceptance test.
The CPU run verifies functionality, not real-time throughput or cross-backend
numerical equivalence.

## Chair-Fix Scope

107 model-free tests passed for the chair-fix archive. Prior chair checks covered
476 old chair-box observations across 13 fragmented tracks; none of those reviewed
chair detections remained. One overlapping observation was a real foreground head.
Some weak head-only detections were also filtered; this is not a zero-regression
or universal false-positive claim.

The identical head-box sequence in the clean reproduction preserves those reviewed
chair-case outcomes. The same footage influenced development, so this is regression
testing rather than an independent accuracy benchmark. Validate on separate site
footage with confirmed identity labels before making accuracy claims.

Model-use authorization is separate from technical validation. See
[the commercial-use notes](MODELS_AND_LIMITATIONS.md).
