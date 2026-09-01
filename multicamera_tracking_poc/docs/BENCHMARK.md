# Benchmark

## Packaged Three-Camera Acceptance

The release package was tested on 1 September 2026 using a Titan X Pascal and
three synchronized 1280x720, 10 FPS office clips for 60 seconds.

| Metric | Measured result |
|---|---:|
| Submitted / processed frames | 1,800 / 1,800 |
| Aggregate effective throughput | 28.82 FPS |
| Wall duration | 62.47 s |
| Replaced / stale frames | 0 / 0 |
| Result-queue drops / callback errors | 0 / 0 |
| Peak total GPU memory in telemetry | 645 MiB |
| Mean / peak GPU utilization | 19.01% / 90% |
| Global IDs observed in multiple views | 2 |

The video and event stream were decoded end to end. Visual review included a
person whose camera-local track ID changed while the site-wide global ID stayed
`G10` across the two workspace views. This is functional evidence only. The
footage has no annotated person identities, so it cannot establish IDF1, HOTA,
false-merge rate, false-split rate, or a 95% accuracy claim.

## Packaged 20-Feed Live API Regression

Twenty independent producers submitted 1280x720 frames at 10 FPS for 30
seconds through the public `TrackingService` API. The sources were 20 distinct
clips from three physical camera views at different wall-clock times. This is
an integration and throughput test, not a 20-camera identity-accuracy test.

| Metric | Measured result |
|---|---:|
| Requested / accepted submissions | 6,000 / 6,000 |
| Callback results | 2,384 |
| Aggregate processed throughput | 71.62 FPS |
| Effective throughput per feed | 3.58 FPS |
| Latest-slot replacements | 3,616 |
| Stale timestamps / callback errors | 0 / 0 |
| Mean / p50 / p95 batch latency | 239.83 / 227.44 / 273.25 ms |
| Peak total GPU memory in telemetry | 1,049 MiB |
| Mean / peak GPU utilization | 30.96% / 91% |

All submitted frames reached the bounded input adapter. When inference could
not sustain 200 FPS, newer frames replaced older pending frames rather than
building latency. The callback received every processed result. The reported
1,872 result-queue drops occurred because the callback-only harness did not
also consume the optional polling queue; they are not callback losses.

## Previous Two-Minute Computational Test

The previous two-minute test used 20 logical streams at 1280x720 and a requested 10 FPS per stream.

| Metric | Measured result |
|---|---:|
| Aggregate throughput | 86.27 FPS |
| Effective throughput per stream | 4.31 FPS |
| Peak VRAM | 5,941 MB |
| Mean GPU utilization | 40.72% |
| Mean processing cycle | 231.81 ms |
| p95 processing cycle | 246.84 ms |

That recording set contained 20 distinct clips from three physical views at multiple times. It is valid as a computational load test, not as a 20-physical-camera accuracy test.

The run produced 13 model decisions in which a global ID appeared in more than one synchronized view. No annotated person identities were available, so IDF1, HOTA, false-merge, and false-split accuracy were not measured.

The required 20 x 10 FPS target must be repeated on the intended RTX 3090.
Use the client platform's production decoder during that acceptance run.
TensorRT, asynchronous decode/inference, and profiling of serial CPU tracking
and ReID stages remain optimization options if the 3090 misses the target.
