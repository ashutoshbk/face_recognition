# Configuration

`configs/demo_3cam.yaml` is the functional example. `configs/live_20cam.example.yaml` is the integration template.

Important runtime fields:

| Field | Purpose |
|---|---|
| `site_id` | Namespace reported in every result. |
| `runtime.target_fps` | Maximum processing cadence per camera. |
| `runtime.model_batch_size` | Static detector batch; use 3 for the demo and 20 for the PoC. |
| `runtime.result_queue_size` | Bounded local result queue. |
| `cameras` | Stable registered camera IDs. |

The 20 live cameras must use one site configuration and one identity gallery. The A-G identity groups used to analyze recordings from unrelated times are intentionally absent.

Model and threshold values reproduce the tested PoC configuration. Tune them only against an annotated validation interval and never against the final test interval.

Do not place RTSP credentials in tracked YAML files. Read secrets from the platform's existing secret store or environment.
