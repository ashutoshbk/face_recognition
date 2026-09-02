# Configuration

`configs/demo_3cam.yaml` is the functional example. `configs/live_20cam.example.yaml` is the integration template.

Important runtime fields:

| Field | Purpose |
|---|---|
| `site_id` | Namespace reported in every result. |
| `runtime.target_fps` | Maximum processing cadence per camera. |
| `runtime.model_batch_size` | Static detector batch; use 3 for the demo and 20 for the PoC. |
| `runtime.result_queue_size` | Bounded local result queue. |
| `runtime.identity_update_queue_size` | Bounded queue for global-ID alias events. |
| `cameras` | Stable registered camera IDs. |

The 20 live cameras must use one site configuration and one identity gallery. The A-G identity groups used to analyze recordings from unrelated times are intentionally absent.

Model and threshold values reproduce the tested PoC configuration. Tune them only against an annotated validation interval and never against the final test interval.

Important identity fields:

| Field | Purpose |
|---|---|
| `minimum_track_observations` | Number of appearance observations collected before assigning a global ID. |
| `evidence_window_size` | Rolling appearance evidence retained for one local track. |
| `reassignment_grace_seconds` | Period in which a singleton provisional ID can be corrected. |
| `reassignment_threshold` | Minimum similarity for guarded provisional correction. |
| `record_reconciliation_enabled` | Enables established multi-camera identities to absorb earlier fragments. |
| `record_reconciliation_similarity_threshold` | Minimum multi-observation similarity for record reconciliation. |
| `record_reconciliation_margin` | Required separation from the next candidate in both matching directions. |
| `record_reconciliation_max_transition_seconds` | Longest permitted supporting cross-camera transition. |

The transition window is meaningful only when all submitted camera timestamps
share a comparable timebase. Adjust it for the real camera layout using labeled
handoff examples; it is not a universal physical constant.

Do not place RTSP credentials in tracked YAML files. Read secrets from the platform's existing secret store or environment.
