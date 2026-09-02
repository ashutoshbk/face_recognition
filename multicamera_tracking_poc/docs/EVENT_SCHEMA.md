# Event Schema

Every processed camera frame produces one `FrameResult`.

```json
{
  "schema_version": "1.0",
  "site_id": "client-office",
  "camera_id": "camera_01",
  "timestamp_ms": 1788264000123,
  "processing_latency_ms": 87.41,
  "tracks": [
    {
      "local_id": 14,
      "global_id": 7,
      "bbox_xyxy": [120.0, 80.0, 260.0, 410.0],
      "detection_confidence": 0.91,
      "reid_confidence": 0.84,
      "status": "reidentified"
    }
  ]
}
```

Statuses:

| Status | Meaning |
|---|---|
| `new` | A new global identity was created. |
| `reidentified` | A gallery identity was reused from another camera. |
| `same_camera_reconnected` | A broken local track reconnected in the same camera. |
| `corrected_reidentified` | A provisional local assignment was corrected using stronger cross-camera evidence. |
| `corrected_same_camera` | A provisional local assignment was corrected using stronger same-camera evidence. |
| `record_reconciled` | The identity became canonical through record-level reconciliation. |
| `pending` | The configured minimum appearance evidence is not yet available. |

Coordinates are pixel-space `x1, y1, x2, y2` values in the submitted frame. Confidence values are model scores, not calibrated probabilities.

## Global-ID Alias Event

Reconciliation can occur after the source track has left a camera. It is
therefore published separately from `FrameResult`:

```json
{
  "schema_version": "1.0",
  "event": "global_id_alias",
  "site_id": "client-office",
  "sequence": 6,
  "timestamp_ms": 89000,
  "source_global_id": 18,
  "canonical_global_id": 19,
  "similarity": 0.7202,
  "transition_gap_s": 6.8,
  "source_camera_id": "reception",
  "source_local_id": 12,
  "target_camera_id": "workspace_1",
  "target_local_id": 10,
  "reason": "record_reconciled"
}
```

The numbers above document one observed validation case; they are not
hardcoded. Both IDs and the similarity are generated from runtime observations.
The platform must replace references to `source_global_id` with
`canonical_global_id`, including historical events already stored. An alias can
itself later resolve to another canonical ID, so use
`TrackingService.resolve_global_id()` when reading long-lived state.

Alias reasons:

| Reason | Meaning |
|---|---|
| `provisional_reassigned` | A recent singleton ID was corrected after stronger evidence accumulated. |
| `record_reconciled` | An earlier fragment was absorbed by an identity already supported by multiple cameras. |
