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
| `pending` | No usable appearance embedding is available yet. |

Coordinates are pixel-space `x1, y1, x2, y2` values in the submitted frame. Confidence values are model scores, not calibrated probabilities.
