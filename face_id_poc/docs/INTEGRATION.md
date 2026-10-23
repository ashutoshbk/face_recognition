# Integration Contract

Use `FaceIDService.process(frame, timestamp_ms)` from `camex_faceid.pipeline`.
Results are JSON-serializable; image arrays stay in the optional evidence/renderer
path, not the JSON response. Enrollment must be completed before creating a service.

```json
{
  "schema_version": "1.0",
  "camera_id": "entrance_01",
  "timestamp_ms": 1200,
  "sequence": 13,
  "processing_ms": 31.2,
  "tracks": [],
  "events": []
}
```

Each track includes `track_id`, `head_bbox` (source-pixel XYXY), `head_score`,
nullable `face_bbox`, `recognition_state`, nullable `person_id` and `display_name`,
nullable `similarity` and `margin`, face-quality measurements when available,
`last_face_confirmed_ms`, and `face_observed_now`.

- `no_usable_face`: no usable recognition evidence in the current evidence window.
- `pending`: insufficient or ambiguous evidence; do not use it as an identity.
- `unknown`: observed evidence did not establish an enrolled identity.
- `identified`: the temporal policy accepted an enrolled label. This remains a
  model decision, not independently verified truth.

Events use `event: identity_confirmed` or `event: identity_revoked`, along with
camera ID, source timestamp, local track ID, enrolled person ID and label. A
revocation can carry a reason. Clear the affected assignment when revoked; an
event-only client must not retain an obsolete name indefinitely. Treat each
`tracks` list as the current observed snapshot; absent overlays must be removed.

Reset downstream local-track state when the camera/service session restarts.
Track IDs are not globally unique. The preview's gallery IDs derive from filename
labels; map them explicitly to canonical employee records. Do not auto-enroll
unknown faces or interpret unknown as a new employee.

Create models, gallery and service once per lifecycle, not per frame. Calls are
serialized and synchronous; the caller owns stream scheduling and bounded buffering.
Multi-camera batching, body ReID and floor-map fusion are separate integrations.

The platform owns consent/authorization, retention/deletion, encryption, role-based
access, audit trails, stream failure handling and human confirmation for consequential
actions. This local library provides no authentication/web server. Do not expose
filesystem CLI arguments as an unauthenticated API.
