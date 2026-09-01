# Integration

## Boundary

The platform owns RTSP connections, reconnection, decoding, camera credentials, and source health. This package owns GPU batching, person detection, one SCT state per camera, appearance embeddings, and site-wide global-ID assignment.

No camera URLs or credentials are required by the core library.

## Lifecycle

```python
from camex_mct import TrackingService


def handle_result(result):
    platform.publish_tracking(result.to_dict())


service = TrackingService.from_yaml(
    "configs/live_20cam.example.yaml",
    on_result=handle_result,
)
service.start()

# Call from the platform's existing decoder callback.
accepted = service.submit_frame(camera_id, bgr_frame, timestamp_ms)

service.close()
```

Frames must be contiguous or convertible BGR `uint8` arrays with shape `H x W x 3`. Camera IDs must be registered in the configuration and remain stable across restarts. Timestamps are integer milliseconds and must increase independently for each camera.

The service does not copy a contiguous input frame. The caller must not mutate or reuse its buffer until the next frame for that camera is submitted.

## Backpressure

Each camera owns one pending frame slot. If a newer frame arrives before inference consumes the previous frame, the newer frame replaces it. This maintains current results instead of accumulating latency. `submit_frame()` returns `False` for a duplicate or stale timestamp.

The result callback runs on the tracking worker thread and should enqueue work rather than perform slow I/O. Results are also available through `poll_result(timeout=...)`.

The callback and polling queue are independent delivery paths. In a
callback-only integration, the unconsumed bounded polling queue will eventually
overwrite its oldest entries and increment `result_queue_drops`; this does not
mean callback results were dropped. Consume `poll_result()` instead of using a
callback when polling is the platform's chosen delivery path.

## Identity Semantics

`local_id` belongs to one camera. `global_id` belongs to the configured site and may be reused after a person disappears and reappears in the same or another camera. Two simultaneously active tracks in one camera cannot share a global ID.

The gallery is in memory. A process restart resets all global IDs. Appearance matching is probabilistic and must be validated on labeled client footage before accuracy claims.

## Deployment Shape

For the PoC, run the service in the same Python process as the platform or in a dedicated GPU worker. If process isolation is required later, wrap the unchanged `FrameResult` schema with the platform's existing transport rather than sending raw images as JSON.
