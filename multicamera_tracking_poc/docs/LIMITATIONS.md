# Limitations

- Global identity currently uses full-body appearance plus a configurable transition-time constraint. It does not yet use face recognition, calibrated floor-map position, or per-camera topology.
- Reconciliation can revise an earlier provisional ID. The integrating platform must consume identity-alias events and update historical references.
- IDs are retained in memory and reset when the process restarts.
- Similar clothing, occlusion, poor lighting, and low person resolution can cause false merges or splits.
- Confidence values are matching scores, not calibrated probabilities.
- No identity ground truth exists for the supplied office recordings.
- The packaged Titan X live-feed regression reached 71.62 aggregate FPS, not the 200 FPS required for 20 x 10 FPS; the Dell Pro Max Tower T2 with RTX 5000 Ada Generation 32 GB must be benchmarked.
- The three-camera demo verifies functionality and integration, not production accuracy.
- Camera health, stream reconnection, authentication, and retention policy remain platform responsibilities.
