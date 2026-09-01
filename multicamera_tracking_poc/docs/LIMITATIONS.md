# Limitations

- Global identity currently uses full-body appearance only. It does not use face recognition, floor-map position, camera topology, or transition-time constraints.
- IDs are retained in memory and reset when the process restarts.
- Similar clothing, occlusion, poor lighting, and low person resolution can cause false merges or splits.
- Confidence values are matching scores, not calibrated probabilities.
- No identity ground truth exists for the supplied office recordings.
- The packaged Titan X live-feed regression reached 71.62 aggregate FPS, not the 200 FPS required for 20 x 10 FPS; the RTX 3090 must be benchmarked.
- The three-camera demo verifies functionality and integration, not production accuracy.
- Camera health, stream reconnection, authentication, and retention policy remain platform responsibilities.
