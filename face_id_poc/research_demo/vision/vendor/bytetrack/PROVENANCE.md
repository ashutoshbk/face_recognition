# ByteTrack Core Provenance

Repository: https://github.com/FoundationVision/ByteTrack

Pinned commit: `d1bf0191adff59bc8fcfeaa0b33d3d1642552a99`

Source tree: https://github.com/FoundationVision/ByteTrack/tree/d1bf0191adff59bc8fcfeaa0b33d3d1642552a99

License: MIT, Copyright (c) 2021 Yifu Zhang. The adjacent `LICENSE` is copied
verbatim from the repository root at this commit. Keep it with redistributed
copies or substantial portions of this code.

## Extracted Files

Only these upstream files were extracted, not the YOLOX package, detector,
training code, models, demo renderer, or evaluation dependencies:

| Local file | Upstream path | Original SHA-256 |
| --- | --- | --- |
| `basetrack.py` | `yolox/tracker/basetrack.py` | `c938afcf7c516bba4a77155c2499fcae6f926df7a2faa6f73a4be1b073f9bb2d` |
| `byte_tracker.py` | `yolox/tracker/byte_tracker.py` | `7cecdcdd7998103969a4ba1772f4c9fb5560fd5eef05ca03e0d2df28346ca50b` |
| `kalman_filter.py` | `yolox/tracker/kalman_filter.py` | `9842d0dd9957c97e312d42e5789441add55082066d87cc051ac768143a46eb39` |
| `matching.py` | `yolox/tracker/matching.py` | `2f54bf0c183f577087391901562ae2bcd3c01633a3e077bd5d4168fafbd899a6` |
| `LICENSE` | `LICENSE` | `e9d83751678cbaba6dfba902ea39215c423d93425e58f8450d6001f7261696f9` |

The hashes identify original upstream bytes, not the adapted local files.
`__init__.py`, this record, `camex_faceid/tracking.py`, and `tests/test_tracking.py`
are local additions. No claim of an unmodified upstream distribution is made.

## Core Adaptations

- Imports into `yolox.tracker` now use package-relative imports.
- Replaced `np.float` with builtin `float` for NumPy 1.26 compatibility.
- Removed unused `deque`, `os`, `os.path`, `copy`, `torch`, and
  `torch.nn.functional` imports from `byte_tracker.py`. Removed the alternative
  tensor/Nx6 detector-output branch; this extraction accepts numeric Nx5 arrays.
  Bboxes are copied before upstream scale conversion, avoiding input mutation.
- `matching.py` retains only four upstream functions called by this tracker:
  `linear_assignment`, `ious`, `iou_distance`, and `fuse_score`. Removed unused
  `merge_matches`, `_indices_to_matches`, `v_iou_distance`,
  `embedding_distance`, `gate_cost_matrix`, `fuse_motion`, and `fuse_iou`, plus
  their now-unused SciPy sparse/spatial and Kalman imports. Also removed unused
  OpenCV and `time` imports. The SciPy dependency remains needed by the Kalman
  filter. Empty assignment results consistently have integer shape `(0, 2)`.
- Kept `lap.lapjv(..., extend_cost=True, cost_limit=...)` and `cython_bbox` IoU
  rather than substituting a different assignment objective or IoU convention.
  In particular, upstream inclusive-pixel (`+1`) IoU behavior is preserved.
- Removed `BaseTrack`'s global ID counter and unused appearance/history/location
  fields. `BYTETracker` owns an `itertools.count(1)` allocator, optionally
  supplied by the wrapper. `STrack.activate` receives an explicit ID;
  `re_activate` always retains its ID, matching both original call sites that
  used `new_id=False`. No class-level mutable identity state remains.
- High-score selection is inclusive (`>= track_thresh`), so a detection exactly
  at the configured threshold is not discarded from both association stages.
- Lost tracks expire before association, and current-update removals are
  removed from the lost pool immediately. This prevents the original
  post-association expiry order from reviving an already-expired ID. The removed
  list holds only the current update's removals rather than growing forever.
- Output is explicitly restricted to activated `Tracked` objects whose last
  observation is the current frame. No predicted-only/lost objects are exposed.
- `kalman_filter.py` is unchanged except for a final newline. Its state model,
  matrices, uncertainty weights, and numerical update are preserved. The
  two-stage score/IoU association, confirmation behavior, score fusion, and
  active/lost duplicate suppression otherwise remain upstream implementations,
  apart from the bounded head-geometry extension below.

## Bounded Head Geometry Extension

The wrapper enables this generic extension by default; pass the keyword-only
`geometry_fallback=False` for the preceding IoU-only association behavior.
Bare `BYTETracker` instances remain IoU-only unless their args enable it.
No camera coordinates, person labels, appearance features, or identities are
used. Kalman prediction/correction and exclusive LAP assignment are unchanged.

- Store each track's last observed TLWH on every match/reactivation. Use the
  Kalman-predicted center for association, but last-observed width/height for
  gates. This prevents predicted scale drift from expanding the permitted
  size ratio during detector/source box changes.
- Let `d = norm((predicted_center - detected_center) / min(observed_WH,
  detected_WH))` and `r = max(observed_WH / detected_WH, detected_WH /
  observed_WH)` over both axes. All head-mode associations require positive
  dimensions and `r <= 2.5`; even high IoU cannot bypass this size gate.
- Only when `d <= 0.35`, allow the alternative raw cost
  `0.80 * d / 0.35 + 0.10 * log(r) / log(2.5) + 0.10 * (1 - IoU)`.
  Use the minimum of this and the original IoU cost. Outside the center gate,
  no geometry bonus is added, but normal IoU association is still permitted.
  The gate uses the smaller width and height, not the larger box or diagonal.
- Apply the existing detection-score fusion and stage thresholds afterward.
  Use this cost for high-score/lost recovery, low-score tracked association,
  and tentative-track confirmation. Low scores still cannot create tracks or
  recover lost tracks; source-time expiry and the lost buffer are unchanged.
- Among viable candidate edges, check each row and column for competing
  normalized-center distances within `0.10` of the nearest candidate. Veto all
  affected rows/columns, including IoU-only candidates, rather than choosing
  an arbitrary identity from a geometric near-tie. Carry these vetoes through
  subsequent association stages and suppress births from ambiguous detections.
  Unrelated unambiguous observations can still be returned.
- Expose `association_ambiguous: bool` and `ambiguous_track_ids: set[int]` on
  `HeadTracker`. The set includes affected existing allocated local IDs even
  when they are missing from the observed output, enabling immediate live
  identity suspension by the caller. It is returned as a defensive copy.
  Diagnostics describe the last valid update, clear on the next unambiguous
  update (including empty input) or lifecycle reset, and are unchanged by
  invalid input. They do not themselves modify any recognition/pipeline state.

Limitations: these are bounded geometry heuristics, not appearance-based
identity. Complete overlap/crossing, two indistinguishable nearby centers,
duplicate first-frame detections, movement outside viable gates, source gaps,
and losses beyond the existing buffer can still fragment or swap tracks.
Conservative ambiguity deferral may reduce recall; `False`/an empty set does
not prove that a person's identity is correct. Published Kalman-filtered
boxes do not expose the raw detector/fallback inputs or rejected candidates,
so they cannot establish the cause of every real-clip fragment.

## Public Wrapper API

```python
from camex_faceid.tracking import HeadTracker

tracker = HeadTracker(
    fps=10,
    threshold=0.4,
    buffer_seconds=1.0,
    match_threshold=0.8,
    geometry_fallback=True,
)
tracks = tracker.update(detections, frame_shape, timestamp_ms)
ambiguous_ids = tracker.ambiguous_track_ids  # May be nonempty when tracks == [].
```

- `detections`: numeric Nx5 array-like, source-pixel `[x1, y1, x2, y2, score]`.
  Scores must be finite and in `[0, 1]`. `None`, `[]`, and `(0, 5)` arrays are
  empty observations and still advance tracking. Input is never modified.
- `frame_shape`: positive integer `(H, W, ...)`. The wrapper passes identical
  image/input dimensions to the core, so no model-input scaling is applied.
  Boxes are clipped to the source frame and zero-area/inverted boxes discarded.
  There is no person aspect-ratio or minimum-area filter.
- `timestamp_ms`: integer source timestamp with any origin, strictly increasing
  for every call, including empty observations. Invalid timestamp, shape,
  nonfinite input, or out-of-range score raises `ValueError` before changing
  session state. There is no wall-clock dependency.
- Return: `list[dict]` sorted by `track_id`, with exactly
  `{"track_id": int, "bbox": [float, float, float, float], "score": float}`.
  Bboxes are current Kalman-corrected source-pixel XYXY clipped to the frame;
  scores come from current observations. Only confirmed, currently observed
  active tracks are returned. No identities, features, or model inference are
  included.
- `threshold` is inclusive for high-score association and new-track creation.
  The wrapper overrides upstream's additional `+0.1` birth margin so its single
  threshold has a direct meaning. Scores strictly above `0.1` but below
  `threshold` can update currently tracked heads in the second association,
  but do not create new tracks or recover lost tracks.
- Upstream confirmation is preserved: tracks created on the first update are
  immediately visible; later arrivals need a second matched observation.
  Empty first updates count as frames. Tentative tracks can consume IDs even
  when never returned.
- `fps` is the nominal update rate. The lost-frame buffer is
  `floor(fps * buffer_seconds)`, not the camera's full capture rate. Source-time
  age is also capped at `buffer_seconds`, checked before association, so skipped
  inputs cannot revive stale tracks. An age exactly at the limit is allowed.
  Motion prediction remains one nominal step per update; timestamps are not
  used as variable Kalman timesteps.
- A gap strictly greater than 2000 ms between valid source updates starts a
  fresh lifecycle, even when `buffer_seconds` is larger. The same allocator
  survives this reset, so no visible ID is reused within the wrapper session.
  Each new wrapper starts its own counter at 1. No process-global ID counter
  is read, reset, or modified. Use one instance per source, with serial updates.

## Minimal Dependencies And Verification

Runtime pip packages: **numpy, scipy, lap, cython_bbox**. No PyTorch, YOLOX,
OpenCV, model weights, or detector packages are required by this extraction.
Tests use standard-library `unittest` and are also pytest-discoverable; pytest
is optional. These are tracker-only requirements, not changes to application
dependency files or host environments.

`cython_bbox==0.1.5` also declares `Cython` as a transitive pip dependency.
When no compatible wheel is available, its source build needs Cython, NumPy
headers, and a working C compiler; those are not vendored into this package.

Verified combination: `numpy==1.26.4`, `scipy==1.14.1`, `lap==0.5.13`,
`cython_bbox==0.1.5`. Also verified with NumPy 2.2.6 / SciPy 1.17.1. All 43
model-free tests pass, including continuity, dropout/low-score association,
exclusive assignment, session isolation, source timestamp validation, source
gaps, expiry, input immutability, and source-pixel/aspect-ratio handling.
Geometry regressions include a five-minute stationary sequence alternating
40x50/96x110 boxes at one center, adjacent heads with scale changes and dropout,
low-confidence/confirmation stages, per-axis gates, resolution scaling,
collision deferral, and per-update ambiguity-ID isolation/reset. The synthetic
five-minute sequence produced 300 visible IDs with the preceding IoU-only
tracker and one with the geometry extension; this is not a real-clip result.

Read-only aggregate inspection of the supplied first-run published trajectories
found 117 distinct visible local IDs over 3000 frames, with a median published
track span of 2.9 seconds. The actual revised-clip fragment count must be
measured by rerunning the full pipeline on raw observations. The original
rendered artifact and output log were not modified.

From the package directory, in an environment with those dependencies:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -B -m unittest discover -s tests -p test_tracking.py -v
```
