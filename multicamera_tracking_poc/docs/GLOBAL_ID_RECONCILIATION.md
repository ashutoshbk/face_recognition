# Global-ID Reconciliation

## Purpose

A weak first crop can create a provisional identity before a clearer view is
available. The reconciliation layer uses later evidence to correct this type of
false split while protecting against unsafe merges. It operates on arbitrary
runtime IDs; no person or global-ID number is hardcoded.

## Decision Process

The identity manager:

1. Collects a rolling window of appearance embeddings for each local track.
2. Builds a robust prototype after discarding the least consistent samples.
3. Allows recent singleton assignments to be corrected using a stricter match.
4. Allows an identity supported by multiple cameras to absorb an earlier
   single-camera fragment.
5. Requires reciprocal best matching, a configured ambiguity margin, and a
   plausible transition interval.
6. Rejects any merge that conflicts with historical or active tracks in the
   same camera.

When a correction is accepted, all future frame results use the canonical ID
and the service publishes one `IdentityAliasEvent`. The integrating platform
uses that event to canonicalize earlier database, cache, search, and UI records.

## Validated Example

One reviewed handoff followed this path:

| View | Local track | Initial result | Canonical result | Interval |
|---|---|---|---|---:|
| Reception | `L12` | `E-G18` | `E-G19` | 72.1-74.1 s |
| Workspace 1 | `L10`, `L12` | `E-G19` | `E-G19` | 75.9-83.7 s |
| Workspace 2 | `L11` | `E-G19` | `E-G19` | 80.9-91.5 s |

The system discovered `18 -> 19` from runtime evidence. The same run discovered
five other aliases with different IDs, demonstrating that this was not a
case-specific remap.

## Integration Requirements

- Submit timestamps from a comparable clock across all cameras.
- Consume every identity-alias event and apply it idempotently.
- Resolve aliases transitively when querying stored state.
- Tune thresholds and transition limits with labeled client handoffs.
- Retest every camera pair after camera placement or lighting changes.

## Evidence Boundary

The supplied recordings do not contain person-level ground truth. Visual review
and collision invariants support the observed corrections, but they do not
measure overall IDF1, HOTA, false-merge rate, or false-split rate.
