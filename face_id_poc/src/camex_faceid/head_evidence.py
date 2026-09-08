"""Admission policy for unconfirmed head tracks, independent of scene geometry."""

from collections import deque


def update_head_evidence(proof, timestamp_ms, score, has_face, movement, config):
    previous = proof.get("last_evidence_ms")
    if previous is not None and timestamp_ms <= previous:
        raise ValueError("Head evidence requires increasing source timestamps")
    proof["last_evidence_ms"] = timestamp_ms
    window_ms = config.get("head_confirmation_window_ms", 700)
    samples = proof.setdefault("scores", deque(maxlen=32))
    samples.append((timestamp_ms, float(score)))
    while samples and timestamp_ms - samples[0][0] > window_ms:
        samples.popleft()

    proof["motion"] = (
        proof.get("motion", 0) + 1
        if movement >= config.get("head_motion_threshold", 5.0)
        else 0
    )
    if proof["motion"] >= 2:
        proof["motion_seen"] = True
    strong_samples = sum(
        value >= config.get("head_confirmation_threshold", 0.65) for _, value in samples
    )
    # Motion alone may come from lighting, reflections or somebody behind furniture.
    # Keep low-score observations for already verified tracks, but not new births.
    proof["verified"] = bool(
        proof.get("verified", False)
        or has_face
        or (
            proof.get("motion_seen", False)
            and strong_samples >= config.get("head_confirmation_samples", 2)
        )
    )
    return proof["verified"]
