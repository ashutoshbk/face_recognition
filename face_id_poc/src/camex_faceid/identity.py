"""Temporal decisions without camera-, person-, or clip-specific rules."""

from collections import deque
from numbers import Integral


class IdentityState:
    def __init__(self, config):
        self.cfg = config
        self.samples = deque(maxlen=12)
        self.person_id = None
        self.last_confirmed = None
        self.last_input = None
        self.conflicts = 0

    def observe(self, timestamp_ms, candidate, similarity, margin):
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, Integral)
            or timestamp_ms < 0
        ):
            raise ValueError("Timestamp must be a nonnegative integer in milliseconds")
        if self.last_input is not None and timestamp_ms <= self.last_input:
            raise ValueError(
                "Recognition evidence must have unique increasing timestamps"
            )
        self.last_input = timestamp_ms
        valid = (
            candidate is not None
            and similarity >= self.cfg["match_threshold"]
            and margin >= self.cfg["match_margin"]
        )
        chosen = candidate if valid else None
        self.samples.append((timestamp_ms, chosen, similarity, margin))
        while (
            self.samples
            and timestamp_ms - self.samples[0][0] > self.cfg["evidence_window_ms"]
        ):
            self.samples.popleft()
        if self.person_id is not None:
            if chosen == self.person_id:
                self.last_confirmed = timestamp_ms
                self.conflicts = 0
                return False
            # Weak/nonmatching views do not negate an established track identity.
            # A strong contradictory identity starts a new evidence epoch.
            if chosen is not None:
                self.person_id = None
                self.last_confirmed = None
                self.samples.clear()
                self.samples.append((timestamp_ms, chosen, similarity, margin))
            else:
                return False
        if not chosen:
            return False
        matches = [s for s in self.samples if s[1] == chosen]
        enough = len(matches) >= self.cfg["confirm_samples"]
        span = matches[-1][0] - matches[0][0] if matches else 0
        consistent = len(matches) / len(self.samples) >= 0.75
        if enough and consistent and span >= self.cfg["confirm_span_ms"]:
            changed = self.person_id != chosen
            self.person_id = chosen
            self.last_confirmed = timestamp_ms
            self.conflicts = 0
            return changed
        return False

    def expire(self, timestamp_ms):
        while (
            self.samples
            and timestamp_ms - self.samples[0][0] > self.cfg["evidence_window_ms"]
        ):
            self.samples.popleft()
        if (
            self.last_confirmed is not None
            and timestamp_ms - self.last_confirmed > self.cfg["identity_ttl_ms"]
        ):
            self.person_id = None
            self.last_confirmed = None
            self.samples.clear()

    def suspend(self):
        self.person_id = None
        self.last_confirmed = None
        self.samples.clear()
        self.conflicts = 0
