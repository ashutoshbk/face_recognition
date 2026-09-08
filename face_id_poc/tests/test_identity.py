from camex_faceid.identity import IdentityState


CFG = dict(
    match_threshold=0.45,
    match_margin=0.08,
    confirm_samples=3,
    confirm_span_ms=600,
    evidence_window_ms=3000,
    identity_ttl_ms=10000,
)


def test_requires_independent_multiple_samples():
    s = IdentityState(CFG)
    assert not s.observe(0, "a", 0.7, 0.2)
    assert not s.observe(300, "a", 0.7, 0.2)
    assert s.observe(600, "a", 0.7, 0.2)
    assert s.person_id == "a"


def test_ambiguous_never_confirmed():
    s = IdentityState(CFG)
    for ts in range(0, 3000, 300):
        s.observe(ts, "a", 0.8, 0.02)
    assert s.person_id is None


def test_duplicate_evidence_rejected():
    import pytest

    s = IdentityState(CFG)
    s.observe(100, "a", 0.7, 0.2)
    with pytest.raises(ValueError):
        s.observe(100, "a", 0.7, 0.2)


def test_conflicting_evidence_suspends_identity():
    s = IdentityState(CFG)
    for ts in [0, 300, 600]:
        s.observe(ts, "a", 0.7, 0.2)
    s.observe(900, "b", 0.8, 0.3)
    s.observe(1200, "b", 0.8, 0.3)
    assert s.person_id is None


def test_expiration():
    s = IdentityState(CFG)
    for ts in [0, 300, 600]:
        s.observe(ts, "a", 0.7, 0.2)
    s.expire(11000)
    assert s.person_id is None


def test_revoked_samples_cannot_immediately_reconfirm():
    s = IdentityState(CFG)
    for ts in range(0, 2400, 300):
        s.observe(ts, "a", 0.8, 0.2)
    s.observe(2400, "b", 0.8, 0.2)
    s.observe(2700, "b", 0.8, 0.2)
    assert not s.observe(3000, "a", 0.8, 0.2)
    assert s.person_id is None


def test_nonfinite_timestamp():
    import pytest

    s = IdentityState(CFG)
    with pytest.raises(ValueError):
        s.observe(float("nan"), "a", 0.8, 0.2)


def test_weak_view_does_not_replace_confirmed_identity():
    s = IdentityState(CFG)
    for ts in [0, 300, 600]:
        s.observe(ts, "a", 0.7, 0.2)
    s.observe(900, "b", 0.25, 0.01)
    s.observe(1200, "b", 0.2, 0.02)
    assert s.person_id == "a"
    s.expire(11000)
    assert s.person_id is None
