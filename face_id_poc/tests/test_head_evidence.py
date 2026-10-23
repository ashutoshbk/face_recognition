import pytest

from camex_faceid.head_evidence import update_head_evidence


def observe(proof, timestamp, score=0.8, face=False, movement=10):
    return update_head_evidence(proof, timestamp, score, face, movement, {})


def test_motion_on_low_confidence_furniture_never_admits_a_head():
    proof = {}
    for ts in range(0, 5000, 100):
        assert not observe(proof, ts, score=0.63, movement=40)


def test_repeated_strong_moving_head_is_admitted():
    proof = {}
    assert not observe(proof, 0)
    assert observe(proof, 100)
    assert observe(proof, 200)


def test_confirmed_person_survives_later_low_confidence():
    proof = {}
    for ts in [0, 100, 200]:
        observe(proof, ts)
    assert observe(proof, 300, score=0.2, movement=0)


def test_face_can_verify_a_seated_stationary_head():
    assert observe({}, 0, score=0.3, face=True, movement=0)


def test_head_verification_does_not_assign_a_person_identity():
    proof = {}
    assert observe(proof, 0, face=True)
    assert "person_id" not in proof


def test_single_confidence_spike_is_not_enough():
    proof = {}
    assert not observe(proof, 0, score=0.9)
    for ts in range(100, 2000, 100):
        assert not observe(proof, ts, score=0.4)


def test_old_high_scores_cannot_accumulate_across_long_gaps():
    proof = {}
    for ts in [0, 1000, 2000]:
        assert not observe(proof, ts)


def test_duplicate_frame_cannot_count_twice():
    proof = {}
    observe(proof, 100)
    with pytest.raises(ValueError):
        observe(proof, 100)


def test_confidence_can_arrive_after_motion_has_stopped():
    proof = {}
    assert not observe(proof, 0, score=0.4)
    assert not observe(proof, 100, score=0.4)
    assert not observe(proof, 900, score=0.8, movement=0)
    assert observe(proof, 1000, score=0.8, movement=0)
