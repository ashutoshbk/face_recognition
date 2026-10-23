import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domain import best_quality_crops, SingleReferenceReview


def crop(number, quality, person="A"):
    return dict(id=str(number), quality_score=quality, timestamp=float(number), digest=str(number), person_id=person)


def test_keep_best_five_not_latest_five():
    chosen = []
    for i, quality in enumerate([95, 92, 89, 86, 83, 50, 45, 99]):
        chosen = best_quality_crops(chosen, crop(i, quality))
    assert [s["quality_score"] for s in chosen] == [99, 95, 92, 89, 86]
    assert {s["id"] for s in chosen} == {"0", "1", "2", "3", "7"}


def test_tied_quality_keeps_earlier_and_duplicate_is_not_another_slot():
    chosen = [crop(i, 80) for i in range(5)]
    assert best_quality_crops(chosen, crop(10, 80)) == chosen
    duplicate = dict(crop(11, 99), digest="0")
    assert best_quality_crops(chosen, duplicate) == chosen


def test_new_identity_cannot_inherit_old_person_crops():
    assert best_quality_crops([crop(0, 99)], crop(1, 70, "B")) == [crop(1, 70, "B")]


QUALITY = dict(accepted=True, score=80, width=100)


def ranking(score=.75, registered=None, rival=.2, revision=1):
    return [dict(person_id="A", registered=[score] if registered is None else registered,
                 registered_score=score, score=.99, live=[.99], anchor_version=1,
                 registered_revision=revision, reference_scores=[dict(id="ref", pool="registered")]),
            dict(person_id="B", registered=[rival], registered_score=rival)]


def add(gate, i, ranked=None, quality=None, now=None, digest=None):
    return gate.observe(ranked or ranking(), "A", quality or QUALITY,
                        10+i if now is None else now, i, str(i) if digest is None else digest)


def test_five_spaced_registered_only_agreements_propose_but_never_auto_admit():
    gate = SingleReferenceReview()
    for i in range(4):
        assert add(gate, i) is None
    result = add(gate, 4)
    assert result["count"] == 5 and result["span_seconds"] == 4
    assert result["reference_id"] == "ref"
    assert "operator" in result["note"]


@pytest.mark.parametrize("quality", [dict(accepted=False, score=90, width=100),
    dict(accepted=True, score=64, width=100), dict(accepted=True, score=90, width=79)])
def test_weak_crops_cannot_propose(quality):
    gate = SingleReferenceReview()
    for i in range(8):
        assert add(gate, i, quality=quality) is None


def test_live_matches_do_not_substitute_for_anchor_similarity():
    gate = SingleReferenceReview()
    for i in range(8):
        assert add(gate, i, ranked=ranking(score=.59)) is None


def test_competing_registered_person_blocks_review_proposal():
    gate = SingleReferenceReview()
    for i in range(8):
        assert add(gate, i, ranked=ranking(score=.75, rival=.65)) is None


def test_duplicate_crops_do_not_become_five_independent_observations():
    gate = SingleReferenceReview()
    for i in range(8):
        assert add(gate, i, digest="same-pixels") is None
    assert len(gate.observations) == 1


def test_gallery_change_resets_review_evidence():
    gate = SingleReferenceReview()
    for i in range(4):
        add(gate, i)
    assert add(gate, 4, ranked=ranking(revision=2)) is None
    assert len(gate.observations) == 1


def test_weak_observation_breaks_streak_and_long_gap_restarts_it():
    gate = SingleReferenceReview()
    for i in range(4):
        add(gate, i)
    assert add(gate, 4, ranked=ranking(score=.4)) is None
    assert len(gate.observations) == 0
    add(gate, 5)
    assert add(gate, 10) is None and len(gate.observations) == 1


def test_multi_reference_enrollment_keeps_existing_consensus_policy():
    gate = SingleReferenceReview()
    assert add(gate, 0, ranked=ranking(registered=[.75, .8])) is None


def test_nine_tenths_cadence_eventually_proposes():
    gate = SingleReferenceReview()
    found = [add(gate, i, now=10+i*.9) for i in range(15)]
    assert any(found)
    assert min(r["span_seconds"] for r in found if r) >= 4
