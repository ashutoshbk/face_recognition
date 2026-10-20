import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domain import Gallery, TrackEvidence, HeadTrackIdentity, registered_gate, safe_name, unit


class Index:
    fail = False

    def replace(self, refs):
        if self.fail:
            raise RuntimeError("index down")
        self.refs = list(refs)

    def scores(self, vector, references):
        return {r["id"]: float(r["vector"] @ vector) for r in references}


def vector(seed):
    return unit(np.random.default_rng(seed).normal(size=512))


@pytest.fixture
def gallery(tmp_path):
    g = Gallery(tmp_path, Index())
    g.sync()
    return g


def put(gallery, p, pool, seed, score):
    return gallery.put(p, pool, bytes([seed]), vector(seed),
                       {"accepted": True, "score": score, "width": 100})


def test_independent_top_five_and_restart(gallery):
    p = gallery.create_person("Test Person")
    for i in range(1, 8):
        put(gallery, p, "registered", i, i*10)
    for i in range(10, 17):
        put(gallery, p, "live", i, i)
    person = gallery.people()[0]
    assert len(person["registered"]) == len(person["live"]) == 5
    assert sorted(x["score"] for x in person["registered"]) == [30, 40, 50, 60, 70]
    assert not put(gallery, p, "registered", 30, 10)["accepted"]
    g2 = Gallery(gallery.root, Index())
    g2.sync()
    assert len(g2.references()) == 10


def test_duplicate_and_quality_rejection(gallery):
    p = gallery.create_person("Tester")
    assert put(gallery, p, "registered", 1, 70)["accepted"]
    assert not put(gallery, p, "registered", 1, 99)["accepted"]
    assert not gallery.put(p, "live", b"x", vector(2), {"accepted": False})["accepted"]


def test_index_failure_preserves_existing_files_and_rows(gallery):
    p = gallery.create_person("Tester")
    old = put(gallery, p, "registered", 1, 70)
    gallery.index.fail = True
    with pytest.raises(RuntimeError):
        put(gallery, p, "registered", 2, 90)
    assert len(gallery.references()) == 1
    assert gallery.photo_path(old["id"]).read_bytes() == b"\x01"
    assert not gallery.healthy


def test_registered_change_invalidates_live(gallery):
    p = gallery.create_person("Tester")
    put(gallery, p, "registered", 1, 70)
    put(gallery, p, "live", 2, 80)
    assert len(gallery.references()) == 2
    put(gallery, p, "registered", 3, 90)
    assert len(gallery.references()) == 2
    assert gallery.people()[0]["live"][0]["active"] is False


@pytest.mark.parametrize("count,passes,expected", [(1, 1, False), (2, 1, False), (2, 2, True),
    (3, 2, False), (3, 3, True), (4, 2, False), (4, 3, True), (5, 3, False), (5, 4, True)])
def test_registered_consensus(count, passes, expected):
    ranked = [dict(person_id="A", registered=[.9]*passes+[.1]*(count-passes),
                   registered_score=.9, live=[.99], score=.99)]
    assert registered_gate(ranked, "A", .7, .1)[0] is expected


def test_live_photos_cannot_authorize_admission():
    ranked = [dict(person_id="A", registered=[.1, .1], registered_score=.1,
                   live=[.99]*5, score=.99)]
    assert registered_gate(ranked, "A", .7, .1)[0] is False


def test_competitor_margin_required():
    ranked = [dict(person_id="A", registered=[.9, .9], registered_score=.9),
              dict(person_id="B", registered=[.88, .88], registered_score=.88)]
    assert not registered_gate(ranked, "A", .7, .1)[0]


SETTINGS = dict(match_threshold=.45, match_margin=.06, admission_threshold=.7, admission_margin=.1)


def ranking(identity="A"):
    return [dict(person_id=identity, name=identity, score=.9, registered_score=.9,
                 registered=[.9, .9], live=[])]


def test_progressive_vote_duplicate_and_expiry():
    evidence = TrackEvidence()
    assert evidence.add(ranking(), 10, 1, SETTINGS)["state"] == "Tentative"
    assert evidence.add(ranking(), 10.6, 2, SETTINGS)["votes"] == 2
    result = evidence.add(ranking(), 11.2, 3, SETTINGS)
    assert result["state"] == "Confirmed" and result["admission"]
    assert evidence.add(ranking(), 11.8, 3, SETTINGS)["observations"] == 3
    assert evidence.summary(30)["state"] == "Unknown"


def test_conflict_breaks_admission():
    evidence = TrackEvidence()
    for i, identity in enumerate(["A", "A", "B", "A", "A"]):
        result = evidence.add(ranking(identity), 10+i*.6, i, SETTINGS)
    assert result["state"] == "Confirmed"
    assert not result["admission"]


def confirmed_head_identity():
    identity = HeadTrackIdentity()
    for frame, timestamp in enumerate((10., 10.6, 11.2)):
        result = identity.add(ranking(), timestamp, frame, SETTINGS)
    assert result["state"] == "Confirmed"
    return identity


def test_confirmed_name_outlives_vote_window_on_continuous_head():
    identity = confirmed_head_identity()
    result = identity.summary(40)
    assert result["name"] == "A" and result["state"] == "Tracked"
    assert result["identity_basis"] == "head_track"
    assert result["last_face_match_age_seconds"] == 28.8
    assert not result["admission"]


def test_unconfirmed_name_has_only_short_term_persistence():
    identity = HeadTrackIdentity()
    identity.add(ranking(), 10, 1, SETTINGS)
    assert identity.summary(11)["state"] == "Tentative"
    assert identity.summary(19)["name"] is None


def test_same_face_revalidates_carried_name_without_new_admission():
    identity = confirmed_head_identity()
    result = identity.add(ranking(), 40, 10, SETTINGS)
    assert result["name"] == "A" and result["state"] == "Confirmed"
    assert result["identity_basis"] == "face" and result["observations"] == 1
    assert not result["admission"]


def test_new_identity_cannot_inherit_previous_confirmed_votes():
    identity = confirmed_head_identity()
    result = identity.add(ranking("B"), 11.8, 8, SETTINGS)
    assert result["name"] == "B" and result["state"] == "Tentative"
    assert result["votes"] == result["observations"] == 1
    assert identity.confirmed is None


def test_unknown_usable_face_revokes_carried_identity():
    identity = confirmed_head_identity()
    uncertain = ranking()
    uncertain[0]["score"] = .2
    result = identity.add(uncertain, 11.8, 8, SETTINGS)
    assert result["name"] is None and identity.confirmed is None
    assert identity.summary(12)["name"] is None


def test_duplicate_observation_cannot_revoke_identity():
    identity = confirmed_head_identity()
    assert identity.add(ranking("B"), 11.8, 2, SETTINGS)["name"] == "A"


def test_fresh_contradiction_revokes_before_vote_throttle():
    identity = confirmed_head_identity()
    result = identity.add(ranking("B"), 11.4, 8, SETTINGS)
    assert result["name"] == "B" and result["state"] == "Tentative"
    assert identity.confirmed is None and not result["admission"]


def test_throttled_matching_face_does_not_add_vote_or_admit():
    identity = confirmed_head_identity()
    result = identity.add(ranking(), 11.4, 8, SETTINGS)
    assert result["name"] == "A" and result["votes"] == 3
    assert not result["admission"]


def test_unconfirmed_conflict_breaks_strict_admission_even_when_throttled():
    identity = HeadTrackIdentity()
    identity.add(ranking(), 10., 1, SETTINGS)
    identity.add(ranking(), 10.6, 2, SETTINGS)
    identity.add(ranking("B"), 10.8, 3, SETTINGS)
    result = identity.add(ranking(), 11.2, 4, SETTINGS)
    assert result["state"] == "Confirmed" and not result["admission"]
    identity.add(ranking(), 11.8, 5, SETTINGS)
    assert identity.add(ranking(), 12.4, 6, SETTINGS)["admission"]


def test_throttled_weak_reference_match_also_breaks_live_admission():
    identity = HeadTrackIdentity()
    identity.add(ranking(), 10., 1, SETTINGS)
    identity.add(ranking(), 10.6, 2, SETTINGS)
    weak = ranking()
    weak[0].update(registered=[.5, .5], registered_score=.5, score=.5)
    identity.add(weak, 10.8, 3, SETTINGS)
    assert not identity.add(ranking(), 11.2, 4, SETTINGS)["admission"]


def test_continuity_break_clears_both_votes_and_confirmed_name():
    identity = confirmed_head_identity()
    identity.clear()
    assert identity.summary(12)["name"] is None
    assert identity.summary(12)["observations"] == 0


@pytest.mark.parametrize("name", ["../a", "a/b", "a\\b", ".", "", "a\x00b"])
def test_path_names_are_safe(name):
    with pytest.raises(ValueError):
        safe_name(name)


def test_votes_do_not_multiply_by_reference_count():
    evidence = TrackEvidence()
    r = ranking()
    r[0]["live"] = [.9]*5
    r[0]["registered"] = [.9]*5
    result = evidence.add(r, 10, 1, SETTINGS)
    assert result["votes"] == 1 and result["observations"] == 1


def test_stale_anchor_cannot_admit_live(gallery):
    person = gallery.create_person("Anchor Test")
    put(gallery, person, "registered", 1, 70)
    version = gallery.people()[0]["anchor_version"]
    put(gallery, person, "registered", 2, 75)
    with pytest.raises(ValueError, match="changed"):
        gallery.put(person, "live", b"new", vector(3), {"accepted": True, "score": 90},
                    expected_anchor_version=version)
    assert gallery.people()[0]["live"] == []


def test_inactive_live_slots_can_be_replaced(gallery):
    person = gallery.create_person("Stale References")
    put(gallery, person, "registered", 1, 70)
    for seed in range(10, 15):
        put(gallery, person, "live", seed, 90)
    put(gallery, person, "registered", 2, 75)
    assert not any(x["active"] for x in gallery.people()[0]["live"])
    assert put(gallery, person, "live", 20, 60)["accepted"]
    photos = gallery.people()[0]["live"]
    assert len(photos) == 5 and sum(x["active"] for x in photos) == 1


def test_person_deletion_has_no_orphans(gallery):
    person = gallery.create_person("Delete Test")
    identifiers = [put(gallery, person, "registered", seed, 70)["id"] for seed in (1, 2)]
    gallery.delete_person(person)
    assert gallery.people() == [] and gallery.references(False) == []
    with gallery.db() as db:
        assert db.execute("SELECT COUNT(*) FROM refs").fetchone()[0] == 0
    for identifier in identifiers:
        with pytest.raises(ValueError):
            gallery.photo_path(identifier)


def test_competitor_change_invalidates_admission(gallery):
    person = gallery.create_person("Target")
    other = gallery.create_person("Competitor")
    put(gallery, person, "registered", 1, 90)
    revision = gallery.registered_revision()
    own_version = gallery.people()[1]["anchor_version"]
    put(gallery, other, "registered", 2, 90)
    with pytest.raises(ValueError, match="gallery changed"):
        gallery.put(person, "live", b"test", vector(3), {"accepted": True, "score": 90},
                    expected_anchor_version=own_version, expected_registered_revision=revision)


def test_failed_file_delete_is_retryable(gallery, monkeypatch):
    person = gallery.create_person("Cleanup Test")
    ref = put(gallery, person, "registered", 1, 90)
    path = gallery.photo_path(ref["id"])
    original = Path.unlink
    def fail_once(self, *args, **kwargs):
        if self == path:
            raise PermissionError("simulated")
        return original(self, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "unlink", fail_once)
        gallery.delete_person(person)
        assert gallery.pending_cleanup_count() > 0 and path.exists()
    gallery.cleanup_pending()
    assert gallery.pending_cleanup_count() == 0 and not path.exists()
