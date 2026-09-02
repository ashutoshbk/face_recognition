import numpy as np

from camex_mct.global_identity import GlobalIdentityStore


def make_store(retention: float = 1800) -> GlobalIdentityStore:
    return GlobalIdentityStore(
        similarity_threshold=0.72,
        ambiguity_margin=0.035,
        active_timeout_seconds=2.0,
        gallery_retention_seconds=retention,
    )


def test_cross_camera_reappearance_reuses_global_id() -> None:
    store = make_store()
    embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    first_id, _, first_status = store.assign("cam01", 1, embedding, 0.0)
    second_id, similarity, second_status = store.assign("cam02", 8, embedding, 0.1)

    assert first_status == "new"
    assert second_status == "reidentified"
    assert first_id == second_id
    assert similarity > 0.99


def test_two_active_tracks_in_one_camera_cannot_share_global_id() -> None:
    store = make_store()
    embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    store.begin_frame("cam01", {1, 2}, 0.0)
    first_id, _, _ = store.assign("cam01", 1, embedding, 0.0)
    second_id, _, _ = store.assign("cam01", 2, embedding, 0.0)

    assert first_id != second_id


def test_disappeared_track_reconnects_in_same_camera() -> None:
    store = make_store()
    embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    store.begin_frame("cam01", {1}, 0.0)
    first_id, _, _ = store.assign("cam01", 1, embedding, 0.0)
    store.begin_frame("cam01", set(), 0.1)
    store.begin_frame("cam01", {2}, 0.2)
    second_id, _, status = store.assign("cam01", 2, embedding, 0.2)

    assert first_id == second_id
    assert status == "same_camera_reconnected"


def test_expired_gallery_does_not_reuse_global_id() -> None:
    store = make_store(retention=1.0)
    embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    first_id, _, _ = store.assign("cam01", 1, embedding, 0.0)
    second_id, _, status = store.assign("cam02", 1, embedding, 2.0)

    assert second_id != first_id
    assert status == "new"


def test_waits_for_multiple_observations_before_assigning_global_id() -> None:
    store = GlobalIdentityStore(
        similarity_threshold=0.72,
        ambiguity_margin=0.035,
        active_timeout_seconds=2.0,
        gallery_retention_seconds=1800,
        minimum_track_observations=3,
    )
    embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    first_id, _, first_status = store.assign("cam01", 1, embedding, 0.0)
    second_id, _, second_status = store.assign("cam01", 1, embedding, 0.1)
    third_id, _, third_status = store.assign("cam01", 1, embedding, 0.2)

    assert first_id is None
    assert second_id is None
    assert first_status == second_status == "pending"
    assert third_id == 1
    assert third_status == "new"


def test_long_term_gallery_recovers_an_early_appearance() -> None:
    store = make_store()
    first_appearance = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    changed_appearance = np.asarray([0.6, 0.8, 0.0], dtype=np.float32)

    first_id, _, _ = store.assign("cam01", 1, first_appearance, 0.0)
    for index in range(1, 51):
        blend = index / 50.0
        current = np.asarray([1.0 - 0.4 * blend, 0.8 * blend, 0.0], dtype=np.float32)
        store.assign("cam01", 1, current, index * 0.1)
    store.assign("cam01", 1, changed_appearance, 6.0)

    store.begin_frame("cam01", {2}, 75.0)
    recovered_id, similarity, status = store.assign("cam01", 2, first_appearance, 75.0)

    assert recovered_id == first_id
    assert status == "same_camera_reconnected"
    assert similarity > 0.85


def test_singleton_provisional_id_is_corrected_by_later_evidence() -> None:
    store = GlobalIdentityStore(
        similarity_threshold=0.72,
        ambiguity_margin=0.035,
        active_timeout_seconds=2.0,
        gallery_retention_seconds=1800,
        minimum_track_observations=3,
        evidence_window_size=6,
        reassignment_interval_seconds=0.0,
        reassignment_threshold=0.80,
        reassignment_margin=0.04,
    )
    identity = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    poor_view = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    for index in range(3):
        established_id, _, _ = store.assign("cam01", 1, identity, index * 0.1)
    for index in range(3):
        provisional_id, _, _ = store.assign("cam02", 7, poor_view, 1.0 + index * 0.1)

    assert established_id != provisional_id
    for index in range(8):
        corrected_id, _, corrected_status = store.assign("cam02", 7, identity, 2.0 + index * 0.1)

    assert corrected_id == established_id
    assert corrected_status == "corrected_reidentified"
    assert store.summary()["corrected_assignments"] == 1
    events = store.drain_reconciliation_events()
    assert len(events) == 1
    assert events[0]["source_global_id"] == provisional_id
    assert events[0]["canonical_global_id"] == established_id
    assert events[0]["reason"] == "provisional_reassigned"


def test_established_singleton_is_not_reassigned_after_grace_period() -> None:
    store = GlobalIdentityStore(
        similarity_threshold=0.80,
        ambiguity_margin=0.04,
        active_timeout_seconds=2.0,
        gallery_retention_seconds=1800,
        reassignment_interval_seconds=0.0,
        reassignment_grace_seconds=0.5,
        reassignment_threshold=0.80,
        reassignment_margin=0.04,
    )
    first_view = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    other_view = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    established_id, _, _ = store.assign("cam01", 1, first_view, 0.0)
    other_id, _, _ = store.assign("cam02", 2, other_view, 1.0)
    result_id, _, _ = store.assign("cam01", 1, other_view, 2.0)

    assert established_id != other_id
    assert result_id == established_id
    assert store.summary()["corrected_assignments"] == 0


def test_historically_overlapping_same_camera_tracks_cannot_merge() -> None:
    store = GlobalIdentityStore(
        similarity_threshold=0.72,
        ambiguity_margin=0.035,
        active_timeout_seconds=2.0,
        gallery_retention_seconds=1800,
        reassignment_interval_seconds=0.0,
        reassignment_grace_seconds=10.0,
        reassignment_threshold=0.80,
    )
    first_view = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    second_view = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)

    store.begin_frame("cam01", {1, 2}, 0.0)
    first_id, _, _ = store.assign("cam01", 1, first_view, 0.0)
    second_id, _, _ = store.assign("cam01", 2, second_view, 0.0)
    store.begin_frame("cam01", {1, 2}, 0.5)
    store.assign("cam01", 1, first_view, 0.5)
    store.assign("cam01", 2, second_view, 0.5)
    store.begin_frame("cam01", {2}, 1.0)
    result_id, _, _ = store.assign("cam01", 2, first_view, 1.0)

    assert first_id != second_id
    assert result_id == second_id


def test_revived_tracks_do_not_share_an_active_same_camera_id() -> None:
    store = make_store()
    embedding = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)

    store.begin_frame("cam01", {1}, 0.0)
    global_id, _, _ = store.assign("cam01", 1, embedding, 0.0)
    store.begin_frame("cam01", {2}, 1.0)
    reconnected_id, _, _ = store.assign("cam01", 2, embedding, 1.0)
    assert global_id == reconnected_id

    store.begin_frame("cam01", {1, 2}, 2.0)
    active_assignments = {
        store.local_assignments.get(("cam01", local_id))
        for local_id in (1, 2)
        if store.local_assignments.get(("cam01", local_id)) is not None
    }
    assert active_assignments == {global_id}
    assert sum(("cam01", local_id) in store.local_assignments for local_id in (1, 2)) == 1
    assert store.summary()["active_conflict_splits"] == 1


def reconciliation_store(max_transition_seconds: float = 10.0) -> GlobalIdentityStore:
    return GlobalIdentityStore(
        similarity_threshold=0.80,
        ambiguity_margin=0.035,
        active_timeout_seconds=2.0,
        gallery_retention_seconds=1800,
        record_reconciliation_enabled=True,
        record_reconciliation_interval_seconds=0.0,
        record_reconciliation_similarity_threshold=0.72,
        record_reconciliation_margin=0.04,
        record_reconciliation_max_transition_seconds=max_transition_seconds,
        record_reconciliation_min_observations=1,
    )


def test_multicamera_identity_absorbs_fragment_and_emits_runtime_alias() -> None:
    store = reconciliation_store()
    fragment_view = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    later_first_view = np.asarray([0.65, 0.7599342, 0.0], dtype=np.float32)
    later_second_view = np.asarray([0.74, 0.6726069, 0.0], dtype=np.float32)

    fragment_id, _, _ = store.assign("reception", 4, fragment_view, 0.0)
    established_id, _, _ = store.assign("workspace_1", 8, later_first_view, 2.0)
    continued_id, _, _ = store.assign("workspace_2", 5, later_second_view, 4.0)

    assert fragment_id != established_id
    assert continued_id == established_id
    assert store.resolve_global_id(fragment_id) == established_id
    assert store.local_assignments[("reception", 4)] == established_id

    events = store.drain_reconciliation_events()
    assert len(events) == 1
    assert events[0]["source_global_id"] == fragment_id
    assert events[0]["canonical_global_id"] == established_id
    assert events[0]["source_track"] == ("reception", 4)
    assert store.drain_reconciliation_events() == ()


def test_record_reconciliation_rejects_ambiguous_fragment() -> None:
    store = reconciliation_store()
    store.record_reconciliation_enabled = False
    fragment = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    first_candidate = np.asarray([0.75, 0.6614378, 0.0], dtype=np.float32)
    second_candidate = np.asarray([0.74, -0.6726069, 0.0], dtype=np.float32)

    fragment_id, _, _ = store.assign("cam01", 1, fragment, 0.0)
    first_id, _, _ = store.assign("cam02", 2, first_candidate, 2.0)
    store.assign("cam03", 3, first_candidate, 3.0)
    second_id, _, _ = store.assign("cam04", 4, second_candidate, 4.0)
    store.assign("cam05", 5, second_candidate, 5.0)
    assert first_id != second_id

    store.record_reconciliation_enabled = True
    store._maybe_reconcile_records(6.0)

    assert store.local_assignments[("cam01", 1)] == fragment_id
    assert store.drain_reconciliation_events() == ()


def test_record_reconciliation_rejects_implausible_transition() -> None:
    store = reconciliation_store(max_transition_seconds=5.0)
    fragment_view = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    later_first_view = np.asarray([0.65, 0.7599342, 0.0], dtype=np.float32)
    later_second_view = np.asarray([0.74, 0.6726069, 0.0], dtype=np.float32)

    fragment_id, _, _ = store.assign("cam01", 1, fragment_view, 0.0)
    established_id, _, _ = store.assign("cam02", 2, later_first_view, 20.0)
    store.assign("cam03", 3, later_second_view, 21.0)

    assert fragment_id != established_id
    assert store.local_assignments[("cam01", 1)] == fragment_id
    assert store.drain_reconciliation_events() == ()
