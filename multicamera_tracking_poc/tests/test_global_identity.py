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
