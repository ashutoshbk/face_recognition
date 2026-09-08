from types import SimpleNamespace

import numpy as np
import pytest

from camex_faceid import pipeline


def _track(track_id, box):
    return {"track_id": track_id, "bbox": list(box), "score": 0.95}


def _face(box, person_id="a", similarity=0.9):
    box = np.asarray(box, float)
    landmarks = box[:2] + (box[2:] - box[:2]) * np.array(
        [[0.3, 0.3], [0.7, 0.3], [0.5, 0.5], [0.35, 0.75], [0.65, 0.75]]
    )
    return {
        "bbox": box,
        "landmarks": landmarks,
        "score": 0.95,
        "person_id": person_id,
        "similarity": similarity,
    }


class FakeTracker:
    def __init__(self, **kwargs):
        self.tracks = []

    def update(self, detections, frame_shape, timestamp_ms):
        return [dict(t, bbox=list(t["bbox"])) for t in self.tracks]


class FakeNetworks:
    def __init__(self):
        self.tracks = []
        self.observations = []
        self.embedding_calls = 0

    def heads_and_bodies(self, frame, threshold):
        heads = np.asarray(
            [t["bbox"] + [t["score"]] for t in self.tracks], np.float32
        ).reshape(-1, 5)
        bodies = np.asarray(
            [
                [t["bbox"][0], 0, t["bbox"][2], frame.shape[0] - 1, 0.95]
                for t in self.tracks
            ],
            np.float32,
        ).reshape(-1, 5)
        return heads, bodies

    def faces(self, frame):
        return [
            dict(f, bbox=f["bbox"].copy(), landmarks=f["landmarks"].copy())
            for f in self.observations
        ]

    def embedding(self, frame, landmarks):
        self.embedding_calls += 1
        face = next(
            f for f in self.observations if np.array_equal(f["landmarks"], landmarks)
        )
        vector = np.zeros(512, np.float32)
        vector[ord(face["person_id"]) - ord("a")] = face["similarity"]
        return vector, np.zeros((112, 112, 3), np.uint8)


class FakeGallery:
    def __init__(self):
        self.by_id = {pid: {"display_name": pid.upper()} for pid in ("a", "b")}
        self.references = {pid: np.zeros((112, 112, 3), np.uint8) for pid in self.by_id}

    def match(self, embedding):
        index = int(np.argmax(embedding))
        return chr(ord("a") + index), float(embedding[index]), 0.3


@pytest.fixture
def scene(monkeypatch):
    monkeypatch.setattr(pipeline, "HeadTracker", FakeTracker)
    monkeypatch.setattr(
        pipeline, "quality", lambda frame, face, config: {"usable": True, "rank": 100.0}
    )
    config = dict(
        fps=10,
        head_threshold=0.5,
        head_low_threshold=0.1,
        tracker_buffer_seconds=2,
        face_interval_ms=100,
        roi_face_detection=False,
        require_body_support=True,
        require_head_evidence=True,
        match_threshold=0.6,
        match_margin=0.1,
        confirm_samples=3,
        confirm_span_ms=200,
        evidence_window_ms=3000,
        identity_ttl_ms=1000,
    )
    networks = FakeNetworks()
    service = pipeline.FaceIDService(networks, FakeGallery(), config, "synthetic")
    return SimpleNamespace(
        service=service,
        networks=networks,
        frame=np.zeros((200, 400, 3), np.uint8),
    )


def _process(scene, timestamp_ms, tracks, faces=()):
    scene.service.tracker.tracks = tracks
    scene.networks.tracks = tracks
    scene.networks.observations = list(faces)
    return scene.service.process(scene.frame, timestamp_ms)


def _confirm(scene, two_people=False):
    tracks = [_track(11, [20, 20, 120, 120])]
    faces = [_face([40, 40, 100, 100], "a")]
    expected = {(11, "a")}
    if two_people:
        tracks.append(_track(42, [220, 20, 320, 120]))
        faces.append(_face([240, 40, 300, 100], "b"))
        expected.add((42, "b"))
    for timestamp in (0, 100, 200):
        result = _process(scene, timestamp, tracks, faces)
        if timestamp < 200:
            assert all(t["person_id"] is None for t in result["tracks"])
            assert not result["events"]
    assert {(t["track_id"], t["person_id"]) for t in result["tracks"]} == expected
    assert all(t["recognition_state"] == "identified" for t in result["tracks"])
    assert len(result["events"]) == len(expected)
    assert all(e["event"] == "identity_confirmed" for e in result["events"])
    assert set(scene.service.evidence) == {tid for tid, _ in expected}
    return tracks


def _assert_revocations(result, expected):
    assert len(result["events"]) == len(expected)
    assert all(e["event"] == "identity_revoked" for e in result["events"])
    assert {(e["track_id"], e["person_id"]) for e in result["events"]} == expected
    for event in result["events"]:
        assert event["timestamp_ms"] == result["timestamp_ms"]
        assert event["camera_id"] == "synthetic"
        assert event["display_name"] == event["person_id"].upper()


def _assert_unidentified(result, track_ids, status=None):
    assert {t["track_id"] for t in result["tracks"]} == set(track_ids)
    for track in result["tracks"]:
        assert track["person_id"] is None
        assert track["display_name"] is None
        assert track["last_face_confirmed_ms"] is None
        assert track["recognition_state"] != "identified"
        if status is not None:
            assert track["recognition_state"] == status


@pytest.mark.parametrize("ambiguity", ["competing_heads", "competing_faces"])
def test_ambiguity_clears_live_ids_and_emits_revocations(scene, ambiguity):
    tracks = _confirm(scene, two_people=ambiguity == "competing_heads")
    expected = {(11, "a")}
    if ambiguity == "competing_heads":
        tracks[1] = _track(42, [80, 20, 180, 120])
        faces = [_face([85, 50, 115, 90])]
        expected.add((42, "b"))
    else:
        faces = [_face([25, 45, 60, 95]), _face([80, 45, 115, 95], "b")]
    embedding_calls = scene.networks.embedding_calls

    result = _process(scene, 300, tracks, faces)

    _assert_unidentified(result, {tid for tid, _ in expected}, "pending")
    _assert_revocations(result, expected)
    assert scene.networks.embedding_calls == embedding_calls
    for tid, _ in expected:
        assert scene.service.states[tid].person_id is None
        assert not scene.service.states[tid].samples
        assert not scene.service.crops[tid]
        assert scene.service.evidence[tid]["revoked"] is True
    assert not _process(scene, 400, tracks, faces)["events"]


def test_ttl_emits_revocation_without_a_tracking_gap(scene):
    tracks = _confirm(scene)
    # Keep head updates within 700 ms so only identity TTL causes this reset.
    for timestamp in (700, 1200):
        result = _process(scene, timestamp, tracks)
        assert result["tracks"][0]["person_id"] == "a"
        assert result["tracks"][0]["last_face_confirmed_ms"] == 200
        assert not result["events"]

    result = _process(scene, 1201, tracks)

    _assert_unidentified(result, {11})
    _assert_revocations(result, {(11, "a")})
    assert scene.service.states[11].person_id is None
    assert scene.service.evidence[11]["revoked"] is True
    assert not _process(scene, 1301, tracks)["events"]


def test_reappearing_track_gap_revokes_before_identity_ttl(scene):
    tracks = _confirm(scene)

    result = _process(scene, 901, tracks)

    _assert_unidentified(result, {11})
    _assert_revocations(result, {(11, "a")})
    assert scene.service.states[11].person_id is None
    assert not scene.service.crops[11]
    assert scene.service.evidence[11]["revoked"] is True
    assert not _process(scene, 1001, tracks)["events"]


def test_lost_track_revokes_after_tracker_buffer(scene):
    _confirm(scene)
    for timestamp in (300, 2200):
        result = _process(scene, timestamp, [])
        assert not result["tracks"]
        assert not result["events"]

    result = _process(scene, 2201, [])

    _assert_unidentified(result, set())
    _assert_revocations(result, {(11, "a")})
    assert scene.service.states[11].person_id is None
    assert scene.service.evidence[11]["revoked"] is True
    assert not _process(scene, 2301, [])["events"]


def test_removed_track_emits_revocation_before_state_is_discarded(scene):
    _confirm(scene)

    result = _process(scene, 30201, [])

    _assert_unidentified(result, set())
    _assert_revocations(result, {(11, "a")})
    assert 11 not in scene.service.states
    assert scene.service.evidence[11]["revoked"] is True
    assert not _process(scene, 30301, [])["events"]


@pytest.mark.parametrize("reverse_order", [False, True])
def test_duplicate_claim_has_no_confidently_assigned_winner(scene, reverse_order):
    tracks = _confirm(scene)
    tracks.append(_track(42, [220, 20, 320, 120]))
    if reverse_order:
        tracks.reverse()
    faces = [_face([240, 40, 300, 100], "a", similarity=0.7)]
    # The weaker claimant is newer than the established owner, in either order.
    for timestamp in (300, 400):
        result = _process(scene, timestamp, tracks, faces)
        by_track = {t["track_id"]: t for t in result["tracks"]}
        assert by_track[11]["person_id"] == "a"
        assert by_track[42]["person_id"] is None
        assert not result["events"]

    result = _process(scene, 500, tracks, faces)

    _assert_unidentified(result, {11, 42}, "pending")
    _assert_revocations(result, {(11, "a"), (42, "a")})
    assert scene.service.diagnostics["duplicate_identity_suppressed"] == 2
    for tid in (11, 42):
        assert scene.service.states[tid].person_id is None
        assert not scene.service.crops[tid]
        assert scene.service.evidence[tid]["revoked"] is True
    assert not _process(scene, 600, tracks)["events"]
