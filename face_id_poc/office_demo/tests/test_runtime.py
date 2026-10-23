import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import job_current
import runtime
from domain import HeadTrackIdentity
import threading
import numpy as np


def test_job_rejected_after_association_break():
    job = dict(track_token="old", generation=1, epoch=0, captured=10)
    track = dict(token="new", face_visible=True)
    assert not job_current(job, SimpleNamespace(generation=1), track, 0, 11)


def test_job_matches_only_current_fresh_state():
    job = dict(track_token="same", generation=1, epoch=0, captured=10)
    track = dict(token="same", face_visible=True)
    camera = SimpleNamespace(generation=1)
    assert job_current(job, camera, track, 0, 11)
    assert not job_current(job, camera, track, 1, 11)
    assert not job_current(job, camera, track, 0, 13)
    assert not job_current(job, SimpleNamespace(generation=2), track, 0, 11)
    track["face_visible"] = False
    assert job_current(job, camera, track, 0, 11)
    assert not job_current(job, camera, None, 0, 11)


def test_startup_index_retry_starts_workers_once(tmp_path, monkeypatch):
    class Gallery:
        count = 0
        def sync(self):
            self.count += 1
            if self.count == 1:
                raise RuntimeError("temporary index failure")
    monkeypatch.setattr(runtime, "Vision", lambda *args: SimpleNamespace())
    gallery = Gallery()
    manager = runtime.Manager(tmp_path, tmp_path, "CPUExecutionProvider", gallery)
    monkeypatch.setattr(manager.stop_event, "wait", lambda timeout: False)
    for name in ("_tracking_loop", "_recognition_loop", "_resources_loop"):
        monkeypatch.setattr(manager, name, lambda: None)
    manager._load()
    assert manager.workers_ready and gallery.count == 2 and len(manager.threads) == 3


def test_capture_does_not_wait_for_analysis_lock():
    camera = runtime.Camera("Synthetic", "rtsp://127.0.0.1/test")
    finished = threading.Event()
    def publish():
        camera._publish_frame(np.zeros((8, 8, 3), np.uint8), 10)
        finished.set()
    with camera.lock:
        thread = threading.Thread(target=publish)
        thread.start()
        assert finished.wait(1), "Decoder capture was blocked by the analysis lock"
    thread.join()
    assert camera.sequence == 1


def prepared_tracking_manager(tmp_path, monkeypatch):
    manager = runtime.Manager(tmp_path, tmp_path, "CPUExecutionProvider", SimpleNamespace())
    camera = runtime.Camera("Synthetic", "rtsp://127.0.0.1/test")
    camera.state = "running"
    camera.tracker = SimpleNamespace(ambiguous_track_ids=set(),
        update=lambda *args: [{"track_id": 7, "bbox": [10, 10, 50, 50]}])
    identity = HeadTrackIdentity()
    rank = [dict(person_id="tester", name="Tester", score=.9, registered_score=.9, registered=[.9, .9], live=[])]
    for i in range(3):
        identity.add(rank, 10+i*.6, i, runtime.DEFAULTS)
    camera.track_cache["7"] = dict(id="7", bbox=[10, 10, 50, 50], evidence=identity,
                                  token="original", last_seen=19.9, samples=[])
    manager.cameras[camera.id] = camera
    manager.vision = SimpleNamespace(detect=lambda frame, **kwargs: (manager.stop_event.set() or [], []))
    monkeypatch.setattr(manager, "_draw", lambda *args: b"test")
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 20.)
    camera._publish_frame(np.zeros((64, 64, 3), np.uint8), 20.)
    return manager, camera


def test_face_absence_preserves_name_on_visible_head(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    manager._tracking_loop()
    assert camera.tracks["7"]["name"] == "Tester"
    assert camera.tracks["7"]["state"] == "Tracked"
    assert camera.tracks["7"]["token"] == "original"
    assert not camera.tracks["7"]["admission"]


def test_ambiguous_head_cannot_keep_name(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    camera.tracker.ambiguous_track_ids = {7}
    manager._tracking_loop()
    assert camera.tracks["7"]["name"] is None
    assert camera.tracks["7"]["token"] != "original"


def test_expired_head_cannot_recover_old_name(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    camera.track_cache["7"]["last_seen"] = 17.
    manager._tracking_loop()
    assert camera.tracks["7"]["name"] is None


def test_missing_head_is_held_without_new_face_evidence(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    camera.tracker.update = lambda *args: []
    manager._tracking_loop()
    assert camera.tracks["7"]["tracking_state"] == "coasting"
    assert camera.tracks["7"]["name"] == "Tester"
    assert not camera.tracks["7"]["face_visible"]
    assert not camera.tracks["7"]["admission"]
    assert camera.track_cache["7"]["evidence"].summary(20)["name"] == "Tester"


def test_settings_reset_also_revokes_cached_names(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    manager.reset_evidence()
    assert camera.track_cache["7"]["evidence"].summary(20)["name"] is None
    assert camera.track_cache["7"]["token"] != "original"


def test_low_quality_face_does_not_erase_continuous_head_identity(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    manager.vision.detect = lambda frame, **kwargs: (manager.stop_event.set() or [], [np.zeros(15)])
    manager.vision.describe = lambda *args: ({"accepted": False, "reasons": ["Head angle"]}, None, None)
    monkeypatch.setattr(runtime, "associate_faces", lambda *args: ({0: 0}, set()))
    manager._tracking_loop()
    assert camera.tracks["7"]["name"] == "Tester"
    assert camera.tracks["7"]["state"] == "Tracked"
    assert not camera.tracks["7"]["face_visible"]


def test_ambiguous_face_association_clears_carried_name(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    monkeypatch.setattr(runtime, "associate_faces", lambda *args: ({}, {0}))
    manager._tracking_loop()
    assert camera.tracks["7"]["name"] is None
    assert camera.tracks["7"]["state"] == "Association ambiguous"


def test_short_head_gap_reconnects_same_local_track(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    camera.tracker.update = lambda *args: []
    manager._tracking_loop()
    manager.stop_event.clear()
    monkeypatch.setattr(runtime.time, "monotonic", lambda: 20.5)
    camera.tracker.update = lambda *args: [{"track_id": 7, "bbox": [10, 10, 50, 50]}]
    camera._publish_frame(np.zeros((64, 64, 3), np.uint8), 20.5)
    manager._tracking_loop()
    assert camera.tracks["7"]["name"] == "Tester"
    assert camera.tracks["7"]["token"] == "original"


def test_hold_ends_after_eight_tenths_of_a_second(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    camera.track_cache["7"]["last_seen"] = 19.
    camera.tracker.update = lambda *args: []
    manager._tracking_loop()
    assert camera.tracks == {}
    assert "7" in camera.track_cache


def test_pending_face_result_is_not_applied_to_coasting_track():
    job = dict(track_token="same", generation=1, epoch=0, captured=10)
    track = dict(token="same", tracking_state="coasting")
    assert not job_current(job, SimpleNamespace(generation=1), track, 0, 10.2)


def test_bytetrack_recovers_same_id_after_three_missing_detections():
    tracker = runtime.HeadTracker(fps=10, threshold=.35, buffer_seconds=2)
    shape = (480, 640, 3)
    identity = None
    for i in range(3):
        tracks = tracker.update([[100+i*2, 100, 150+i*2, 165, .95]], shape, 1000+i*100)
        identity = tracks[0]["track_id"]
    for i in range(3, 6):
        assert tracker.update([], shape, 1000+i*100) == []
    recovered = tracker.update([[112, 100, 162, 165, .95]], shape, 1600)
    assert recovered[0]["track_id"] == identity


def test_new_head_over_old_position_does_not_get_a_ghost_identity(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    camera.track_cache["7"]["bbox"] = [10, 10, 50, 50]
    camera.tracker.update = lambda *args: [{"track_id": 8, "bbox": [12, 12, 52, 52]}]
    manager._tracking_loop()
    assert "7" not in camera.tracks and "7" not in camera.track_cache
    assert camera.tracks["8"]["name"] is None


def test_kept_best_crop_stays_available_while_track_is_active(tmp_path, monkeypatch):
    manager, camera = prepared_tracking_manager(tmp_path, monkeypatch)
    sample = dict(id="best", quality_score=99)
    manager.samples["best"] = (21, b"image")
    assert manager._renew_samples([sample], 20) == [sample]
    assert manager._renew_samples([sample], 29) == [sample]
    assert manager.samples["best"][0] == 39
    manager._release_samples([sample])
    assert "best" not in manager.samples


def test_rejected_candidate_cannot_later_be_approved(tmp_path):
    manager = runtime.Manager(tmp_path, tmp_path, "CPUExecutionProvider", SimpleNamespace())
    manager.candidates["test"] = {"id":"test"}
    assert manager.decide_candidate("test", "reject") == {"ok": True}
    import pytest
    with pytest.raises(ValueError, match="already decided"):
        manager.decide_candidate("test", "approve", True)
