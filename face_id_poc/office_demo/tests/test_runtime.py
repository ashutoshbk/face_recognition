import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from runtime import job_current
import runtime
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
