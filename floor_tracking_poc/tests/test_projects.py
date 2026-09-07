import json

import cv2
import numpy as np
import pytest

from camex_floor.pipeline import Projects
from camex_floor.geometry import calibration_fingerprint


def image_upload(store, name="floor.png"):
    path = store.workspace / "uploads" / "image.bin"
    cv2.imencode(".png", np.full((480, 640, 3), 255, np.uint8))[1].tofile(path)
    return store.register_upload(path, name)["upload_id"]


def video_upload(store):
    path = store.workspace / "uploads" / "video.bin"
    temp = path.with_suffix(".mp4")
    writer = cv2.VideoWriter(str(temp), cv2.VideoWriter_fourcc(*"mp4v"), 10, (640, 480))
    for _ in range(20):
        writer.write(np.zeros((480, 640, 3), np.uint8))
    writer.release()
    temp.rename(path)
    return store.register_upload(path, "new_scene.mp4")["upload_id"]


def prepared(tmp_path):
    store = Projects(tmp_path)
    site = store.create("New building", image_upload(store))
    pid = site["project_id"]
    store.add_camera(pid, "arbitrary_99", "Camera", video_upload(store))
    return store, pid


def test_new_site_video_reference_and_persistent_reopen(tmp_path):
    store, pid = prepared(tmp_path)
    s = store.snapshot(pid, "arbitrary_99", 1.0)
    assert s["cameras"]["arbitrary_99"]["reference_time_s"] == 1
    reopened = Projects(tmp_path).info(pid)
    assert reopened["cameras"]["arbitrary_99"]["video_duration_s"] == 2
    assert reopened["cameras"]["arbitrary_99"]["calibration_state"] == "not_calibrated"


def test_finish_does_not_approve_uncalibrated_camera(tmp_path):
    store, pid = prepared(tmp_path)
    with pytest.raises(ValueError):
        store.finish(pid, False)
    with pytest.raises(ValueError):
        store.finish(pid, True)


def test_finish_save_and_reopen_approved_site(tmp_path):
    store, pid = prepared(tmp_path)
    s = store.load(pid)
    s["floor"]["pixels_per_metre"] = 80
    c = s["cameras"]["arbitrary_99"]
    c["image_to_floor_h"] = np.eye(3).tolist()
    c["anchors"] = [
        {"role": "check", "image_px": [100, 100], "map_px": [100, 100]},
        {"role": "check", "image_px": [400, 350], "map_px": [400, 350]},
    ]
    c["approved_fingerprint"] = calibration_fingerprint(c, s["floor"])
    store.save(pid, s)
    finished = store.finish(pid, False)
    assert finished["state"] == "approved"
    assert (store.project_dir(pid) / "finished_site.json").is_file()


def test_camera_lock_and_revision_guard(tmp_path):
    store, pid = prepared(tmp_path)
    s = store.load(pid)
    s = store.edit(pid, "arbitrary_99", {"revision": s["revision"], "action": "lock"})
    with pytest.raises(ValueError):
        store.edit(pid, "arbitrary_99", {"revision": s["revision"], "action": "save"})
    with pytest.raises(ValueError):
        store.edit(pid, "arbitrary_99", {"revision": 0, "action": "unlock"})
    s = store.edit(pid, "arbitrary_99", {"revision": s["revision"], "action": "unlock"})
    assert not s["cameras"]["arbitrary_99"]["locked"]


def test_bad_input_paths_and_reference_time_fail(tmp_path):
    store, pid = prepared(tmp_path)
    with pytest.raises(ValueError):
        store.project_dir("../outside")
    with pytest.raises(ValueError):
        store.snapshot(pid, "arbitrary_99", 200)
    with pytest.raises(ValueError):
        store.add_camera(pid, "arbitrary_99", "duplicate", video_upload(store))


def test_job_restart_is_explicitly_interrupted(tmp_path):
    store, pid = prepared(tmp_path)
    folder = store.project_dir(pid) / "jobs" / "testjob"
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(
        json.dumps({"id": "testjob", "state": "running"})
    )
    assert Projects(tmp_path).job_info(pid, "testjob")["state"] == "interrupted"


def test_missing_engine_does_not_fall_back_to_cached_demo(tmp_path):
    store, pid = prepared(tmp_path)
    with pytest.raises((FileNotFoundError, ValueError)):
        store.enqueue(pid, "calibrate", "arbitrary_99")


def test_remove_camera_preserves_files_and_other_projects(tmp_path):
    store, pid = prepared(tmp_path)
    store.add_camera(pid, "keep_me", "Other camera", video_upload(store))
    before = store.load(pid)
    folder = store.project_dir(pid)
    media = folder / before["cameras"]["arbitrary_99"]["video"]
    original_bytes = media.read_bytes()
    (folder / "finished_site.json").write_text(json.dumps(before))
    other = store.create("Unrelated project", image_upload(store))
    untouched = store.load(other["project_id"])
    result = store.remove_camera(pid, "arbitrary_99", before["revision"])
    assert set(result["cameras"]) == {"keep_me"}
    assert result["state"] == "draft"
    assert media.read_bytes() == original_bytes
    assert result["cameras"]["keep_me"] == before["cameras"]["keep_me"]
    archives = list((folder / "camera_archive").glob("*.json"))
    assert len(archives) == 1
    assert json.loads(archives[0].read_text())["site_before_removal"] == before
    assert json.loads((folder / "finished_site.json").read_text())["requires_finish"]
    assert store.load(other["project_id"]) == untouched


def test_remove_last_camera_leaves_empty_persistent_site(tmp_path):
    store, pid = prepared(tmp_path)
    current = store.load(pid)
    store.remove_camera(pid, "arbitrary_99", current["revision"])
    assert Projects(tmp_path).load(pid)["cameras"] == {}
    assert store.list()[0]["camera_count"] == 0
    with pytest.raises(ValueError):
        store.finish(pid, True)
    added = store.add_camera(pid, "arbitrary_99", "Readded", video_upload(store))
    assert added["cameras"]["arbitrary_99"]["calibration_state"] == "not_calibrated"


def test_remove_camera_revision_lock_and_job_guards(tmp_path):
    store, pid = prepared(tmp_path)
    current = store.load(pid)
    with pytest.raises(ValueError, match="Project changed"):
        store.remove_camera(pid, "arbitrary_99", -1)
    with pytest.raises(ValueError, match="not registered"):
        store.remove_camera(pid, "missing", current["revision"])
    locked = store.edit(
        pid, "arbitrary_99", {"revision": current["revision"], "action": "lock"}
    )
    with pytest.raises(ValueError, match="Unlock"):
        store.remove_camera(pid, "arbitrary_99", locked["revision"])
    unlocked = store.edit(
        pid, "arbitrary_99", {"revision": locked["revision"], "action": "unlock"}
    )
    job = store.project_dir(pid) / "jobs" / "running"
    job.mkdir(parents=True)
    (job / "status.json").write_text(json.dumps({"id": "running", "state": "running"}))
    with pytest.raises(ValueError, match="Wait"):
        store.remove_camera(pid, "arbitrary_99", unlocked["revision"])
    assert "arbitrary_99" in store.load(pid)["cameras"]
    assert not (store.project_dir(pid) / "camera_archive").exists()


def test_cached_calibration_is_reapplied_to_readded_camera(tmp_path, monkeypatch):
    from camex_floor.geometry import digest
    from camex_floor.pipeline import file_hash
    import camex_floor.pipeline as pipeline
    from types import SimpleNamespace

    store, pid = prepared(tmp_path)
    (tmp_path / "gpu").mkdir()
    (tmp_path / "gpu/worker.py").write_text("# worker fixture")
    (tmp_path / "runtime.json").write_text(
        json.dumps(
            {
                "mode": "full",
                "gpu_image": "test",
                "gpu_image_id": "test",
                "model_hashes": {},
            }
        )
    )
    monkeypatch.setattr(store, "verify_models", lambda runtime: None)
    monkeypatch.setattr(
        pipeline.subprocess, "run", lambda *a, **kw: SimpleNamespace(returncode=0)
    )
    camera = store.load(pid)["cameras"]["arbitrary_99"]
    source = {
        "camera_id": "arbitrary_99",
        "reference_sha256": file_hash(store.project_dir(pid) / camera["raw_reference"]),
        "floor_polygon": None,
        "level_method": "auto",
    }
    fingerprint = digest(
        {
            "source": source,
            "models": {},
            "image": "test",
            "worker_code": file_hash(tmp_path / "gpu/worker.py"),
            "importer_code": file_hash(
                pipeline.Path(pipeline.__file__).parent / "vggt_import.py"
            ),
        }
    )
    job = store.project_dir(pid) / "jobs/cached"
    job.mkdir(parents=True)
    (job / "status.json").write_text(
        json.dumps({"id": "cached", "state": "complete", "fingerprint": fingerprint})
    )
    applied = []
    monkeypatch.setattr(store, "accept_calibration", lambda *args: applied.append(args))
    result = store.enqueue(pid, "calibrate", "arbitrary_99")
    assert result["reused"] and applied[0][:2] == (pid, "cached")
    current = store.load(pid)
    current["cameras"]["arbitrary_99"]["calibration_fingerprint"] = fingerprint
    store.save(pid, current)
    assert store.enqueue(pid, "calibrate", "arbitrary_99")["reused"]
    assert len(applied) == 1, (
        "Unchanged calibration must not replace operator alignment"
    )


def changing_video_upload(store):
    path = store.workspace / "uploads/changing.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 10, (160, 120))
    for index in range(20):
        writer.write(np.full((120, 160, 3), 10 + index * 10, np.uint8))
    writer.release()
    return store.register_upload(path, "changing.mp4")["upload_id"]


def test_snapshot_pixels_follow_requested_time_not_first_frame(tmp_path):
    store = Projects(tmp_path)
    pid = store.create("Frame test", image_upload(store))["project_id"]
    store.add_camera(pid, "camera", "Changing frames", changing_video_upload(store))
    first = store.load(pid)["cameras"]["camera"]
    first_path = store.project_dir(pid) / first["raw_reference"]
    hashes = []
    for seconds, expected in [(0.4, 50), (1.2, 130), (1.8, 190)]:
        camera = store.snapshot(pid, "camera", seconds)["cameras"]["camera"]
        pixels = cv2.imread(str(store.project_dir(pid) / camera["reference"]))
        assert abs(float(pixels.mean()) - expected) < 8
        assert abs(camera["reference_actual_time_s"] - seconds) < 0.15
        assert camera["reference_capture"]["requested_time_s"] == seconds
        assert camera["image_size_wh"] == [160, 120]
        hashes.append(camera["reference_sha256"])
    assert len(set(hashes)) == 3
    assert first_path.exists()
    assert first["reference_sha256"] not in hashes


@pytest.mark.parametrize("seek_ok", [False, True])
def test_failed_or_ignored_seek_never_replaces_reference(
    tmp_path, monkeypatch, seek_ok
):
    import camex_floor.pipeline as pipeline

    store, pid = prepared(tmp_path)
    before = store.load(pid)

    class Capture:
        released = False

        def isOpened(self):
            return True

        def set(self, key, value):
            return seek_ok

        def get(self, key):
            return 10 if key == cv2.CAP_PROP_FPS else 0

        def read(self):
            return True, np.zeros((480, 640, 3), np.uint8)

        def release(self):
            self.released = True

    capture = Capture()
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda _: capture)
    with pytest.raises(ValueError, match="seek|confirm"):
        store.snapshot(pid, "arbitrary_99", 1.0)
    assert capture.released
    assert store.load(pid) == before


def test_new_reference_clears_derived_geometry_but_not_video_or_scale(tmp_path):
    store, pid = prepared(tmp_path)
    site = store.load(pid)
    site["floor"]["pixels_per_metre"] = 80
    camera = site["cameras"]["arbitrary_99"]
    camera.update(
        all_point_layer="old.png",
        local_geometry_status="provisional",
        gravity_camera=[0, 1, 0],
        camera_to_floor_R=np.eye(3).tolist(),
        fit_audit={"old": True},
    )
    camera["video_start_seconds"] = 0.5
    store.save(pid, site)
    result = store.snapshot(pid, "arbitrary_99", 1.0)
    new = result["cameras"]["arbitrary_99"]
    for key in (
        "all_point_layer",
        "local_geometry_status",
        "gravity_camera",
        "camera_to_floor_R",
        "fit_audit",
    ):
        assert key not in new
    assert new["video"] == camera["video"] and new["video_start_seconds"] == 0.5
    assert result["floor"]["pixels_per_metre"] == 80
    assert new["calibration_state"] == "not_calibrated" and new["anchors"] == []


def test_locked_camera_reference_is_preserved(tmp_path):
    store, pid = prepared(tmp_path)
    site = store.load(pid)
    store.edit(pid, "arbitrary_99", {"action": "lock", "revision": site["revision"]})
    before = store.load(pid)
    with pytest.raises(ValueError, match="Unlock"):
        store.snapshot(pid, "arbitrary_99", 1.0)
    assert store.load(pid) == before
