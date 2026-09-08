import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

spec = importlib.util.spec_from_file_location(
    "launcher", Path(__file__).parents[1] / "scripts/run_demo.py"
)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_kit_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError):
        launcher.kit_path(tmp_path, "../secret")
    with pytest.raises(ValueError):
        launcher.kit_path(tmp_path, "/absolute")


def test_kit_verification_detects_modified_data(tmp_path):
    data = tmp_path / "clip.mp4"
    data.write_bytes(b"original")
    (tmp_path / "FILE_MANIFEST.json").write_text(
        json.dumps(
            {
                "files": [
                    {
                        "path": "clip.mp4",
                        "size": 8,
                        "sha256": hashlib.sha256(b"original").hexdigest(),
                    }
                ]
            }
        )
    )
    launcher.verify_kit(tmp_path)
    data.write_bytes(b"modified")
    with pytest.raises(ValueError):
        launcher.verify_kit(tmp_path)


def data(tmp_path):
    return {
        "models": tmp_path / "models with spaces",
        "photos": tmp_path / "photos",
        "video": tmp_path / "video.mp4",
        "config": tmp_path / "config.json",
        "output": tmp_path / "out",
        "camera_id": "different-camera",
        "clock_start": "00:00:00",
    }


def test_docker_mounts_are_explicit_and_renamed_source_independent(tmp_path):
    steps = launcher.commands("docker", data(tmp_path), 10)
    assert len(steps) == 2
    assert "--gpus" in steps[0] and "none" in steps[0]
    assert any("models with spaces" in value for value in steps[0])
    assert not any("/work/package" in value for value in steps[0])
    assert steps[0][-5:] == [
        "enroll",
        "--photos",
        "/photos",
        "--output",
        "/results/gallery",
    ]
    assert "/results/demo" in steps[1]


def test_bad_mount_and_duration_rejected(tmp_path):
    with pytest.raises(ValueError):
        launcher.mount(tmp_path / "comma,path", "/input")
    with pytest.raises(ValueError):
        launcher.commands("docker", data(tmp_path), float("nan"))
    with pytest.raises(ValueError):
        launcher.commands("docker", data(tmp_path), -1)


def test_cpu_backend_uses_separate_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    binary = tmp_path / ".venv-cpu/bin/camex-faceid"
    binary.parent.mkdir(parents=True)
    binary.write_text("placeholder")
    steps = launcher.commands("cpu", data(tmp_path), 2)
    assert steps[0][0] == str(binary)
    assert "CPUExecutionProvider" in steps[0]
    assert "docker" not in steps[0]


def test_site_details_come_from_kit_not_source(tmp_path):
    (tmp_path / "custom.mov").write_bytes(b"video")
    (tmp_path / "references").mkdir()
    (tmp_path / "weights").mkdir()
    config = tmp_path / "custom.json"
    config.write_text("{}")
    (tmp_path / "FILE_MANIFEST.json").write_text(json.dumps({"files": []}))
    (tmp_path / "demo.json").write_text(
        json.dumps(
            {
                "video": "custom.mov",
                "photos": "references",
                "models": "weights",
                "camera_id": "new_site",
                "clock_start": "11:00:00",
            }
        )
    )
    args = SimpleNamespace(
        kit=str(tmp_path),
        video=None,
        photos=None,
        models=None,
        camera_id=None,
        clock_start=None,
        config=str(config),
        output=str(tmp_path.parent / "new_output"),
    )
    resolved = launcher.resolve_inputs(args)
    assert resolved["video"] == tmp_path / "custom.mov"
    assert resolved["camera_id"] == "new_site"
