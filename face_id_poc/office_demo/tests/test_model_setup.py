import hashlib
import importlib.util
import json
from pathlib import Path
import stat
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("prepare_models", ROOT / "scripts/prepare_models.py")
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def digest(value):
    return hashlib.sha256(value).hexdigest()


def test_nested_bundle_is_flattened_and_reusable(tmp_path):
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("models/detector.onnx", b"detector")
        archive.writestr("../../unexpected.txt", b"never extracted")
    output = tmp_path / "private-models"
    wanted = {"detector.onnx": digest(b"detector")}
    assert setup.install(bundle, output, wanted)["reused"] is False
    assert (output / "detector.onnx").read_bytes() == b"detector"
    assert not (tmp_path / "unexpected.txt").exists()
    assert stat.S_IMODE((output / "detector.onnx").stat().st_mode) == 0o600
    assert setup.install(bundle, output, wanted)["reused"] is True


def test_mismatch_leaves_no_partially_installed_directory(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "detector.onnx").write_bytes(b"wrong")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="checksum"):
        setup.install(source, output, {"detector.onnx": digest(b"right")})
    assert not output.exists()


def test_existing_output_is_not_overwritten(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "detector.onnx").write_bytes(b"keep")
    with pytest.raises(ValueError, match="nothing overwritten"):
        setup.install(tmp_path, output, {"detector.onnx": digest(b"different")})
    assert (output / "detector.onnx").read_bytes() == b"keep"


def test_ambiguous_archive_and_symbolic_links_are_rejected(tmp_path):
    bundle = tmp_path / "bundle.zip"
    with zipfile.ZipFile(bundle, "w") as archive:
        archive.writestr("a/detector.onnx", b"one")
        archive.writestr("b/detector.onnx", b"two")
    with pytest.raises(ValueError, match="exactly one"):
        setup.install(bundle, tmp_path / "out1", {"detector.onnx": digest(b"one")})
    with zipfile.ZipFile(bundle, "w") as archive:
        member = zipfile.ZipInfo("models/detector.onnx")
        member.create_system = 3
        member.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(member, "target")
    with pytest.raises(ValueError, match="Invalid model"):
        setup.install(bundle, tmp_path / "out2", {"detector.onnx": digest(b"target")})


def test_default_manifest_needs_only_existing_three_models():
    manifest = json.loads((ROOT / "models_manifest.json").read_text())
    assert len(manifest["models"]) == 3
    assert "scrfd-kps-2.5g-640x640.onnx" in manifest["models"]
    assert "det_10g.onnx" not in manifest["models"]
    compose = (ROOT / "docker-compose.yml").read_text()
    assert "${OFFICE_FACE_MODEL:-scrfd_2.5g}" in compose
    assert "${OFFICE_CAMERA_LIMIT:-32}" in compose
