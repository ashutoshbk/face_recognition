import hashlib, json
from camex_floor.pipeline import Projects
import pytest


def test_model_change_is_detected_before_cache_reuse(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    path = models / "test.bin"
    path.write_bytes(b"approved")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "models_manifest.json").write_text(
        json.dumps({"models": {"test": {"filename": "test.bin", "sha256": sha}}})
    )
    store = Projects(tmp_path)
    runtime = {"models_dir": str(models), "model_hashes": {"test": sha}}
    store.verify_models(runtime)
    store.verify_models(runtime)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError):
        store.verify_models(runtime)


def test_missing_model_is_detected_even_if_verification_was_cached(tmp_path):
    models = tmp_path / "models"
    models.mkdir()
    path = models / "test.bin"
    path.write_bytes(b"approved")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "models_manifest.json").write_text(
        json.dumps({"models": {"test": {"filename": "test.bin", "sha256": sha}}})
    )
    store = Projects(tmp_path)
    runtime = {"models_dir": str(models), "model_hashes": {"test": sha}}
    store.verify_models(runtime)
    path.unlink()
    with pytest.raises(ValueError):
        store.verify_models(runtime)
