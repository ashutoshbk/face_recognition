import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_backend import Vision


@pytest.mark.skipif(not os.environ.get("CAMEX_TEST_MODELS"), reason="Separate model bundle required")
def test_real_model_execution(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMEX_OFFICE_PROFILE", "1")
    provider = os.environ.get("CAMEX_TEST_PROVIDER", "CPUExecutionProvider")
    vision = Vision(os.environ["CAMEX_TEST_MODELS"], provider, tmp_path)
    heads, faces = vision.detect(np.zeros((320, 480, 3), np.uint8))
    assert heads.shape[1] == 5
    descriptor = vision.encode(np.zeros((112, 112, 3), np.uint8))
    assert descriptor.shape == (512,) and np.isfinite(descriptor).all()
    assert abs(np.linalg.norm(descriptor)-1) < 1e-4
    profile = vision.profile_summary()
    assert all(item["provider_node_events"].get(provider, 0) > 0 for item in profile.values())
