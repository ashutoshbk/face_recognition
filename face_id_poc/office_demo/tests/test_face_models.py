import sys
from pathlib import Path

import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from vision_backend import face_model_spec
from vision.networks import MODEL_HASHES


def test_existing_model_spec_is_unchanged():
    name,digest=face_model_spec("scrfd_2.5g")
    assert name=="scrfd-kps-2.5g-640x640.onnx"
    assert digest==MODEL_HASHES[name]


def test_ten_g_model_is_pinned_separately():
    name,digest=face_model_spec("scrfd_10g")
    assert name=="det_10g.onnx"
    assert digest=="5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"
    assert name not in MODEL_HASHES


def test_unknown_model_cannot_bypass_hash_check():
    with pytest.raises(ValueError,match="Unsupported"):
        face_model_spec("/tmp/arbitrary.onnx")
