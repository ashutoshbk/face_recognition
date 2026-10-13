import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import core
from core import Engine, Gallery, Session, decode_image
from settings import MODELS


def test_real_model_plumbing_positive_negative_and_blur(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    engine = Engine(MODELS)
    gallery = Gallery(tmp_path)
    session = Session()
    first = decode_image((root / "qa/synthetic_1.jpg").read_bytes())
    other = decode_image((root / "qa/synthetic_2.jpg").read_bytes())
    start = time.monotonic()
    for i in range(8):
        result = session.process(engine,first,gallery,start+i*.4)
    assert result["enrollment_ready"]
    # Processing uses a simulated frame clock; enrollment must use that same clock.
    with monkeypatch.context() as patch:
        patch.setattr(core, 'time', SimpleNamespace(monotonic=lambda: start+7*.4,
            perf_counter=time.perf_counter, time=time.time))
        session.enroll("Synthetic QA fixture",True,result["active_id"],gallery)
    for i in range(8,16):
        result = session.process(engine,first,gallery,start+i*.4)
    assert result["verification"]["state"] == "Match"
    for i in range(16,26):
        result = session.process(engine,other,gallery,start+i*.4)
    assert result["verification"]["state"] == "No match"
    blurred = cv2.GaussianBlur(first,(61,61),30)
    result = session.process(engine,blurred,gallery,start+12)
    assert not result["enrollment_ready"]
    assert result["verification"]["state"] not in ("Match", "No match")
    assert len(gallery.list()) == 1
