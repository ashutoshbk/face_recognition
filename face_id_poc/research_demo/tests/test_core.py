import time
from pathlib import Path

import numpy as np
import pytest

from benchmark import recall_at_k
from core import Evidence, Gallery, Session, assess_quality, decode_image, jpeg, unit


def face(x=180, y=120):
    return np.array([x,y,160,180,x+45,y+60,x+115,y+60,x+80,y+105,x+50,y+140,x+110,y+140,0.99], dtype=np.float32)


FRAME = np.random.default_rng(1).integers(50,200,size=(480,640,3),dtype=np.uint8)


class FakeEngine:
    faces = [face()]
    heads = np.array([[165,90,355,310,.95]],dtype=np.float32)
    feature = unit(np.ones(512))

    def detect(self, frame):
        return {'heads':self.heads,'faces':self.faces,'bodies':[]}

    def encode(self, frame, face):
        return FRAME[:112,:112], self.feature

    def encode_crop(self,crop):
        return self.feature


def test_evidence_requires_full_spaced_window_and_expires():
    evidence = Evidence()
    for i in range(4):
        evidence.add(i*0.4,0.8,0.45)
    assert evidence.summary(1.2)["state"] == "Collecting evidence"
    evidence.add(1.21,0.8,0.45)
    assert evidence.summary(1.21)["count"] == 4
    evidence.add(1.6,0.8,0.45)
    assert evidence.summary(1.6)["state"] == "Match"
    assert evidence.summary(8)["count"] == 0


def test_votes_are_not_nearest_identity_selection():
    evidence = Evidence()
    for i, score in enumerate([.7,.2,.1,.3,.6]):
        evidence.add(i*.4,score,.45)
    assert evidence.summary(1.6)["state"] == "No match"


def test_quality_gates():
    assert assess_quality(FRAME,face())["accepted"]
    assert "Blur" in assess_quality(np.full_like(FRAME,125),face())["reasons"]
    assert "Clipped face" in assess_quality(FRAME,face(x=-4))["reasons"]
    small = face(); small[2] = 40
    assert "Face too small" in assess_quality(FRAME,small)["reasons"]
    profile = face(); profile[8] += 65
    assert "Head angle" in assess_quality(FRAME,profile)["reasons"]


def test_decode_rejects_invalid_input():
    with pytest.raises(ValueError):
        decode_image(b"")
    with pytest.raises(ValueError):
        decode_image(b"x"*2_000_001)
    with pytest.raises(ValueError):
        decode_image(b"not an image")
    assert decode_image(jpeg(FRAME)).shape == FRAME.shape


def test_selected_tester_registration_votes_reset_and_delete(tmp_path):
    gallery = Gallery(tmp_path)
    session = Session()
    engine = FakeEngine()
    start = time.monotonic()
    for i in range(8):
        result = session.process(engine,FRAME,gallery,start+i*.4)
    assert result["enrollment_ready"]
    track = result["active_id"]
    with pytest.raises(ValueError, match="consent"):
        session.enroll("Tester",False,track,gallery)
    with pytest.raises(ValueError, match="track changed"):
        session.enroll("Tester",True,"invalid",gallery)
    identifier = session.enroll("Tester",True,track,gallery)
    assert len(gallery.list()) == 1
    for i in range(8,15):
        result = session.process(engine,FRAME,gallery,start+i*.4)
    assert result["verification"]["state"] == "Match"
    assert result["active_id"] == track
    engine.faces = []
    result = session.process(engine,FRAME,gallery,start+6)
    assert result["verification"]["state"] == "Head tracked / face unavailable"
    assert result['active_id'] == track
    assert session.candidates and not session.evidence.samples
    engine.faces = [face(),face(x=360)]
    result = session.process(engine,FRAME,gallery,start+6.4)
    assert result["verification"]["state"] == "Multiple heads or faces: paused"
    assert not result["enrollment_ready"]
    gallery.delete(identifier)
    assert not gallery.list()
    with pytest.raises(ValueError):
        gallery.get(identifier)


def test_config_change_and_track_gap_clear_samples(tmp_path):
    session, gallery, engine = Session(), Gallery(tmp_path), FakeEngine()
    start = time.monotonic()
    for i in range(6):
        session.process(engine,FRAME,gallery,start+i*.4)
    assert session.candidates
    session.configure(dict(selected=None,minimum_width=80,minimum_sharpness=50,threshold=.5),gallery)
    assert not session.candidates
    for i in range(6,12):
        session.process(engine,FRAME,gallery,start+i*.4)
    result = session.process(engine,FRAME,gallery,start+8)
    assert len(session.candidates) <= 1
    assert not result["enrollment_ready"]


def test_recall_uses_exact_reference():
    assert recall_at_k([1,2,3],[1,2,3]) == 1
    assert recall_at_k([1,2,4],[1,2,3]) == 2/3
    assert recall_at_k([1],[]) == 0


def test_enrollment_uses_best_photo_and_quality_weighted_template(tmp_path):
    gallery = Gallery(tmp_path)
    candidates = [{"quality":quality,"feature":unit(np.eye(512)[i]),"jpeg":bytes([i])} for i,quality in enumerate([40,95,60,80])]
    identifier = gallery.add("Quality test",candidates,True)
    record = gallery.get(identifier)
    assert record["photo"] == bytes([1])
    expected = np.zeros(512); expected[:4] = [0,95,60,80]
    assert np.allclose(record["template"],unit(expected))
    gallery.delete(identifier)
    with pytest.raises(ValueError):
        gallery.get(identifier)
