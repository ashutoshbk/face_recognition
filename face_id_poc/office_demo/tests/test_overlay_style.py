import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import runtime


def test_held_head_is_kept_in_state_but_not_drawn(monkeypatch):
    monkeypatch.setattr(runtime,"encode_jpeg",lambda frame:frame)
    manager=runtime.Manager.__new__(runtime.Manager)
    track=dict(tracking_state="coasting")
    camera=SimpleNamespace(tracks={"1":track},face_observations=[])
    image=manager._draw(camera,np.zeros((100,100,3),np.uint8))
    assert not image.any()
    assert camera.tracks["1"] is track


def test_head_solid_and_face_dotted(monkeypatch):
    monkeypatch.setattr(runtime,"encode_jpeg",lambda frame:frame)
    manager=runtime.Manager.__new__(runtime.Manager)
    track=dict(id="1",bbox=[10,25,80,95],state="Unknown",name=None,observations=0,
               tracking_state="observed")
    face=dict(bbox=[30,40,50,60],usable=True,landmarks=[],width=20,reasons=[])
    camera=SimpleNamespace(tracks={"1":track},face_observations=[face])
    image=manager._draw(camera,np.zeros((100,100,3),np.uint8))
    assert np.all(np.any(image[30:85,10] != 0,axis=1))
    assert image[40,30].any() and not image[40,33].any()
