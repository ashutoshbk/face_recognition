import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_backend import head_detail_regions
from runtime import Camera, Manager


def test_detail_regions_stay_in_native_coordinates_and_are_bounded():
    heads=np.array([[0,0,50,60,.9],[2500,1400,2560,1440,.9]])
    regions=head_detail_regions(heads,[],(1440,2560,3))
    assert len(regions)==2
    for x1,y1,x2,y2 in regions:
        assert 0 <= x1 < x2 <= 2560 and 0 <= y1 < y2 <= 1440
        assert x2-x1 <= 640 and y2-y1 <= 640
    assert regions[1][0] > 2000


def test_detail_budget_and_round_robin():
    heads=np.array([[i*200,200,i*200+50,260,.9] for i in range(8)])
    first=head_detail_regions(heads,[],(1440,2560,3),cursor=0)
    second=head_detail_regions(heads,[],(1440,2560,3),cursor=4)
    assert len(first)==len(second)==4 and not set(first).intersection(second)


def test_clear_full_frame_face_is_not_repeated():
    heads=np.array([[100,100,200,220,.9]])
    face=np.array([110,110,80,95,*([0]*10),.95])
    assert head_detail_regions(heads,[face],(1440,2560,3))==[]


def test_small_face_is_still_eligible_for_detail_without_inflating_its_size():
    head=np.array([[100,100,200,220,.9]])
    face=np.array([130,120,30,35,*([0]*10),.95])
    assert len(head_detail_regions(head,[face],(1440,2560,3)))==1
    assert face[2]==30


def test_camera_face_counters_are_plain_json_numbers():
    import json
    camera=Camera("test","rtsp://127.0.0.1/example")
    camera.face_observations=[dict(usable=False,small=True,assigned=False)]
    value=camera.public()["face_counts"]
    assert value==dict(detected=1,usable=0,small=1,unassigned=1)
    json.dumps(value)
