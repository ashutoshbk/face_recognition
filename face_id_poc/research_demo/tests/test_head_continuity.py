import json
import sqlite3
import time

import numpy as np
import pytest

from core import Gallery, Session, MODEL_ID
from vision.association import associate_faces, TurnAwayCheck
from vision.head_tracker import HeadTracker
from test_core import FakeEngine, FRAME, face


def test_head_id_survives_five_seconds_without_face(tmp_path):
    session, gallery, engine = Session(), Gallery(tmp_path), FakeEngine()
    start = time.monotonic()
    seen = []
    for i in range(100):
        engine.faces = [face()] if i < 20 or i >= 70 else []
        # Head observations continue independently of all face observations.
        engine.heads = np.array([[165+i*.08,90,355+i*.08,310,.95]],np.float32)
        result = session.process(engine,FRAME,gallery,start+i*.1)
        if result['active_id']:
            seen.append(result['active_id'])
        if 20 <= i < 70:
            assert result['active_id'] is not None
            assert result['verification']['state'] == 'Head tracked / face unavailable'
            assert result['verification']['count'] == 0
            assert not result['enrollment_ready']
    assert len(set(seen)) == 1
    assert result['face_associated']


def test_long_head_loss_does_not_reuse_track_or_samples(tmp_path):
    session, gallery, engine = Session(), Gallery(tmp_path), FakeEngine()
    start = time.monotonic()
    for i in range(20):
        result = session.process(engine,FRAME,gallery,start+i*.1)
    original = result['active_id']
    engine.heads = np.empty((0,5)); engine.faces = []
    for i in range(20,40):
        result = session.process(engine,FRAME,gallery,start+i*.1)
    assert not session.candidates and not session.evidence.samples
    engine.heads = FakeEngine.heads; engine.faces = [face()]
    for i in range(40,50):
        result = session.process(engine,FRAME,gallery,start+i*.1)
    assert result['active_id'] != original


def test_face_must_be_inside_head_and_assignment_unambiguous():
    assert associate_faces([{'bbox':[160,90,360,320]}],[face()])[0] == {0:0}
    assert not associate_faces([{'bbox':[0,0,100,100]}],[face()])[0]
    matches, ambiguous = associate_faces([{'bbox':[160,90,360,320]},{'bbox':[161,90,361,320]}],[face()])
    assert not matches and ambiguous == {0,1}


def test_continuity_check_reports_only_observed_behavior():
    check = TurnAwayCheck()
    check.start('H-1',0)
    for i in range(1,60):
        check.update('H-1',i < 10 or i >= 40,1,False,i*.1)
    assert check.result['status'] == 'passed'
    assert check.result['face_unavailable_seconds'] >= 2.8
    assert check.result['maximum_head_gap_seconds'] == 0


@pytest.mark.parametrize('case',['track_change','head_loss','multiple_heads','ambiguity','timeout'])
def test_continuity_check_failure_modes(case):
    check = TurnAwayCheck(); check.start('H-1',0)
    if case == 'track_change': check.update('H-2',False,1,False,.1)
    if case == 'head_loss':
        check.update(None,False,0,False,.1); check.update(None,False,0,False,1.3)
    if case == 'multiple_heads': check.update('H-1',True,2,False,.1)
    if case == 'ambiguity': check.update('H-1',True,1,True,.1)
    if case == 'timeout': check.update('H-1',True,1,False,31)
    assert check.state in ('failed','interrupted','incomplete')


def test_legacy_sface_registration_preserved_but_not_loaded(tmp_path):
    legacy = tmp_path/'consenting_testers.sqlite3'
    with sqlite3.connect(legacy) as db:
        db.execute('CREATE TABLE testers (id TEXT, template BLOB)')
        db.execute('INSERT INTO testers VALUES (?,?)',('legacy',np.ones(128,dtype=np.float32).tobytes()))
    before = legacy.read_bytes()
    gallery = Gallery(tmp_path)
    assert gallery.legacy_count() == 1
    assert gallery.list() == []
    with pytest.raises(ValueError): gallery.get('legacy')
    assert legacy.read_bytes() == before


def test_model_mismatch_rejected_on_enrollment(tmp_path):
    gallery = Gallery(tmp_path)
    with pytest.raises(ValueError,match='512D'):
        gallery.add('Wrong model',[{'quality':90,'feature':np.ones(128),'jpeg':b'x'}]*3,True)


def test_brief_observation_gap_reports_gap_instead_of_false_continuity():
    check = TurnAwayCheck(); check.start('H-1',0)
    check.update('H-1',False,1,False,.1)
    check.update('H-1',False,1,False,1.2)
    check.update(None,False,0,False,1.3)
    check.update('H-1',False,1,False,1.6)
    check.update('H-1',True,1,False,1.8)
    assert check.result['status'] == 'passed'
    assert check.result['maximum_head_gap_seconds'] == .3


def test_completed_check_saves_metrics_without_images(tmp_path):
    reports = tmp_path/'checks'
    session = Session(report_directory=reports)
    gallery, engine = Gallery(tmp_path/'testers'), FakeEngine()
    start = time.monotonic()
    for i in range(10):
        result = session.process(engine,FRAME,gallery,start+i*.1)
    session.turnaway.start(result['active_id'],start+1)
    for i in range(10,55):
        engine.faces = [] if i < 45 else [face()]
        result = session.process(engine,FRAME,gallery,start+i*.1)
    files = list(reports.glob('*.json'))
    assert len(files) == 1
    report = json.loads(files[0].read_text())
    assert report['status'] == 'passed'
    assert report['model_id'] == MODEL_ID
    assert not {'image','photo','feature','template','name'} & report.keys()
