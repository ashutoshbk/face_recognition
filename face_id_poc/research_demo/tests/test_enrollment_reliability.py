import time

import numpy as np

from core import Session, Gallery, jpeg
from self_test import SelfSession
from test_core import FakeEngine, FRAME
from test_self_test import setup


def test_slow_but_continuous_frames_can_collect_three_samples(tmp_path):
    gallery,search,_=setup(tmp_path)
    session=SelfSession(search,dispatch=lambda fn:fn());engine=FakeEngine();now=time.monotonic()
    ids=[]
    for i in range(4):
        result=session.process(engine,FRAME,gallery,now+i*1.25)
        ids.append(result['active_id'])
    assert len(set(ids))==1
    assert result['enrollment_ready'] and len(result['samples'])>=3
    assert result['capture']['reset_count']==0


def test_good_samples_survive_quality_pauses_with_same_visible_head(tmp_path):
    gallery,search,_=setup(tmp_path);session=SelfSession(search,dispatch=lambda fn:fn())
    engine=FakeEngine();now=time.monotonic();blank=np.zeros_like(FRAME)
    for i in range(201):
        good=i in (0,100,200)
        result=session.process(engine,FRAME if good else blank,gallery,now+i*.1)
        if not good:assert result['verification']['state']!='Match'
    assert result['enrollment_ready'] and len(result['samples'])==3
    assert result['capture']['sample_window_seconds']==30


def test_repeated_identical_settings_do_not_clear_samples(tmp_path):
    session=Session();gallery=Gallery(tmp_path);engine=FakeEngine();now=time.monotonic()
    settings={'selected':None,'threshold':.45,'minimum_width':70,'minimum_sharpness':45}
    for i in range(4):
        result=session.process(engine,FRAME,gallery,now+i*.4)
        session.configure(settings,gallery)
    assert result['enrollment_ready'] and len(session.candidates)>=3
    session.configure({**settings,'minimum_sharpness':50},gallery)
    assert not session.candidates
    assert session.last_capture_reset=='Capture settings or profile changed'


class DeferredRecognizer:
    def status(self):return {'pid':777,'state':'ready','kind':'process'}
    def try_submit(self,crop,selected,callback):
        self.crop=crop;self.callback=callback;return True
    def complete(self,search,feature):
        self.callback({'feature':feature,'found':search.search(feature,None),'photo':jpeg(self.crop),
          'encoding_ms':20,'recognition_ms':22,'completed_at':time.monotonic(),'worker_pid':777})


def test_good_inflight_crop_survives_blur_without_publishing_old_identity(tmp_path):
    gallery,search,_=setup(tmp_path);worker=DeferredRecognizer()
    session=SelfSession(search,recognizer=worker);engine=FakeEngine();now=time.monotonic()
    session.process(engine,FRAME,gallery,now)
    session.process(engine,np.zeros_like(FRAME),gallery,now+.1)
    worker.complete(search,engine.feature)
    result=session.process(engine,np.zeros_like(FRAME),gallery,now+.2)
    assert len(result['samples'])==1
    assert result['self_search'] is None and result['verification']['state']!='Match'
    assert not result['enrollment_ready']


def test_late_crop_cannot_bridge_a_capture_reset(tmp_path):
    gallery,search,_=setup(tmp_path);worker=DeferredRecognizer()
    session=SelfSession(search,recognizer=worker);engine=FakeEngine();now=time.monotonic()
    session.process(engine,FRAME,gallery,now)
    session.clear_capture('Test explicit reset');worker.complete(search,engine.feature)
    result=session.process(engine,FRAME,gallery,now+.2)
    assert not result['samples'] and result['self_search'] is None


def test_long_head_gap_still_clears_capture_and_reports_why(tmp_path):
    gallery,search,_=setup(tmp_path);session=SelfSession(search,dispatch=lambda fn:fn())
    engine=FakeEngine();now=time.monotonic()
    for i in range(3):result=session.process(engine,FRAME,gallery,now+i*.9)
    original=result['active_id'];assert len(session.candidates)==2
    result=session.process(engine,FRAME,gallery,now+4.1)
    assert result['active_id']!=original
    assert len(result['samples'])<=1 and result['capture']['reset_count']>=1
    assert any('2 seconds' in event['reason'] for event in session.capture_events)


def test_changing_profile_invalidates_pending_enrollment_crop(tmp_path):
    gallery,search,_=setup(tmp_path);worker=DeferredRecognizer()
    session=SelfSession(search,recognizer=worker);engine=FakeEngine();now=time.monotonic()
    session.process(engine,FRAME,gallery,now)
    session.configure({'selected':None,'threshold':.6,'minimum_width':70,'minimum_sharpness':45},gallery)
    worker.complete(search,engine.feature)
    result=session.process(engine,FRAME,gallery,now+.2)
    assert not result['samples']
