import time
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from core import Gallery, MODEL_ID, unit
from self_test import SelfSearch, SearchEvidence, SelfSession, SELF_POINT, suggest_threshold
from test_core import FakeEngine, FRAME, face


def enroll(gallery):
    return gallery.add('Self fixture',[{'quality':80,'feature':unit(np.ones(512)),'jpeg':b'fictional'} for _ in range(3)],True)


class FakeQdrant:
    def __init__(self):
        self.points={}
        self.queries=[]
        self.writes=[]
        self.self_score=.8
        self.synthetic_score=.6
    def collection_exists(self,name):return name in self.points
    def create_collection(self,name,**kwargs):self.points[name]=[]
    def upsert(self,name,points,**kwargs):self.points[name]=points;self.writes.append(name)
    def delete(self,name,**kwargs):self.points[name]=[];self.writes.append(name)
    def query_points(self,name,**kwargs):
        self.queries.append(name)
        if name=='synthetic':
            return SimpleNamespace(points=[SimpleNamespace(id=i,score=self.synthetic_score-.05*i) for i in range(5)])
        return SimpleNamespace(points=[SimpleNamespace(id=p.id,score=self.self_score,payload=p.payload) for p in self.points[name]])


def setup(tmp_path):
    gallery=Gallery(tmp_path,max_testers=1)
    search=SelfSearch(tmp_path,gallery,personal_collection='self-test')
    q=FakeQdrant();search.collection='synthetic';search.synthetic_count=109619;search.connect=lambda:q
    return gallery,search,q


def test_self_registration_and_union_search(tmp_path):
    gallery,search,q=setup(tmp_path)
    identifier=enroll(gallery);search.register(identifier)
    result=search.search(unit(np.ones(512)),identifier)
    assert q.queries==['synthetic','self-test']
    assert result['total']==109620 and result['self_is_first']
    assert result['candidates'][0]['label']=='Self fixture'
    assert len(result['candidates'])==5
    assert q.writes==['self-test']
    q.synthetic_score=.9
    assert not search.search(unit(np.ones(512)),identifier)['self_is_first']
    search.delete(identifier)
    assert not gallery.list() and not q.points['self-test']
    assert all(name=='self-test' for name in q.writes)


def test_only_one_self_and_explicit_consent(tmp_path):
    gallery,search,q=setup(tmp_path)
    with pytest.raises(ValueError,match='consent'):gallery.add('Me',[],False)
    enroll(gallery)
    with pytest.raises(ValueError,match='1 tester'):enroll(gallery)


def test_no_self_returns_only_fictional_candidates(tmp_path):
    gallery,search,q=setup(tmp_path)
    result=search.search(unit(np.ones(512)),None)
    assert result['self_score'] is None and not result['self_is_first']
    assert all(c['kind']=='synthetic' for c in result['candidates'])
    assert q.writes==[]


def test_mismatched_template_fails_closed(tmp_path):
    gallery,search,q=setup(tmp_path)
    identifier=enroll(gallery);search.register(identifier)
    q.points['self-test'][0].payload['model_id']='incompatible'
    with pytest.raises(ValueError,match='out of sync'):
        search.search(unit(np.ones(512)),identifier)


def test_votes_require_rank_threshold_and_current_pass():
    v=SearchEvidence()
    for i in range(5):v.add(i*.4,{'self_score':.7,'self_is_first':True},.6)
    assert v.summary(1.6)['state']=='Match'
    v.add(2,{'self_score':.9,'self_is_first':False},.6)
    assert v.summary(2)['state']=='No match'
    assert v.summary(2)['votes']==4
    v.clear()
    for i in range(5):v.add(i*.4,{'self_score':.7,'self_is_first':True},.8)
    assert v.summary(1.6)['state']=='No match'
    assert v.summary(20)['count']==0


def test_session_face_absence_and_threshold_reset(tmp_path):
    gallery,search,q=setup(tmp_path)
    identifier=enroll(gallery);search.register(identifier)
    session=SelfSession(search,dispatch=lambda fn:fn());engine=FakeEngine();start=time.monotonic()
    session.configure({'selected':identifier,'minimum_width':70,'minimum_sharpness':45,'threshold':.6},gallery)
    for i in range(51):r=session.process(engine,FRAME,gallery,start+i*.1)
    assert r['verification']['state']=='Match' and r['self_search']['total']==109620
    before=len(q.queries)
    session.process(engine,FRAME,gallery,start+5.1)
    assert len(q.queries)==before
    head=r['active_id'];engine.faces=[]
    r=session.process(engine,FRAME,gallery,start+5.2)
    assert r['active_id']==head and r['self_search'] is None
    assert r['verification']['state']=='Head tracked / face unavailable'
    assert not session.search_evidence.samples
    engine.faces=[face()]
    for i in range(53,110):r=session.process(engine,FRAME,gallery,start+i*.1)
    assert r['verification']['state']=='Match'
    session.configure({'selected':identifier,'minimum_width':70,'minimum_sharpness':45,'threshold':.95},gallery)
    assert session.last_search is None and not session.search_evidence.samples
    for i in range(110,165):r=session.process(engine,FRAME,gallery,start+i*.1)
    assert r['verification']['state']=='No match'


def test_query_failure_never_falls_back_to_1to1(tmp_path):
    gallery,search,q=setup(tmp_path)
    identifier=enroll(gallery);search.register(identifier)
    session=SelfSession(search,dispatch=lambda fn:fn());engine=FakeEngine();session.selected=identifier
    def failed(*args):raise RuntimeError('Qdrant unavailable')
    search.search=failed
    start=time.monotonic()
    for i in range(8):r=session.process(engine,FRAME,gallery,start+i*.4)
    assert r['verification']['state']=='Search unavailable'
    assert r['self_search'] is None and not session.search_evidence.samples


class FastFakeEngine(FakeEngine):
    def __init__(self):self.full_calls=0;self.head_calls=0
    def detect_roi(self,frame):
        self.full_calls+=1
        return self.detect(frame)
    def detect_heads(self,frame):
        self.head_calls+=1
        return {'heads':self.heads,'faces':[],'bodies':[],'face_checked':False}


def test_every_frame_detects_faces_but_recognition_runs_once_a_second(tmp_path):
    gallery,search,q=setup(tmp_path);identifier=enroll(gallery);search.register(identifier)
    session=SelfSession(search,dispatch=lambda fn:fn());engine=FastFakeEngine();session.selected=identifier
    start=time.monotonic()
    for i in range(51):result=session.process(engine,FRAME,gallery,start+i*.1,frame_id=i)
    assert result['verification']['state']=='Match'
    assert engine.head_calls==0 and engine.full_calls==51
    assert result['face_checked'] and result['frame_id']==50
    assert session.recognition_completed<=6
    assert result['timing']['embedding_ms']==0
    head=result['active_id']
    engine.faces=[]
    result=session.process(engine,FRAME,gallery,start+5.1)
    assert result['active_id']==head and result['verification']['state']!='Match'
    assert result['self_search'] is None
    assert not session.search_evidence.samples


def test_async_recognition_never_bridges_multiple_heads(tmp_path):
    gallery,search,q=setup(tmp_path);identifier=enroll(gallery);search.register(identifier)
    session=SelfSession(search,dispatch=lambda fn:fn());engine=FastFakeEngine();session.selected=identifier
    start=time.monotonic()
    for i in range(51):session.process(engine,FRAME,gallery,start+i*.1)
    engine.heads=np.array([[165,90,355,310,.95],[400,90,580,310,.95]],np.float32)
    result=session.process(engine,FRAME,gallery,start+5.1)
    assert result['verification']['state']!='Match' and result['self_search'] is None


def test_threshold_has_disjoint_validation_and_no_guarantee():
    scores=np.linspace(-.2,.4,1000)
    result=suggest_threshold(scores)
    assert result==suggest_threshold(scores)
    assert result['calibration_count']==800 and result['validation_count']==200
    assert result['threshold']>=.45 and result['validation_exceedances']==0
    assert 'No held-out genuine' in result['limitations']
    with pytest.raises(ValueError):suggest_threshold([np.nan]*100)
    with pytest.raises(ValueError):suggest_threshold([.2]*5)


class BlockingEncoder(FastFakeEngine):
    def __init__(self):
        super().__init__();self.entered=threading.Event();self.release=threading.Event()
    def encode_crop(self,crop):
        self.entered.set()
        assert self.release.wait(3),'test did not release recognizer'
        return self.feature


def wait_idle(session):
    deadline=time.monotonic()+3
    while session._worker_busy and time.monotonic()<deadline:time.sleep(.01)
    assert not session._worker_busy


@pytest.mark.parametrize('invalidate',['face_loss','reset','close','threshold'])
def test_slow_encoder_does_not_block_tracking_or_publish_stale_results(tmp_path,invalidate):
    gallery,search,q=setup(tmp_path);identifier=enroll(gallery);search.register(identifier)
    session=SelfSession(search);session.selected=identifier;engine=BlockingEncoder()
    now=time.monotonic();started=time.perf_counter()
    first=session.process(engine,FRAME,gallery,now,frame_id=1)
    assert time.perf_counter()-started<.4
    assert engine.entered.wait(1)
    for i in range(2,7):
        result=session.process(engine,FRAME,gallery,now+i*.05,frame_id=i)
        assert result['frame_id']==i and result['face_checked']
    assert result['recognition']['queued_crops']==1
    assert q.queries==[]
    if invalidate=='face_loss':
        engine.faces=[];session.process(engine,FRAME,gallery,now+.4)
    elif invalidate=='reset':session.reset()
    elif invalidate=='close':session.close()
    else:session.configure({'selected':identifier,'minimum_width':70,'minimum_sharpness':45,'threshold':.9},gallery)
    engine.release.set();wait_idle(session)
    assert session.last_search is None and not session.search_evidence.samples


def test_background_completion_publishes_timing_and_gallery_size(tmp_path):
    gallery,search,q=setup(tmp_path);identifier=enroll(gallery);search.register(identifier)
    session=SelfSession(search);session.selected=identifier;engine=BlockingEncoder();now=time.monotonic()
    first=session.process(engine,FRAME,gallery,now,frame_id=10)
    assert engine.entered.wait(1);engine.release.set();wait_idle(session)
    second=session.process(engine,FRAME,gallery,now+.2,frame_id=11)
    assert second['self_search']['source_frame_id']==10
    assert second['recognition']['gallery_size']==109620
    assert second['recognition']['completed']==1 and second['recognition']['candidates_returned']==5
    assert second['recognition']['latest_total_ms']>=second['recognition']['latest_encoding_ms']>=0
    assert second['recognition']['vectors_examined'] is None
    session.close()


def test_face_loss_does_not_bypass_one_hz_limit(tmp_path):
    gallery,search,q=setup(tmp_path);identifier=enroll(gallery);search.register(identifier)
    session=SelfSession(search,dispatch=lambda fn:fn());session.selected=identifier;engine=FakeEngine()
    now=time.monotonic();session.process(engine,FRAME,gallery,now)
    assert session.recognition_completed==1
    engine.faces=[];session.process(engine,FRAME,gallery,now+.1)
    engine.faces=[face()];session.process(engine,FRAME,gallery,now+.2)
    assert session.recognition_completed==1
    session.process(engine,FRAME,gallery,now+1.01)
    assert session.recognition_completed==2
