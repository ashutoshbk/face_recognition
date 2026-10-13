"""Local single-self-template experiment with a read-only synthetic gallery."""
import json
import hashlib
import os
import sqlite3
import threading
import time
from collections import deque
from functools import lru_cache
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from qdrant_client import QdrantClient, models

from core import Session, MODEL_ID, unit, align_face, jpeg, data_url
from settings import RESULT,GALLERY_DB,DATASET,QDRANT_PORT,QDRANT_GRPC_PORT

SELF_POINT = '6290a7d8-6a36-4b56-b2ab-0fce48751585'
SELF_COLLECTION = 'local_single_self_test_arcface_r50_v1'


class FrameEngine:
    def __init__(self,engine):
        self.engine=engine

    def detect(self,frame):
        return self.engine.detect_roi(frame) if hasattr(self.engine,'detect_roi') else self.engine.detect(frame)

    def encode(self,frame,face):
        raise RuntimeError('Face encoding must not run in the foreground tracking path')


def suggest_threshold(scores, seed=20261013):
    scores=np.asarray(scores,dtype=np.float32)
    if len(scores)<100 or not np.isfinite(scores).all():
        raise ValueError('At least 100 finite synthetic comparisons are required')
    order=np.random.default_rng(seed).permutation(len(scores))
    cut=int(len(scores)*.8)
    calibration=scores[order[:cut]];validation=scores[order[cut:]]
    proposal=min(1.,max(.45,float(np.ceil((float(calibration.max())+.05)*100)/100)))
    return {'threshold':proposal,'calibration_count':len(calibration),'validation_count':len(validation),
        'calibration_max':float(calibration.max()),'validation_max':float(validation.max()),
        'calibration_exceedances':int((calibration>=proposal).sum()),
        'validation_exceedances':int((validation>=proposal).sum()),
        'method':'Deterministic 80/20 identity split; calibration maximum + 0.05, rounded up to 0.01; floor 0.45.',
        'limitations':'Synthetic negatives only. No held-out genuine samples, real-person false-positive rate or zero-error guarantee.'}


class SelfSearch:
    def __init__(self, root, gallery, personal_collection=SELF_COLLECTION):
        self.root=Path(root)
        self.gallery=gallery
        self.lock=threading.RLock()
        self.threshold_lock=threading.Lock()
        self.client=None
        self.collection=None
        self.synthetic_count=0
        self.personal_collection=personal_collection

    def connect(self):
        if self.client is None:
            self.client=QdrantClient(host='127.0.0.1',port=QDRANT_PORT,grpc_port=QDRANT_GRPC_PORT,prefer_grpc=True,timeout=10)
        if self.collection is None:
            result=json.loads(RESULT.read_text())
            if result['dimensions']!=512 or result['gallery_vectors']<=0:
                raise ValueError('A completed synthetic gallery is required')
            if result['model_hashes']['arcface-r50-w600k-112x112.onnx']!='4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43':
                raise ValueError('Incompatible synthetic template model')
            self.collection=result['collection']
            self.synthetic_count=result['gallery_vectors']
            if self.personal_collection==self.collection:
                self.collection=None
                raise ValueError('Self templates cannot be written to the benchmark collection')
        return self.client

    def status(self):
        with self.lock:
            try:
                q=self.connect()
                count=q.count(self.collection,exact=True).count
                if count!=self.synthetic_count:
                    raise ValueError('Synthetic gallery size changed')
                testers=self.gallery.list()
                return {'ready':True,'synthetic_count':count,'self_count':len(testers),
                    'total':count+len(testers),'scope':'One self-enrolled tester plus fictional distractors',
                    'source_collection':self.collection,'self_collection':self.personal_collection,
                    'model_id':MODEL_ID}
            except Exception as error:
                return {'ready':False,'error':str(error)}

    def register(self, identifier):
        with self.lock:
            q=self.connect()
            if len(self.gallery.list())!=1:
                raise ValueError('This research mode accepts exactly one self-enrolled tester')
            record=self.gallery.get(identifier)
            if not q.collection_exists(self.personal_collection):
                q.create_collection(self.personal_collection,vectors_config=models.VectorParams(size=512,distance=models.Distance.COSINE))
            q.upsert(self.personal_collection,points=[models.PointStruct(id=SELF_POINT,vector=record['template'].tolist(),
                payload={'kind':'self','tester_id':identifier,'model_id':MODEL_ID})],wait=True)

    def delete(self, identifier):
        with self.lock:
            self.gallery.get(identifier)
            q=self.connect()
            if q.collection_exists(self.personal_collection):
                q.delete(self.personal_collection,points_selector=models.PointIdsList(points=[SELF_POINT]),wait=True)
            self.gallery.delete(identifier)
            (self.gallery.path.parent/'threshold_analysis.json').unlink(missing_ok=True)

    def search(self, feature, identifier):
        vector=unit(feature)
        if vector.size!=512:
            raise ValueError('Expected a 512D ArcFace descriptor')
        with self.lock:
            q=self.connect()
            started=time.perf_counter()
            synthetic=q.query_points(self.collection,query=vector.tolist(),limit=5,
                search_params=models.SearchParams(hnsw_ef=256),with_payload=False,with_vectors=False).points
            candidates=[{'id':str(p.id),'kind':'synthetic','label':f'Synthetic {p.id}',
                'score':max(-1.,min(1.,float(p.score))), 'photo_url':f'/api/self-test/synthetic/{p.id}/photo'} for p in synthetic]
            self_score=None
            self_label=None
            if identifier:
                record=self.gallery.get(identifier)
                if not q.collection_exists(self.personal_collection):
                    raise ValueError('Self template is missing from Qdrant; re-enroll')
                own=q.query_points(self.personal_collection,query=vector.tolist(),limit=1,
                    with_payload=True,with_vectors=False).points
                if len(own)!=1 or own[0].payload.get('tester_id')!=identifier or own[0].payload.get('model_id')!=MODEL_ID:
                    raise ValueError('Self template is out of sync; delete it and re-enroll')
                self_score=max(-1.,min(1.,float(own[0].score)))
                self_label=record['name']
                candidates.append({'id':identifier,'kind':'self','label':self_label,'score':self_score,
                    'photo_url':f'/api/testers/{identifier}/photo?mode=self'})
            elapsed=(time.perf_counter()-started)*1000
            # Rank the union. The synthetic search is approximate HNSW; the self collection is exact.
            candidates.sort(key=lambda c:(-c['score'],c['kind']=='self',c['id']))
            top=candidates[:5]
            lead=(self_score-max((c['score'] for c in candidates if c['kind']=='synthetic'),default=-1)) if self_score is not None else None
            return {'candidates':top,'self_score':self_score,'self_label':self_label,
                'self_is_first':bool(top and top[0]['kind']=='self'), 'lead':lead,
                'search_ms':elapsed,'synthetic_count':self.synthetic_count,'total':self.synthetic_count+bool(identifier),
                'source':'Qdrant synthetic HNSW top 5 + exact single self template; combined ranking'}

    def threshold_analysis(self):
        with self.threshold_lock:
            return self._threshold_analysis()

    def _threshold_analysis(self):
        records=self.gallery.list()
        if len(records)!=1:
            raise ValueError('Enroll one current ArcFace self profile first; an older SFace template cannot be used')
        record=self.gallery.get(records[0]['id'])
        signature=hashlib.sha256(record['template'].tobytes()).hexdigest()
        destination=self.gallery.path.parent/'threshold_analysis.json'
        if destination.exists():
            previous=json.loads(destination.read_text())
            if previous.get('template_sha256')==signature and previous.get('tester_id')==record['id']:
                return previous
        path=GALLERY_DB
        scores=[]
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
            cursor=db.execute("SELECT vector FROM samples WHERE role='gallery' AND vector IS NOT NULL ORDER BY identity")
            while rows:=cursor.fetchmany(2048):
                matrix=np.stack([np.frombuffer(row[0],np.float32) for row in rows])
                scores.extend(matrix@record['template'])
        result=suggest_threshold(scores)
        result.update(tester_id=record['id'],name=record['name'],model_id=MODEL_ID,
            template_sha256=signature,synthetic_comparisons=len(scores),timestamp=time.time())
        if not any(r['id']==record['id'] for r in self.gallery.list()):
            raise ValueError('Profile was removed during analysis')
        temporary=destination.with_suffix('.tmp')
        temporary.write_text(json.dumps(result,indent=2)+'\n');temporary.chmod(0o600);temporary.replace(destination)
        return result

    @lru_cache(maxsize=128)
    def synthetic_photo(self, identifier):
        if not isinstance(identifier,int) or identifier<0:
            raise ValueError('Invalid synthetic identity')
        path=GALLERY_DB
        with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
            row=db.execute("SELECT archive,member FROM samples WHERE identity=? AND role='gallery' AND vector IS NOT NULL",(identifier,)).fetchone()
        if not row:
            raise ValueError('Synthetic identity not enrolled')
        with ZipFile(DATASET/row[0]) as archive:
            return archive.read(row[1])


class SearchEvidence:
    def __init__(self):
        self.samples=deque(maxlen=5)

    def clear(self):
        self.samples.clear()

    def add(self,now,result,threshold):
        if self.samples and now-self.samples[-1]['time']<.35:
            return
        self.samples.append({'time':now,'score':result['self_score'],
            'pass':bool(result['self_is_first'] and result['self_score'] is not None and result['self_score']>=threshold)})

    def summary(self,now):
        while self.samples and now-self.samples[0]['time']>8:
            self.samples.popleft()
        values=list(self.samples)
        votes=sum(s['pass'] for s in values)
        ready=len(values)==5 and values[-1]['time']-values[0]['time']>=1.3
        state='Collecting evidence'
        if ready:
            state='Match' if votes>=3 and values[-1]['pass'] else 'No match'
        return {'state':state,'votes':votes,'count':len(values),'required':3,'window':5,
            'scores':[s['score'] if s['score'] is not None else -1 for s in values],
            'vote_passes':[s['pass'] for s in values]}


class SelfSession(Session):
    mode='self'

    def __init__(self,search,report_directory=None,dispatch=None,recognizer=None):
        self.search=search
        self.recognizer=recognizer
        self._mailbox=deque(maxlen=1)
        self._mailbox_lock=threading.Lock()
        self._epoch=0
        self._worker_busy=False
        self._closed=False
        self._next_recognition_at=0.0
        self._dispatch=dispatch or (lambda task: threading.Thread(target=task,daemon=True).start())
        self.recognition_interval=1.0
        self.recognition_ttl=2.5
        self.recognition_completed=0
        super().__init__(report_directory)

    def reset(self):
        super().reset()
        self.search_evidence=SearchEvidence()
        self.search_events=deque(maxlen=200)
        self._invalidate_recognition()

    def _invalidate_recognition(self):
        self._epoch+=1
        self.last_search=None
        self._pending=None
        self._usable_now=False
        self._search_error=None
        if hasattr(self,'search_evidence'):
            self.search_evidence.clear()

    def clear_capture(self):
        super().clear_capture()
        self._invalidate_recognition()

    def close(self):
        with self.lock:
            self._closed=True
            self.clear_capture()

    def _current_job(self,job):
        return (not self._closed and job['epoch']==self._epoch and self._usable_now
            and self.active_id==job['track_id'] and self.selected==job['selected']
            and self.last_used-job['timestamp']<=self.recognition_ttl)

    def _recognize(self,engine,job):
        try:
            with self.lock:
                if not self._current_job(job):
                    return
            started=time.perf_counter()
            feature=engine.encode_crop(job['crop'])
            encoding_ms=(time.perf_counter()-started)*1000
            with self.lock:
                if not self._current_job(job):
                    return
            found=self.search.search(feature,job['selected'])
            recognition_ms=(time.perf_counter()-started)*1000
            photo=jpeg(job['crop'])
            completed=time.monotonic()
            with self.lock:
                self._accept_result(job,{'feature':feature,'found':found,'photo':photo,'encoding_ms':encoding_ms,
                    'recognition_ms':recognition_ms,'completed_at':completed,'worker_pid':os.getpid()})
        except Exception as error:
            with self.lock:
                if self._current_job(job):
                    self.last_search=None
                    self.search_evidence.clear()
                    self._search_error=str(error)
        finally:
            with self.lock:
                self._worker_busy=False

    def _accept_result(self,job,packet):
        if not self._current_job(job):return
        if packet.get('error'):
            self.last_search=None;self.search_evidence.clear();self._search_error=packet['error'];return
        feature,found=packet['feature'],packet['found']
        if self.last_feature is not None and float(np.dot(feature,self.last_feature))<.35:
            self.candidates.clear();self.search_evidence.clear();self._pending=None
        self.last_feature=feature;self.last_sample=job['timestamp']
        sample={'timestamp':job['timestamp'],'quality':job['quality'],'feature':feature,'jpeg':packet['photo']}
        self.candidates=sorted(self.candidates+[sample],key=lambda c:c['quality'],reverse=True)[:5]
        self.recognition_completed+=1
        self.last_search={**found,'encoding_ms':packet['encoding_ms'],'recognition_ms':packet['recognition_ms'],
            'sample_timestamp':job['timestamp'],'completed_at':packet['completed_at'],
            'source_frame_id':job['frame_id'],'source_sequence':job['sequence'],
            'completion_sequence':self.recognition_completed,'candidates_returned':len(found['candidates']),
            'vectors_examined':None,'worker_pid':packet.get('worker_pid')}
        self._search_error=None
        self.search_evidence.add(job['timestamp'],found,job['threshold'])

    def _receive(self,job,packet):
        # Completion never waits for the tracking/session lock or does image work.
        with self._mailbox_lock:self._mailbox.append((job,packet))

    def _drain_results(self):
        with self._mailbox_lock:
            item=self._mailbox.popleft() if self._mailbox else None
        if item:
            self._worker_busy=False
            self._accept_result(*item)

    def preview_snapshot(self,epoch,sequence):
        with self.lock:
            if epoch!=self._epoch or not self.last_search or sequence!=self.last_search['completion_sequence']:
                raise ValueError('Recognition result changed')
            images=self.recognizer.preview_images(self.last_search['candidates']) if self.recognizer else {}
            return {'epoch':epoch,'sequence':sequence,'images':images}

    def _offer_crop(self,engine,frame,result,now):
        quality=result['quality']['score']
        if (self._pending is None or now-self._pending['timestamp']>.75
                or quality>=self._pending['quality']):
            self._pending={'crop':align_face(frame,self.current_face),'quality':quality,
                'timestamp':now,'track_id':self.active_id,'selected':self.selected,'threshold':self.threshold,
                'epoch':self._epoch,'frame_id':result['frame_id'],'sequence':result['sequence']}
        if self._worker_busy or now<self._next_recognition_at:
            return
        job=self._pending
        self._pending=None
        self._worker_busy=True
        try:
            if self.recognizer:
                if not self.recognizer.try_submit(job['crop'],job['selected'],lambda packet:self._receive(job,packet)):
                    self._worker_busy=False;self._pending=job;return
            else:
                self._dispatch(lambda:self._recognize(engine,job))
            self._next_recognition_at=now+self.recognition_interval
        except Exception:
            self._worker_busy=False
            raise

    def process(self,engine,frame,gallery,now=None,frame_id=None):
        now=time.monotonic() if now is None else now
        with self.lock:
            if self._closed:
                raise ValueError('Session closed; restart the camera')
            started=time.perf_counter()
            result=super().process(FrameEngine(engine),frame,gallery,now,encode_samples=False,frame_id=frame_id)
            result['mode']='self'
            result['face_checked']=True
            result['face_sample_age_ms']=0
            usable=bool(result['active_id'] and result['quality'] and result['quality']['accepted'] and result['face_associated'])
            if not usable:
                self._invalidate_recognition()
            else:
                self._usable_now=True
                if self.last_search and now-self.last_search['sample_timestamp']>self.recognition_ttl:
                    self.last_search=None
                    self.search_evidence.clear()
            self._drain_results()
            if usable:
                self._offer_crop(engine,frame,result,now)
                self._drain_results()
            if usable:
                result['verification']=self.search_evidence.summary(now)
                if self._search_error:
                    result['verification']['state']='Search unavailable'
                    result['search_error']=self._search_error
                elif not self.selected:
                    result['verification']['state']='No self profile enrolled'
            result['self_search']={**self.last_search,'sample_age_ms':max(0,round((now-self.last_search['sample_timestamp'])*1000))} if self.last_search else None
            result['samples']=[{'quality':c['quality'],'age_seconds':round(now-c['timestamp'],1),'image':data_url(c['jpeg'])} for c in self.candidates]
            times=[c['timestamp'] for c in self.candidates]
            result['enrollment_ready']=bool(usable and len(times)>=3 and max(times)-min(times)>=.65)
            result['timing']['comparison_ms']=self.last_search['search_ms'] if self.last_search else 0
            result['timing']['embedding_ms']=0.0
            result['recognition_interval_ms']=round(self.recognition_interval*1000)
            worker=self.recognizer.status() if self.recognizer else {'state':'ready','pid':os.getpid(),'kind':'thread-test'}
            result['recognition']={'in_flight':self._worker_busy,'queued_crops':int(self._pending is not None),
                'completed':self.recognition_completed,'interval_ms':1000,
                'epoch':self._epoch,'worker_pid':worker['pid'],'tracking_pid':os.getpid(),'pipeline':worker['kind'],
                'state':'No usable face' if not usable else 'Worker '+worker['state'] if worker['state']!='ready' else 'Search unavailable' if self._search_error else 'Running' if self._worker_busy else 'Waiting',
                'latest_encoding_ms':self.last_search['encoding_ms'] if self.last_search else None,
                'latest_search_ms':self.last_search['search_ms'] if self.last_search else None,
                'latest_total_ms':self.last_search['recognition_ms'] if self.last_search else None,
                'gallery_size':self.last_search['total'] if self.last_search else (getattr(self.search,'synthetic_count',0)+bool(self.selected) if getattr(self.search,'synthetic_count',0) else None),
                'candidates_returned':self.last_search['candidates_returned'] if self.last_search else 0,
                'vectors_examined':None}
            result['timing']['total_ms']=round((time.perf_counter()-started)*1000,2)
            event={'time':time.time(),'track':result['active_id'],
                'state':result['verification']['state'],'mode':'self-plus-synthetic',
                'search_ms':result['timing']['comparison_ms']}
            if not self.search_events or self.search_events[-1]['state']!=event['state']:
                self.search_events.append(event)
            self.events=deque(self.search_events,maxlen=200)
            self.last_frame=result
            return result

    def enroll(self,name,consent,track_id,gallery):
        with self.lock:
            identifier=super().enroll(name,consent,track_id,gallery)
            try:
                self.search.register(identifier)
            except Exception:
                try:
                    self.search.delete(identifier)
                finally:
                    self.selected=None
                    self.clear_capture()
                raise
            self.clear_capture()
            return identifier
