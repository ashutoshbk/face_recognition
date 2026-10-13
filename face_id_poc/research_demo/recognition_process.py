"""Bounded recognition process and independent thumbnail preparation."""
import base64
import multiprocessing
import os
import queue
import sqlite3
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from zipfile import ZipFile
from settings import DATASET,GALLERY_DB,MODELS,RESULT


class PreviewLoader:
    def __init__(self,root,gallery,output):
        self.root,self.gallery,self.output=Path(root),gallery,output
        self.condition=threading.Condition()
        self.pending=None
        self.stopped=False
        self.archives=OrderedDict()
        self.cache=OrderedDict()
        self.thread=threading.Thread(target=self._run,daemon=True)
        self.thread.start()

    def request(self,candidates):
        with self.condition:
            self.pending=candidates
            self.condition.notify()

    def _image(self,candidate,db):
        key=f"{candidate['kind']}:{candidate['id']}"
        if key in self.cache:
            self.cache.move_to_end(key)
            return key,self.cache[key]
        if candidate['kind']=='self':
            image=self.gallery.get(candidate['id'])['photo'];mime='image/jpeg'
        else:
            row=db.execute("SELECT archive,member FROM samples WHERE identity=? AND role='gallery' AND vector IS NOT NULL",(int(candidate['id']),)).fetchone()
            if not row:raise ValueError('Synthetic reference unavailable')
            if row[0] not in self.archives:
                while len(self.archives)>=2:
                    self.archives.popitem(last=False)[1].close()
                self.archives[row[0]]=ZipFile(DATASET/row[0])
            self.archives.move_to_end(row[0])
            image=self.archives[row[0]].read(row[1]);mime='image/png'
        value=f'data:{mime};base64,'+base64.b64encode(image).decode('ascii')
        self.cache[key]=value
        while len(self.cache)>256:self.cache.popitem(last=False)
        return key,value

    def _run(self):
        db=None
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda:self.pending is not None or self.stopped)
                    if self.stopped:return
                    candidates=self.pending;self.pending=None
                for candidate in candidates:
                    if self.stopped:return
                    try:
                        if db is None:db=sqlite3.connect(f"file:{GALLERY_DB}?mode=ro",uri=True)
                        key,image=self._image(candidate,db)
                        self.output.put({'kind':'preview','key':key,'image':image},timeout=1)
                    except Exception:
                        # Preview failure never invalidates a recognition result.
                        continue
        finally:
            if db is not None:db.close()
            for archive in self.archives.values():archive.close()

    def close(self):
        with self.condition:
            self.stopped=True;self.condition.notify()
        self.thread.join(timeout=2)


def recognition_worker(root,data,collection,requests,results):
    # This process owns ArcFace, Qdrant lookup and all ZIP parsing, never tracking.
    try:
        os.nice(5)
    except (OSError,AttributeError):pass
    os.environ['OMP_NUM_THREADS']='1'
    os.environ['OPENBLAS_NUM_THREADS']='1'
    try:
        import cv2
        import numpy as np
        from core import Gallery,jpeg
        from self_test import SelfSearch
        from vision.networks import session,sha256,MODEL_HASHES,encode_arcface_crop
        cv2.setNumThreads(1)
        root=Path(root)
        model=MODELS/'arcface-r50-w600k-112x112.onnx'
        if sha256(model)!=MODEL_HASHES[model.name]:raise ValueError('ArcFace checksum mismatch')
        rec=session(model,'CPUExecutionProvider',threads=1)
        encode_arcface_crop(rec,np.zeros((112,112,3),np.uint8))
        gallery=Gallery(Path(data),max_testers=1)
        search=SelfSearch(root,gallery,personal_collection=collection)
        if RESULT.exists():search.connect()
        previews=PreviewLoader(root,gallery,results)
        results.put({'kind':'ready','pid':os.getpid()})
        while True:
            job=requests.get()
            if job is None:break
            try:
                start=time.perf_counter()
                feature=encode_arcface_crop(rec,job['crop'])
                encoding_ms=(time.perf_counter()-start)*1000
                found=search.search(feature,job['selected'])
                total_ms=(time.perf_counter()-start)*1000
                packet={'kind':'result','token':job['token'],'feature':feature,'found':found,
                    'photo':jpeg(job['crop']),'encoding_ms':encoding_ms,'recognition_ms':total_ms,
                    'completed_at':time.monotonic(),'worker_pid':os.getpid()}
                results.put(packet)
                previews.request(found['candidates'])
            except Exception as error:
                results.put({'kind':'result','token':job['token'],'error':str(error),'worker_pid':os.getpid()})
        previews.close()
        if search.client is not None:search.client.close()
    except BaseException as error:
        results.put({'kind':'fatal','error':str(error),'pid':os.getpid()})


class ProcessRecognizer:
    def __init__(self,root,data,collection,target=recognition_worker):
        self.root,self.data,self.collection=str(root),str(data),collection
        self.target=target
        self.context=multiprocessing.get_context('spawn')
        self.requests=None
        self.results=None
        self.process=None
        self.listener=None
        self.worker_pid=None
        self.guard=threading.Lock()
        self.pending={}
        self.images=OrderedDict()
        self.state='stopped'
        self.error=None
        self.stopping=threading.Event()

    def start(self):
        if self.process is not None:return
        self.requests=self.context.Queue(maxsize=1)
        self.results=self.context.Queue(maxsize=8)
        self.state='warming'
        self.process=self.context.Process(target=self.target,args=(self.root,self.data,self.collection,self.requests,self.results),daemon=True)
        self.process.start()
        self.worker_pid=self.process.pid
        self.listener=threading.Thread(target=self._listen,daemon=True)
        self.listener.start()

    def status(self):
        with self.guard:
            return {'state':self.state,'pid':self.worker_pid,
                'in_flight':bool(self.pending),'error':self.error,'kind':'process','threads':1}

    def try_submit(self,crop,selected,callback):
        with self.guard:
            if self.state!='ready' or self.pending:return False
            token=uuid.uuid4().hex
            self.pending[token]=callback
            try:self.requests.put_nowait({'token':token,'crop':crop.copy(),'selected':selected})
            except queue.Full:
                self.pending.pop(token,None);return False
            return True

    def preview_images(self,candidates):
        with self.guard:
            return {f"{c['kind']}:{c['id']}":self.images[f"{c['kind']}:{c['id']}"]
                for c in candidates if f"{c['kind']}:{c['id']}" in self.images}

    def _fail(self,message):
        with self.guard:
            self.state='error';self.error=message
            callbacks=list(self.pending.values());self.pending.clear()
        for callback in callbacks:callback({'error':message})

    def _listen(self):
        while not self.stopping.is_set():
            try:packet=self.results.get(timeout=.2)
            except queue.Empty:
                if self.process and not self.process.is_alive():
                    self._fail('Recognition process stopped');return
                continue
            except (EOFError,OSError,ValueError):
                if not self.stopping.is_set():self._fail('Recognition channel closed')
                return
            if packet['kind']=='ready':
                with self.guard:self.state='ready'
            elif packet['kind']=='fatal':self._fail(packet['error'])
            elif packet['kind']=='preview':
                with self.guard:
                    self.images[packet['key']]=packet['image'];self.images.move_to_end(packet['key'])
                    while len(self.images)>256:self.images.popitem(last=False)
            elif packet['kind']=='result':
                with self.guard:callback=self.pending.pop(packet['token'],None)
                if callback:callback(packet)

    def close(self):
        if self.process:
            try:self.requests.put_nowait(None)
            except queue.Full:pass
            self.process.join(timeout=3)
            if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
        self.stopping.set()
        if self.listener:self.listener.join(timeout=1)
        self._fail('Recognition process closed')
        for channel in (self.requests,self.results):
            if channel is not None:
                channel.cancel_join_thread();channel.close()
        self.requests=None;self.results=None
        if self.process:self.process.close();self.process=None
