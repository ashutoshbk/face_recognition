import os
import threading
import time

import numpy as np

from recognition_process import ProcessRecognizer


def echo_worker(root,data,collection,requests,results):
    results.put({'kind':'ready','pid':os.getpid()})
    while True:
        job=requests.get()
        if job is None:return
        if job['selected']=='crash':os._exit(9)
        time.sleep(.3)
        results.put({'kind':'result','token':job['token'],'worker_pid':os.getpid()})


def wait_ready(worker):
    end=time.monotonic()+5
    while worker.status()['state']!='ready' and time.monotonic()<end:time.sleep(.01)
    assert worker.status()['state']=='ready'


def test_process_isolation_and_bounded_nonblocking_submission(tmp_path):
    worker=ProcessRecognizer(tmp_path,tmp_path,'test',target=echo_worker)
    try:
        worker.start();wait_ready(worker)
        assert worker.status()['pid']!=os.getpid()
        done=threading.Event();received=[]
        def completed(packet):received.append(packet);done.set()
        started=time.perf_counter()
        assert worker.try_submit(np.zeros((112,112,3),np.uint8),'fixture',completed)
        assert time.perf_counter()-started<.1
        assert not worker.try_submit(np.zeros((112,112,3),np.uint8),'fixture',completed)
        assert done.wait(3)
        assert received[0]['worker_pid']==worker.status()['pid']
        assert not worker.status()['in_flight']
    finally:worker.close()


def test_worker_failure_reports_error_without_hanging_parent(tmp_path):
    worker=ProcessRecognizer(tmp_path,tmp_path,'test',target=echo_worker)
    try:
        worker.start();wait_ready(worker)
        done=threading.Event();received=[]
        def completed(packet):received.append(packet);done.set()
        assert worker.try_submit(np.zeros((112,112,3),np.uint8),'crash',completed)
        assert done.wait(3)
        assert 'error' in received[0] and worker.status()['state']=='error'
        assert not worker.try_submit(np.zeros((112,112,3),np.uint8),'fixture',completed)
    finally:worker.close()
