"""One volatile, authorized RTSP test feed per session; no recording."""
import multiprocessing
import os
import queue
import time
from urllib.parse import urlsplit

def validate_url(value):
    if not isinstance(value,str) or len(value)>2048 or any(ord(c)<32 for c in value):raise ValueError('Invalid RTSP address')
    parts=urlsplit(value)
    if parts.scheme not in ('rtsp','rtsps') or not parts.hostname:raise ValueError('An RTSP camera address with a host is required')
    try:parts.port
    except ValueError:raise ValueError('Invalid RTSP port')
    return value

def capture_worker(url,frames,stop):
    # Native decoder messages can contain credentials; suppress them in this child only.
    with open(os.devnull,'w') as sink:os.dup2(sink.fileno(),2)
    os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;tcp'
    import cv2
    cv2.setNumThreads(1)
    cap=cv2.VideoCapture(url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,10000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000])
    try:
        if not cap.isOpened():return
        sequence=0
        while not stop.is_set():
            ok,frame=cap.read()
            if not ok:break
            scale=min(1.,640/frame.shape[1]);frame=cv2.resize(frame,(round(frame.shape[1]*scale),round(frame.shape[0]*scale)))
            ok,data=cv2.imencode('.jpg',frame,[cv2.IMWRITE_JPEG_QUALITY,88])
            if not ok:continue
            sequence+=1;packet=(time.monotonic(),sequence,data.tobytes())
            try:frames.put_nowait(packet)
            except queue.Full:
                try:frames.get_nowait()
                except queue.Empty:pass
                try:frames.put_nowait(packet)
                except queue.Full:pass
    finally:cap.release()

class RTSPInput:
    def __init__(self,url,authorized):
        if not authorized:raise ValueError('Confirm authorization for this controlled test feed')
        url=validate_url(url)
        context=multiprocessing.get_context('spawn')
        self.frames=context.Queue(maxsize=1);self.stop=context.Event();self.latest=None
        self.process=context.Process(target=capture_worker,args=(url,self.frames,self.stop),daemon=True)
        self.process.start()
    def read(self):
        while True:
            try:self.latest=self.frames.get_nowait()
            except queue.Empty:break
        if self.latest is None or time.monotonic()-self.latest[0]>2:
            raise ValueError('No fresh RTSP frame. Check camera reachability, credentials and codec.')
        return self.latest[1],self.latest[2]
    def close(self):
        self.stop.set();self.process.join(timeout=.3)
        if self.process.is_alive():self.process.terminate();self.process.join(timeout=2)
        self.frames.cancel_join_thread();self.frames.close();self.latest=None
