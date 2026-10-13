"""Opt-in local RTSP integration test with synthetic footage and no real camera."""
import json
import os
import socket
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from settings import ROOT,HOME,WEB_URL

def call(path,payload=None,method=None,raw=None):
    data=raw if raw is not None else json.dumps(payload).encode() if payload is not None else None
    headers={'Content-Type':'image/jpeg' if raw is not None else 'application/json'}
    request=urllib.request.Request(WEB_URL+path,data=data,method=method or ('POST' if data is not None else 'GET'),headers=headers)
    with urllib.request.urlopen(request,timeout=15) as response:
        return response.read() if response.headers.get_content_type()=='image/jpeg' else json.load(response)

def main():
    port=18554
    with socket.socket() as sock:
        if sock.connect_ex(('127.0.0.1',port))==0:raise RuntimeError('Test RTSP port occupied; no existing server was changed')
    if call('/api/testers?mode=self'):raise RuntimeError('Use an empty test runtime, not a real enrolled profile')
    config=HOME/'rtsp-test.yaml';config.write_text(json.dumps({'logLevel':'warn','rtspAddress':f'127.0.0.1:{port}',
        'rtspTransports':['tcp'],'rtmp':False,'hls':False,'webrtc':False,'srt':False,'moq':False,'paths':{'all_others':{}}}))
    log=(HOME/'logs/rtsp-test.log').open('ab')
    relay=subprocess.Popen([os.environ.get('MEDIAMTX_BIN',str(HOME/'mediamtx')),str(config)],cwd=HOME,stdout=log,stderr=log)
    publisher=None;identifier=None;profile=None
    try:
        for attempt in range(50):
            if relay.poll() is not None:raise RuntimeError('Test relay did not start')
            with socket.socket() as sock:
                if sock.connect_ex(('127.0.0.1',port))==0:break
            time.sleep(.1)
        else:raise RuntimeError('Test relay did not become ready')
        url=f'rtsp://127.0.0.1:{port}/synthetic'
        publisher=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-re','-stream_loop','-1','-i',str(ROOT/'qa/synthetic_camera.y4m'),
            '-an','-c:v','libx264','-preset','ultrafast','-tune','zerolatency','-g','15','-pix_fmt','yuv420p','-f','rtsp','-rtsp_transport','tcp',url],stdout=log,stderr=log)
        time.sleep(3)
        if publisher.poll() is not None:raise RuntimeError('Synthetic RTSP publisher exited')
        os.environ['OPENCV_FFMPEG_CAPTURE_OPTIONS']='rtsp_transport;tcp'
        import cv2
        cap=cv2.VideoCapture(url,cv2.CAP_FFMPEG,[cv2.CAP_PROP_OPEN_TIMEOUT_MSEC,5000,cv2.CAP_PROP_READ_TIMEOUT_MSEC,5000])
        opened=cap.isOpened();ok,_=cap.read() if opened else (False,None);cap.release()
        print(json.dumps({'rtsp_decoder_open':opened,'frame_available':ok}),flush=True)
        if not ok:raise RuntimeError('Local OpenCV could not decode test RTSP feed')
        identifier=call('/api/sessions?mode=self',method='POST')['id']
        call(f'/api/sessions/{identifier}/rtsp',{'url':url,'authorized':True})
        result=None
        deadline=time.monotonic()+45
        frames=0
        while time.monotonic()<deadline:
            try:image=call(f'/api/sessions/{identifier}/rtsp-frame')
            except urllib.error.HTTPError as error:
                if error.code!=503:raise
                time.sleep(.2);continue
            result=call(f'/api/sessions/{identifier}/frame',raw=image);frames+=1
            if result['enrollment_ready']:break
            time.sleep(.08)
        if not result or not result['enrollment_ready']:
            (HOME/'rtsp_failure.json').write_text(json.dumps(result,indent=2))
            raise RuntimeError('RTSP face capture did not become ready; see private runtime diagnostic')
        profile=call(f'/api/sessions/{identifier}/enroll',{'name':'QA RTSP synthetic tester','consent':True,'track_id':result['active_id']})['id']
        deadline=time.monotonic()+30
        while time.monotonic()<deadline:
            image=call(f'/api/sessions/{identifier}/rtsp-frame')
            result=call(f'/api/sessions/{identifier}/frame',raw=image);frames+=1
            if result['verification']['state']=='Match':break
            time.sleep(.08)
        if result['verification']['state']!='Match':raise RuntimeError('RTSP recognition did not match the synthetic tester')
        publisher.terminate();publisher.wait(timeout=5);publisher=None
        stale=False
        for _ in range(40):
            try:call(f'/api/sessions/{identifier}/rtsp-frame')
            except urllib.error.HTTPError as error:
                if error.code==503:stale=True;break
                raise
            time.sleep(.1)
        if not stale:raise RuntimeError('Stream loss did not reject stale frames')
        report={'passed':True,'source':'Local synthetic RTSP H.264/TCP stream','frames_analyzed':frames,
            'registration_and_match':True,'stream_loss_rejected':True,'gallery_size':result['recognition']['gallery_size'],
            'tracking_pid':result['recognition']['tracking_pid'],'recognition_pid':result['recognition']['worker_pid'],
            'real_ip_camera_tested':False}
        (HOME/'rtsp_smoke_report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2))
    finally:
        if identifier:
            try:call(f'/api/sessions/{identifier}',method='DELETE')
            except Exception:pass
        if profile:
            try:call(f'/api/testers/{profile}?mode=self',method='DELETE')
            except Exception:pass
        for process in [publisher,relay]:
            if process and process.poll() is None:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
        log.close()

if __name__=='__main__':main()
