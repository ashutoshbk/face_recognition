"""Own-process-only service management for macOS and Linux."""
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from settings import ROOT,HOME,DATA,WEB_PORT,GALLERY_PORT,QDRANT_PORT,QDRANT_GRPC_PORT,WEB_URL,GALLERY_URL,process_owned

STATE=HOME/'logs/processes.json'
def ready(url):
    try:
        with urllib.request.urlopen(url,timeout=2) as response:return response.status==200
    except Exception:return False
def occupied(port):
    with socket.socket() as sock:return sock.connect_ex(('127.0.0.1',port))==0
def owned(pid):return process_owned(pid,str(ROOT)) or process_owned(pid,'synthetic_bench.service:app')

def main():
    action=sys.argv[1] if len(sys.argv)>1 else 'start'
    STATE.parent.mkdir(parents=True,exist_ok=True)
    state=json.loads(STATE.read_text()) if STATE.exists() else {}
    if action=='status':
        print(json.dumps({k:{'pid':p,'running':owned(p)} for k,p in state.items()},indent=2));return
    if action=='stop':
        for name,pid in reversed(list(state.items())):
            if owned(pid):
                os.kill(pid,signal.SIGTERM)
                for _ in range(100):
                    if not owned(pid):break
                    time.sleep(.1)
                else:raise SystemExit(f'{name} is still stopping; state retained')
        STATE.unlink(missing_ok=True);return
    if action not in ('start','start-db'):raise SystemExit('Use start, start-db, stop or status')
    storage=DATA/'qdrant';storage.mkdir(parents=True,exist_ok=True)
    config=HOME/'qdrant.yaml'
    config.write_text(json.dumps({'storage':{'storage_path':str(storage)},'service':{'host':'127.0.0.1',
        'http_port':QDRANT_PORT,'grpc_port':QDRANT_GRPC_PORT,'enable_cors':False},'telemetry_disabled':True}))
    services=[('qdrant',QDRANT_PORT,[str(HOME/'vendor/qdrant'),'--config-path',str(config),'--disable-telemetry'],f'http://127.0.0.1:{QDRANT_PORT}/readyz'),
        ('app',WEB_PORT,[sys.executable,str(ROOT/'serve.py')],WEB_URL+'/api/health'),
        ('gallery',GALLERY_PORT,[sys.executable,'-m','uvicorn','synthetic_bench.service:app','--host','127.0.0.1','--port',str(GALLERY_PORT)],GALLERY_URL+'/api/status')]
    for name,port,args,url in services:
        if action=='start-db' and name!='qdrant':continue
        if occupied(port):
            if state.get(name) and owned(state[name]) and ready(url):continue
            raise SystemExit(f'Port {port} belongs to another process. Set CAMEX_* ports; nothing was stopped.')
        with (STATE.parent/f'{name}.log').open('ab') as log:
            env={**os.environ,'MPLCONFIGDIR':str(HOME/'cache/matplotlib'),'PYTHONDONTWRITEBYTECODE':'1'}
            process=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state[name]=process.pid;STATE.write_text(json.dumps(state))
        for _ in range(90):
            if ready(url):break
            if process.poll() is not None:raise SystemExit(f'{name} failed; see {STATE.parent/name}.log')
            time.sleep(1)
        else:raise SystemExit(f'{name} startup timed out')
    print(f'Qdrant ready on {QDRANT_PORT}' if action=='start-db' else f'Live: {WEB_URL}/?mode=self\nResearch: {GALLERY_URL}/')

if __name__=='__main__':main()
