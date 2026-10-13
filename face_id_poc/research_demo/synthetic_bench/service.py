"""Loopback-only viewer for fixed synthetic benchmark runs."""

import json
import re
import signal
import sqlite3
import subprocess
import sys
import time
import zipfile
from functools import lru_cache

import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from qdrant_client import models
from starlette.middleware.trustedhost import TrustedHostMiddleware

from .pipeline import DATA, DATASET, ROOT, client
from settings import WEB_URL,HOME,process_owned

app = FastAPI(title='DigiFace research benchmark', docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware,allowed_hosts=['127.0.0.1','localhost','testserver'])
STATIC = ROOT/'synthetic_bench/static'


@app.middleware('http')
async def local_security(request:Request, call_next):
    origin = request.headers.get('origin')
    if origin and origin.rstrip('/') != str(request.base_url).rstrip('/'):
        return JSONResponse({'detail':'Cross-origin access is disabled'},status_code=403)
    response = await call_next(request)
    response.headers.update({'Cache-Control':'no-store','X-Content-Type-Options':'nosniff',
        'Content-Security-Policy':"default-src 'self'; img-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"})
    return response


def folder(name):
    if not re.fullmatch(r'[A-Za-z0-9_]{1,64}',name):
        raise HTTPException(400,'Invalid run name')
    path = DATA/name
    if not (path/'configuration.json').is_file():
        raise HTTPException(404,'Run not found')
    return path


def read_json(path, fallback=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return fallback


def alive(pid):
    if not isinstance(pid,int):
        return False
    return process_owned(pid, ' -m synthetic_bench.pipeline')


def state(name):
    path = folder(name)
    value = read_json(path/'status.json',{'status':'idle'})
    if value['status'] not in ('complete','error','paused','idle') and not alive(value.get('pid')):
        value.update(status='interrupted',error='Worker exited; completed batches are retained. Resume to continue.')
    value['result_available'] = (path/'result.json').is_file()
    if value['status']=='complete' and value['result_available']:
        result=read_json(path/'result.json')
        value.update(completed=result['identity_target']+result['probe_target'],
            probes=result['encoded_counts'].get('probe',0),rejected=len(result['rejected']))
    return value


@app.get('/')
def index():
    return FileResponse(STATIC/'index.html')


@app.get('/api/status')
def status():
    runs = [state(p.parent.name) for p in DATA.glob('*/configuration.json')]
    return {'scope':'Synthetic-only CPU research benchmark','runs':runs,
            'latest_completed':read_json(DATA/'latest_completed.json'), 'webcam_url':WEB_URL}


@app.get('/api/runs/{name}/result')
def result(name):
    path = folder(name)/'result.json'
    if not path.exists():
        raise HTTPException(409,'Run is not complete; no final benchmark available')
    return FileResponse(path,media_type='application/json',filename=f'{name}_benchmark.json')


@app.get('/api/runs/{name}/examples')
def examples(name):
    value = read_json(folder(name)/'result.json')
    if value is None:
        raise HTTPException(409,'Run is not complete')
    return {k:value[k] for k in ('gallery_vectors','queries','search_latency_ms','ann_recall_at10',
                               'synthetic_closed_set_top1','examples','embedding_ms','indexed_vectors','image_to_results_latency_ms')}


def connection(name):
    return sqlite3.connect(f'file:{folder(name)/"embeddings.sqlite3"}?mode=ro',uri=True)


@lru_cache(maxsize=256)
def image_bytes(archive,member):
    with zipfile.ZipFile(DATASET/archive) as z:
        return z.read(member)


@app.get('/api/runs/{name}/image/{role}/{identity}')
def sample_image(name,role:str,identity:int):
    if role not in ('gallery','probe'):
        raise HTTPException(400,'Invalid sample role')
    with connection(name) as db:
        row = db.execute('SELECT archive,member FROM samples WHERE identity=? AND role=? AND vector IS NOT NULL',(identity,role)).fetchone()
    if not row:
        raise HTTPException(404,'Synthetic image not indexed')
    return Response(image_bytes(*row),media_type='image/png')


@app.post('/api/runs/{name}/probe/{identity}')
def probe(name,identity:int):
    saved = read_json(folder(name)/'result.json')
    if not saved:
        raise HTTPException(409,'Wait for a completed run')
    with connection(name) as db:
        row = db.execute("SELECT vector FROM samples WHERE identity=? AND role='probe' AND vector IS NOT NULL",(identity,)).fetchone()
    if row is None:
        raise HTTPException(404,'Only predefined held-out synthetic probes are accepted')
    with client() as q:
        start = time.perf_counter()
        hits = q.query_points(saved['collection'],query=np.frombuffer(row[0],np.float32).tolist(),
            search_params=models.SearchParams(hnsw_ef=saved['hnsw_ef']),limit=5,with_payload=False,with_vectors=False).points
        latency = (time.perf_counter()-start)*1000
    return {'identity':identity,'search_ms':latency,'cached_embedding':True,
        'matches':[{'identity':p.id,'cosine':p.score} for p in hits],
        'correct_top1':bool(hits and hits[0].id==identity)}


@app.post('/api/build/start')
def start():
    for path in DATA.glob('*/status.json'):
        if alive(read_json(path,{}).get('pid')):
            raise HTTPException(409,'A synthetic build is already running')
    (HOME/'logs').mkdir(exist_ok=True)
    with (HOME/'logs/digiface_build.log').open('ab') as log:
        subprocess.Popen([sys.executable,'-u','-m','synthetic_bench.pipeline'],
            cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    return {'status':'starting','name':'all_identities'}


@app.post('/api/build/pause')
def pause():
    for path in DATA.glob('*/status.json'):
        value = read_json(path,{})
        if alive(value.get('pid')):
            import os
            os.kill(value['pid'],signal.SIGTERM)
    return {'status':'pause requested; current batch will finish'}


app.mount('/static',StaticFiles(directory=STATIC),name='static')
