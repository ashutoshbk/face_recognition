import os
import time
import base64
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from benchmark import Benchmark
from core import Engine, Gallery, Session, decode_image, MODEL_ID
from self_test import SelfSearch, SelfSession
from recognition_process import ProcessRecognizer
from rtsp_input import RTSPInput
from photo_enrollment import PhotoEnrollment
from settings import DATA,MODELS

ROOT = Path(__file__).resolve().parent

@asynccontextmanager
async def lifespan(app):
    recognizer.start()
    try:yield
    finally:
        for source in rtsp_sources.values():source.close()
        for item in sessions.values():
            if isinstance(item,SelfSession):item.close()
        recognizer.close()

app = FastAPI(title="Local Verification Lab", docs_url=None, redoc_url=None,lifespan=lifespan)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "testserver"])
engine = Engine(MODELS)
gallery = Gallery(DATA / "testers")
self_gallery = Gallery(DATA / 'self_tester', max_testers=1)
self_search = SelfSearch(ROOT, self_gallery, os.environ.get('LOCAL_SELF_TEST_COLLECTION','local_single_self_test_arcface_r50_v1'))
recognizer=ProcessRecognizer(ROOT,DATA/'self_tester',self_search.personal_collection)
photo_enrollment=PhotoEnrollment(engine,recognizer,self_search)
benchmark = Benchmark(DATA / "benchmarks")
sessions = {}
rtsp_sources = {}

def close_input(identifier):
    source=rtsp_sources.pop(identifier,None)
    if source:source.close()


@app.middleware("http")
async def local_security(request, call_next):
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        return JSONResponse({"detail": "Cross-origin access is disabled"}, status_code=403)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    return response


@app.exception_handler(ValueError)
async def invalid(request, error):
    return JSONResponse({"detail": str(error)}, status_code=400)


def session_for(identifier):
    session = sessions.get(identifier)
    if not session or time.monotonic() - session.last_used > 600:
        if session:
            close_input(identifier)
            if isinstance(session,SelfSession):session.close()
            del sessions[identifier]
        raise HTTPException(404, "Session expired. Restart the camera.")
    session.last_used = time.monotonic()
    return session


def gallery_for(session):
    return self_gallery if isinstance(session,SelfSession) else gallery


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")


@app.get("/api/health")
def health():
    return {'ready':True,'release':'2026-10-15-enrollment','model':'YOLOX + SCRFD + ArcFace','model_id':MODEL_ID,'dimensions':512,
            'tracker':'ByteTrack / head boxes','mode':'local selected-tester 1:1 verification',
            'tester_count':len(gallery.list()),'legacy_sface_testers':gallery.legacy_count(),
            'recognition_worker':recognizer.status(),'tracking_pid':os.getpid()}


@app.post('/api/sessions/{identifier}/turnaway')
def start_turnaway(identifier):
    return session_for(identifier).start_turnaway()


@app.get('/api/sessions/{identifier}/turnaway/export')
def export_turnaway(identifier):
    session = session_for(identifier)
    with session.lock:
        result = session.turnaway.snapshot()
        if result['status'] in ('idle','waiting_for_face_absence','waiting_for_face_return'):
            raise HTTPException(400,'No completed continuity check')
        return JSONResponse({'timestamp':time.time(),'model_id':MODEL_ID,**result},headers={'Content-Disposition':'attachment; filename=head_continuity_check.json'})


@app.post("/api/sessions")
def create_session(mode: Literal['verify','self']='verify'):
    for key in list(sessions):
        if time.monotonic() - sessions[key].last_used > 600:
            close_input(key)
            if isinstance(sessions[key],SelfSession):sessions[key].close()
            del sessions[key]
    if len(sessions) >= 8:
        raise HTTPException(429, "Too many active sessions")
    session = SelfSession(self_search,report_directory=DATA/'self_turnaway',recognizer=recognizer) if mode=='self' else Session(report_directory=DATA / 'turnaway')
    sessions[session.id] = session
    return {"id": session.id}


@app.post("/api/sessions/{identifier}/reset")
def reset(identifier):
    session = session_for(identifier)
    with session.lock:
        session.reset()
    return {"ok": True}


@app.delete("/api/sessions/{identifier}")
def delete_session(identifier):
    close_input(identifier)
    session=sessions.pop(identifier, None)
    if isinstance(session,SelfSession):session.close()
    return {"ok": True}


class StreamSettings(BaseModel):
    url: str = Field(min_length=8,max_length=2048)
    authorized: bool

@app.post('/api/sessions/{identifier}/rtsp')
def start_rtsp(identifier,form:StreamSettings):
    session_for(identifier)
    close_input(identifier)
    rtsp_sources[identifier]=RTSPInput(form.url,form.authorized)
    return {'status':'connecting'}

@app.get('/api/sessions/{identifier}/rtsp-frame')
def rtsp_frame(identifier):
    session=session_for(identifier)
    source=rtsp_sources.get(identifier)
    if not source:raise HTTPException(404,'No test stream connected')
    try:sequence,image=source.read()
    except ValueError:
        with session.lock:session.clear_capture('RTSP source has no fresh frame')
        raise HTTPException(503,'Waiting for a fresh RTSP frame; check reachability, credentials and codec')
    return Response(image,media_type='image/jpeg',headers={'X-Source-Frame':str(sequence)})


class Settings(BaseModel):
    selected: str | None = None
    minimum_width: int = Field(70, ge=50, le=180)
    minimum_sharpness: float = Field(45, ge=20, le=200)
    threshold: float = Field(0.45, ge=0.0, le=1.0)


@app.put("/api/sessions/{identifier}/settings")
def settings(identifier, settings: Settings):
    session=session_for(identifier)
    session.configure(settings.model_dump(), gallery_for(session))
    return {"ok": True}


@app.post("/api/sessions/{identifier}/frame")
async def frame(identifier, request: Request):
    session = session_for(identifier)
    frame_id=request.headers.get('x-frame-id')
    if frame_id is not None:
        try:frame_id=int(frame_id)
        except ValueError:raise HTTPException(400,'Invalid frame ID')
        if not 0<=frame_id<=9007199254740991:raise HTTPException(400,'Invalid frame ID')
    chunks = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > 2_000_000:
            raise HTTPException(413, "Frame exceeds 2 MB")
        chunks.append(chunk)
    image = await run_in_threadpool(decode_image, b"".join(chunks))
    return await run_in_threadpool(session.process, engine, image, gallery_for(session),frame_id=frame_id)


class Enrollment(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    consent: bool
    track_id: str


class PhotoForm(BaseModel):
    name: str = Field(min_length=1,max_length=60)
    consent: bool
    image_base64: str = Field(min_length=4,max_length=5_333_336)


@app.post('/api/self-test/enroll-photo')
async def enroll_photo(request: Request):
    chunks=[];size=0
    async for chunk in request.stream():
        size+=len(chunk)
        if size>5_400_000:raise HTTPException(413,'Photo request exceeds 4 MB image limit')
        chunks.append(chunk)
    try:form=PhotoForm.model_validate_json(b''.join(chunks))
    except ValidationError:raise HTTPException(422,'Provide a name, consent and a JPEG/PNG photo up to 4 MB')
    if any(isinstance(s,SelfSession) and s.active_id and time.monotonic()-s.last_seen<2.5 for s in sessions.values()):
        raise HTTPException(409,'Stop live capture before photo enrollment')
    return await run_in_threadpool(photo_enrollment.enroll,form.name,form.consent,form.image_base64)


@app.post("/api/sessions/{identifier}/enroll")
def enroll(identifier, form: Enrollment):
    session=session_for(identifier)
    selected_gallery=gallery_for(session)
    tester = session.enroll(form.name, form.consent, form.track_id, selected_gallery)
    return {"id": tester, "testers": selected_gallery.list()}


@app.get("/api/testers")
def testers(mode:Literal['verify','self']='verify'):
    return (self_gallery if mode=='self' else gallery).list()


@app.get("/api/testers/{identifier}/photo")
def photo(identifier,mode:Literal['verify','self']='verify'):
    return Response((self_gallery if mode=='self' else gallery).get(identifier)["photo"], media_type="image/jpeg")


@app.delete("/api/testers/{identifier}")
def delete_tester(identifier,mode:Literal['verify','self']='verify'):
    if mode=='self':
        self_search.delete(identifier)
    else:
        gallery.delete(identifier)
    for session in list(sessions.values()):
        with session.lock:
            if session.selected == identifier:
                session.selected = None
                session.clear_capture()
    return {"ok": True}


@app.get('/api/self-test')
def self_test_status():
    return self_search.status()


@app.post('/api/self-test/threshold-analysis')
def self_threshold_analysis():
    return self_search.threshold_analysis()


@app.get('/api/self-test/synthetic/{identity}/photo')
def synthetic_photo(identity:int):
    image=recognizer.preview_images([{'kind':'synthetic','id':str(identity)}]).get(f'synthetic:{identity}')
    if image is None:raise HTTPException(404,'Preview is not cached yet')
    return Response(base64.b64decode(image.split(',',1)[1]),media_type='image/png')


@app.get('/api/sessions/{identifier}/recognition-previews')
def recognition_previews(identifier,epoch:int,sequence:int):
    session=session_for(identifier)
    if not isinstance(session,SelfSession):raise HTTPException(400,'Self-test session required')
    return session.preview_snapshot(epoch,sequence)


@app.get("/api/sessions/{identifier}/export")
def export_session(identifier):
    session = session_for(identifier)
    with session.lock:
        diagnostics={key:session.last_frame.get(key) for key in ('head_status','head_count','face_count','quality','capture','timing','recognition')} if session.last_frame else None
        return JSONResponse({"scope": "Anonymous local session metrics; no face images or descriptors", "events": list(session.events),
            'capture_events':list(session.capture_events),'latest_diagnostics':diagnostics}, headers={"Content-Disposition": "attachment; filename=session_metrics.json"})


class BenchmarkSettings(BaseModel):
    count: int = 100000
    dimensions: int = 128
    queries: int = 100
    ef: int = 128


@app.get("/api/benchmark")
def benchmark_state():
    return benchmark.snapshot()


@app.post("/api/benchmark")
def benchmark_start(settings: BenchmarkSettings):
    benchmark.start(**settings.model_dump())
    return benchmark.snapshot()


@app.post("/api/benchmark/cancel")
def benchmark_cancel():
    benchmark.cancel_event.set()
    return {"ok": True}


@app.get("/api/benchmark/export")
def benchmark_export():
    result = benchmark.snapshot().get("result")
    if not result:
        raise HTTPException(404, "No completed benchmark")
    return JSONResponse(result, headers={"Content-Disposition": "attachment; filename=search_benchmark.json"})


app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
