"""Local office evaluation service. Never expose this directly to the internet."""
import json
import io
import os
import stat
import time
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool
from starlette.middleware.trustedhost import TrustedHostMiddleware

from domain import Gallery, QdrantIndex
from runtime import DEFAULTS, SETTINGS_PROFILE, Manager
from enrollment import Enrollment

HERE = Path(__file__).resolve().parent
ROOT = Path(os.environ.get("CAMEX_OFFICE_HOME", str(HERE / ".runtime-office"))).resolve()
ROOT.mkdir(parents=True, exist_ok=True, mode=0o700)
for directory in ("config", "reports"):
    (ROOT / directory).mkdir(exist_ok=True, mode=0o700)
index = QdrantIndex(os.environ.get("CAMEX_OFFICE_QDRANT_URL", "http://127.0.0.1:16343"))
gallery = Gallery(ROOT, index)
manager = Manager(ROOT, os.environ.get("CAMEX_OFFICE_MODELS", str(HERE.parent / "research_demo/.runtime/models")),
                  os.environ.get("CAMEX_OFFICE_PROVIDER", "CPUExecutionProvider"), gallery)
enrollment = Enrollment(manager)


@asynccontextmanager
async def lifespan(app):
    manager.start()
    yield
    manager.close()


app = FastAPI(title="CAMEX Office Evaluation", lifespan=lifespan, docs_url=None, redoc_url=None)
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["localhost", "127.0.0.1", "testserver"])


@app.middleware("http")
async def local_security(request, call_next):
    origin = request.headers.get("origin")
    if origin and origin.rstrip("/") != str(request.base_url).rstrip("/"):
        return JSONResponse({"detail": "Cross-origin requests are disabled"}, status_code=403)
    limit = 140_000_000 if request.url.path == "/api/enroll/preview" else 1_000_000
    try:
        if int(request.headers.get("content-length", "0")) > limit:
            return JSONResponse({"detail": "Request is too large"}, status_code=413)
    except ValueError:
        return JSONResponse({"detail": "Invalid request length"}, status_code=400)
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; script-src 'self'; style-src 'self'; connect-src 'self'; frame-ancestors 'none'"
    return response


@app.exception_handler(ValueError)
async def invalid(request, error):
    return JSONResponse({"detail": str(error)}, status_code=400)


@app.exception_handler(RuntimeError)
async def unavailable(request, error):
    return JSONResponse({"detail": str(error)}, status_code=503)


@app.get("/")
def root():
    return FileResponse(HERE / "static/index.html")


@app.get("/api/state")
def state():
    return manager.state()


@app.get("/api/health")
def health():
    return {"release": "office-evaluation-20261023", "ready": manager.workers_ready and gallery.healthy,
            "engine_error": manager.engine_error, "index_error": gallery.error}


@app.get("/api/people")
def people():
    return {"people": gallery.people()}


@app.post("/api/people")
async def create_person(request: Request):
    data = await request.json()
    return {"id": gallery.create_person(data.get("name", ""))}


@app.delete("/api/people/{identifier}")
def delete_person(identifier: str):
    gallery.delete_person(identifier)
    manager.reset_evidence()
    return {"ok": True, "pending_file_cleanup": gallery.pending_cleanup_count()}


@app.delete("/api/photos/{identifier}")
def delete_photo(identifier: str):
    gallery.delete_photo(identifier)
    manager.reset_evidence()
    return {"ok": True, "pending_file_cleanup": gallery.pending_cleanup_count()}


@app.get("/api/photo/{identifier}")
def photo(identifier: str):
    return FileResponse(gallery.photo_path(identifier), media_type="image/jpeg")


@app.post("/api/enroll/preview")
async def preview(request: Request):
    form = await request.form(max_files=200, max_fields=20)
    uploads = form.getlist("files") or form.getlist("files[]")
    paths = json.loads(str(form.get("paths", "[]")))
    if paths and len(paths) != len(uploads):
        raise ValueError("Every uploaded image needs one relative path")
    files = []
    total = 0
    for i, upload in enumerate(uploads):
        is_zip = str(upload.filename).lower().endswith(".zip")
        data = await upload.read(128_000_001 if is_zip else 12_000_001)
        total += len(data)
        if total > 128_000_000:
            raise ValueError("Import exceeds 128 MB")
        if is_zip:
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    entries = [e for e in archive.infolist() if not e.is_dir()]
                    if len(entries) > 200 or sum(e.file_size for e in entries) > 128_000_000:
                        raise ValueError("Archive exceeds 200 files or 128 MB expanded size")
                    for entry in entries:
                        if stat.S_ISLNK(entry.external_attr >> 16):
                            raise ValueError("Archive symbolic links are not allowed")
                        if entry.file_size > 12_000_000 or entry.file_size > max(entry.compress_size, 1)*200:
                            raise ValueError("Archive member is too large or overly compressed")
                        if entry.filename.startswith("__MACOSX/") or Path(entry.filename).name.startswith("."):
                            continue
                        files.append((entry.filename, archive.read(entry)))
            except zipfile.BadZipFile:
                raise ValueError("Invalid ZIP archive")
        else:
            files.append((paths[i] if paths else upload.filename, data))
    return await run_in_threadpool(enrollment.prepare, files, form.get("name") or None,
                                  form.get("person_id") or None, str(form.get("consent")).lower() == "true")


@app.post("/api/enroll/commit")
async def commit(request: Request):
    data = await request.json()
    return await run_in_threadpool(enrollment.commit, data.get("token"))


@app.post("/api/cameras")
async def add_camera(request: Request):
    data = await request.json()
    return manager.add_camera(data.get("name", ""), data.get("url", ""), data.get("authorized") is True)


@app.post("/api/cameras/start-all")
def start_all():
    for c in list(manager.cameras.values()):
        c.start()
    return {"ok": True}


@app.post("/api/cameras/stop-all")
def stop_all():
    for c in list(manager.cameras.values()):
        c.stop()
    return {"ok": True}


def camera(identifier):
    result = manager.cameras.get(identifier)
    if result is None:
        raise HTTPException(404, "Camera not found")
    return result


@app.post("/api/cameras/{identifier}/start")
def start_camera(identifier: str):
    camera(identifier).start()
    return {"ok": True}


@app.post("/api/cameras/{identifier}/stop")
def stop_camera(identifier: str):
    camera(identifier).stop()
    return {"ok": True}


@app.delete("/api/cameras/{identifier}")
def delete_camera(identifier: str):
    camera(identifier).stop()
    del manager.cameras[identifier]
    manager._save_cameras()
    return {"ok": True}


@app.get("/api/cameras/{identifier}/frame.jpg")
def camera_frame(identifier: str):
    c = camera(identifier)
    with c.lock:
        if c.preview is None:
            raise HTTPException(503, "No processed camera frame yet")
        return Response(c.preview, media_type="image/jpeg", headers={"X-Frame-Id": str(c.processed)})


@app.get("/api/settings/defaults")
def settings_defaults():
    return {"profile": SETTINGS_PROFILE, "settings": dict(DEFAULTS)}


@app.post("/api/settings")
async def settings(request: Request):
    return manager.update_settings(await request.json())


@app.get("/api/candidates")
def candidates():
    return {"candidates": manager.candidate_list()}


@app.get("/api/samples/{identifier}/photo")
def sample_photo(identifier: str):
    return Response(manager.sample_photo(identifier), media_type="image/jpeg")


@app.get("/api/candidates/{identifier}/photo")
def candidate_photo(identifier: str):
    manager.candidate_list()
    with manager.candidate_lock:
        c = manager.candidates.get(identifier)
        if not c:
            raise HTTPException(404, "Candidate expired")
        return Response(c["image"], media_type="image/jpeg")


@app.post("/api/candidates/{identifier}/{action}")
async def candidate_action(identifier: str, action: str, request: Request):
    if action not in ("approve", "reject"):
        raise ValueError("Unknown action")
    data = await request.json() if await request.body() else {}
    return await run_in_threadpool(manager.decide_candidate, identifier, action,
                                  isinstance(data, dict) and data.get("identity_verified") is True)


@app.get("/api/report")
def report():
    state = manager.state()
    for c in state["cameras"]:
        c.pop("tracks", None)
        c.pop("url_redacted", None)
    state.pop("people", None)
    state["timestamp"] = time.time()
    state["scope"] = "Local evaluation; timing is not identity accuracy or production capacity"
    return JSONResponse(state, headers={"Content-Disposition": "attachment; filename=office-evaluation-metrics.json"})


@app.get("/api/profile")
def profile():
    if not manager.vision:
        raise ValueError("Models not ready")
    return manager.vision.profile_summary()


app.mount("/static", StaticFiles(directory=HERE / "static", check_dir=False), name="static")
