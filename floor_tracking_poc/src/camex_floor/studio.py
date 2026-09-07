"""Local UI and API for complete spatial projects."""

import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
from urllib.parse import parse_qs, urlparse
import uuid

import cv2
import numpy as np

from .pipeline import Projects
from .geometry import calibration_fingerprint


def run_server(projects, port=8770):
    web = Path(__file__).parent / "studio_web"
    overlay_cache = {}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def json(self, data, code=200):
            encoded = json.dumps(data, allow_nan=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(encoded)

        def content(self, data, mime):
            self.send_response(200)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def file(self, path, download=False):
            if not path.is_file():
                raise ValueError("Asset is not ready")
            size = path.stat().st_size
            start = 0
            end = size - 1
            partial = False
            requested = self.headers.get("Range")
            if requested:
                import re

                m = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                if not m:
                    raise ValueError("Unsupported byte range")
                start = int(m.group(1))
                end = min(size - 1, int(m.group(2))) if m.group(2) else size - 1
                partial = True
                if start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
            self.send_response(206 if partial else 200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(path.name)[0] or "application/octet-stream",
            )
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            if partial:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            if download:
                self.send_header(
                    "Content-Disposition", f'attachment; filename="{path.name}"'
                )
            self.end_headers()
            with path.open("rb") as f:
                f.seek(start)
                remaining = end - start + 1
                while remaining:
                    b = f.read(min(1024 * 1024, remaining))
                    if not b:
                        break
                    self.wfile.write(b)
                    remaining -= len(b)

        def do_GET(self):
            try:
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                pid = q.get("project")
                if u.path == "/api/projects":
                    self.json(
                        {"projects": projects.list(), "engine": projects.engine_info()}
                    )
                    return
                if u.path == "/api/site":
                    self.json(projects.info(pid))
                    return
                if u.path == "/api/job":
                    self.json(projects.job_info(pid, q["job"]))
                    return
                if u.path == "/api/export":
                    self.json(projects.load(pid))
                    return
                if u.path == "/asset":
                    self.file(
                        projects.asset(pid, q["kind"], q.get("camera"), q.get("job")),
                        q.get("download") == "1",
                    )
                    return
                if u.path == "/overlay":
                    site = projects.load(pid)
                    cid = q["camera"]
                    c = site["cameras"][cid]
                    kind = q.get("kind", "points")
                    key = (
                        pid,
                        calibration_fingerprint(c, site["floor"]),
                        c.get("calibration_fingerprint"),
                        kind,
                    )
                    if key not in overlay_cache:
                        h = np.asarray(c["image_to_floor_h"], float)
                        if h.shape != (3, 3):
                            raise ValueError("Calibrate this camera first")
                        if kind in ("points", "allpoints") and c.get("point_layer"):
                            img = cv2.imread(
                                str(
                                    projects.asset(
                                        pid,
                                        "all_point_layer"
                                        if kind == "allpoints"
                                        and c.get("all_point_layer")
                                        else "point_layer",
                                        cid,
                                    )
                                ),
                                cv2.IMREAD_UNCHANGED,
                            )
                            h = h @ np.asarray(c["layer_to_image_h"])
                        else:
                            img = cv2.imread(
                                str(projects.asset(pid, "reference", cid)),
                                cv2.IMREAD_UNCHANGED,
                            )
                        if img is None:
                            raise ValueError("Overlay missing")
                        if img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                        rendered = cv2.warpPerspective(
                            img, h, tuple(site["floor"]["size_wh"])
                        )
                        overlay_cache.clear()
                        overlay_cache[key] = cv2.imencode(".png", rendered)[1].tobytes()
                    self.content(overlay_cache[key], "image/png")
                    return
                name = u.path.lstrip("/") or "index.html"
                if name not in ("index.html", "app.js", "style.css"):
                    self.json({"error": "Not found"}, 404)
                    return
                self.file(web / name)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, OSError, TypeError, cv2.error) as error:
                self.json({"error": str(error)}, 400)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and urlparse(origin).hostname not in (
                    "127.0.0.1",
                    "localhost",
                ):
                    raise ValueError("Cross-origin write denied")
                u = urlparse(self.path)
                q = {k: v[0] for k, v in parse_qs(u.query).items()}
                length = int(self.headers.get("Content-Length", 0))
                if u.path == "/api/upload":
                    if not 0 < length <= 20 * 1024**3:
                        raise ValueError("Upload must be under 20 GB")
                    uid = uuid.uuid4().hex
                    path = projects.workspace / "uploads" / (uid + ".bin")
                    remaining = length
                    with path.open("xb") as f:
                        while remaining:
                            data = self.rfile.read(min(1024 * 1024, remaining))
                            if not data:
                                raise ValueError("Upload interrupted")
                            f.write(data)
                            remaining -= len(data)
                    self.json(
                        projects.register_upload(path, q.get("name", "input.bin"))
                    )
                    return
                if not 0 < length <= 4 * 1024 * 1024:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(length))
                pid = body.get("project")
                if u.path == "/api/create":
                    result = projects.create(
                        body.get("name"), body["upload_id"], body.get("page", 0)
                    )
                elif u.path == "/api/camera":
                    result = projects.add_camera(
                        pid, body["camera_id"], body.get("label"), body["upload_id"]
                    )
                elif u.path == "/api/remove-camera":
                    result = projects.remove_camera(
                        pid, body["camera_id"], body["revision"]
                    )
                elif u.path == "/api/snapshot":
                    result = projects.snapshot(pid, body["camera_id"], body["seconds"])
                elif u.path == "/api/edit":
                    result = projects.edit(pid, body["camera_id"], body)
                elif u.path == "/api/finish":
                    result = projects.finish(pid, bool(body.get("preview")))
                elif u.path == "/api/calibrate":
                    result = projects.enqueue(pid, "calibrate", body["camera_id"], body)
                elif u.path == "/api/run":
                    result = projects.enqueue(pid, "infer", settings=body)
                else:
                    raise ValueError("Unknown action")
                self.json(result)
            except (BrokenPipeError, ConnectionResetError):
                pass
            except (ValueError, KeyError, OSError, TypeError, cv2.error) as error:
                self.json({"error": str(error)}, 400)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"CAMEX Spatial Studio: http://127.0.0.1:{port}/", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        projects.executor.shutdown(wait=False, cancel_futures=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--port", type=int, default=8770)
    a = parser.parse_args()
    run_server(Projects(a.root, a.workspace), a.port)


if __name__ == "__main__":
    main()
