"""Local calibration editor. Only registered assets can be read over HTTP."""

import base64
import copy
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import threading
from urllib.parse import parse_qs, urlparse
import uuid

import cv2
import numpy as np

from .geometry import (
    calibration_fingerprint,
    calibration_report,
    fit_homography,
    fit_layer,
    load_site,
    validate_site,
)


def atomic_json(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(p)


def serve(path, port=8766):
    path = Path(path).resolve()
    lock = threading.RLock()
    web = Path(__file__).parent / "web"
    overlay_cache = {}

    def resolve(p):
        return (path.parent / p).resolve()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, value, status=200, mime="application/json"):
            data = (
                json.dumps(value, allow_nan=False).encode()
                if mime == "application/json"
                else value
            )
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            try:
                route = urlparse(self.path)
                q = parse_qs(route.query)
                with lock:
                    site = load_site(path)
                if route.path == "/api/site":
                    data = copy.deepcopy(site)
                    data["verification"] = validate_site(site, False)
                    self.reply(data)
                    return
                if route.path == "/asset":
                    kind = q.get("kind", ["reference"])[0]
                    cid = q.get("camera", [""])[0]
                    if kind == "floor":
                        target = site["floor"]["asset"]
                    elif (
                        kind in ("reference", "point_layer", "floor_layer")
                        and cid in site["cameras"]
                    ):
                        target = site["cameras"][cid].get(kind)
                    else:
                        raise ValueError("Unknown asset")
                    if not target:
                        raise ValueError("Asset not supplied")
                    p = resolve(target)
                    self.reply(
                        p.read_bytes(),
                        mime=mimetypes.guess_type(p.name)[0]
                        or "application/octet-stream",
                    )
                    return
                if route.path == "/overlay":
                    cid = q["camera"][0]
                    kind = q.get("kind", ["image"])[0]
                    c = site["cameras"][cid]
                    key = (calibration_fingerprint(c, site["floor"]), kind)
                    if key not in overlay_cache:
                        h = np.array(c["image_to_floor_h"], float)
                        if (
                            kind == "points"
                            and c.get("point_layer")
                            and c.get("layer_to_image_h")
                        ):
                            img = cv2.imread(
                                str(resolve(c["point_layer"])), cv2.IMREAD_UNCHANGED
                            )
                            h = h @ np.array(c["layer_to_image_h"])
                        else:
                            img = cv2.imread(
                                str(resolve(c["reference"])), cv2.IMREAD_UNCHANGED
                            )
                        if img is None:
                            raise ValueError("Cannot read overlay")
                        if img.shape[2] == 3:
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
                        rendered = cv2.warpPerspective(
                            img, h, tuple(site["floor"]["size_wh"])
                        )
                        rendered[:, :, 3] = (
                            rendered[:, :, 3].astype(float) * 0.85
                        ).astype(np.uint8)
                        overlay_cache.clear()
                        overlay_cache[key] = cv2.imencode(".png", rendered)[1].tobytes()
                    self.reply(overlay_cache[key], mime="image/png")
                    return
                name = route.path.lstrip("/") or "index.html"
                if name not in ("index.html", "app.js", "style.css"):
                    self.reply({"error": "Not found"}, 404)
                    return
                p = web / name
                self.reply(
                    p.read_bytes(), mime=mimetypes.guess_type(p.name)[0] or "text/plain"
                )
            except (ValueError, KeyError, OSError) as error:
                self.reply({"error": str(error)}, 400)

        def do_POST(self):
            try:
                origin = self.headers.get("Origin")
                if origin and origin not in (
                    f"http://127.0.0.1:{port}",
                    f"http://localhost:{port}",
                ):
                    raise ValueError("Cross-origin write denied")
                length = int(self.headers.get("Content-Length", "0"))
                if length > 25 * 1024 * 1024:
                    raise ValueError("Request too large")
                body = json.loads(self.rfile.read(length))
                route = urlparse(self.path).path
                with lock:
                    site = load_site(path)
                    if body.get("revision") != site.get("revision", 0):
                        self.reply(
                            {"error": "Project changed; reload before saving"}, 409
                        )
                        return
                    if route == "/api/camera":
                        cid = str(body["camera_id"])
                        if (
                            not cid
                            or cid in site["cameras"]
                            or any(x in cid for x in ("/", "\\", "\x1f"))
                        ):
                            raise ValueError("Choose a new camera ID")
                        raw = base64.b64decode(
                            body["image"].split(",")[-1], validate=True
                        )
                        img = cv2.imdecode(
                            np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR
                        )
                        if img is None or max(img.shape[:2]) > 12000:
                            raise ValueError("Invalid or oversized image")
                        name = f"assets/{uuid.uuid4().hex}.png"
                        p = resolve(name)
                        p.parent.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(p), img)
                        site["cameras"][cid] = {
                            "label": body.get("label") or cid,
                            "reference": name,
                            "reference_sha256": hashlib.sha256(
                                p.read_bytes()
                            ).hexdigest(),
                            "image_size_wh": [img.shape[1], img.shape[0]],
                            "distortion": {"model": "none"},
                            "anchors": [],
                            "image_to_floor_h": None,
                        }
                    elif route == "/api/save":
                        cid = body["camera_id"]
                        camera = site["cameras"][cid]
                        if "anchors" in body:
                            camera["anchors"] = body["anchors"]
                        if "pixels_per_metre" in body:
                            ppm = body["pixels_per_metre"]
                            if ppm is not None and (not np.isfinite(ppm) or ppm <= 0):
                                raise ValueError("Scale must be positive")
                            site["floor"]["pixels_per_metre"] = ppm
                        action = body.get("action", "save")
                        if action == "fit":
                            h, audit = fit_homography(
                                camera["anchors"],
                                (site["floor"].get("pixels_per_metre") or 80)
                                * site.get("acceptance", {}).get(
                                    "fit_tolerance_m", 0.6
                                ),
                            )
                            camera["image_to_floor_h"] = h
                            camera["fit_audit"] = audit
                        elif action == "fit_layer":
                            h, audit = fit_layer(
                                camera,
                                (site["floor"].get("pixels_per_metre") or 80)
                                * site.get("acceptance", {}).get(
                                    "fit_tolerance_m", 0.6
                                ),
                            )
                            camera["image_to_floor_h"] = h
                            camera["fit_audit"] = audit
                        elif action == "transform":
                            h = np.asarray(camera["image_to_floor_h"], float)
                            a = np.radians(float(body.get("rotation", 0)))
                            s = float(body.get("scale", 1))
                            dx = float(body.get("dx", 0))
                            dy = float(body.get("dy", 0))
                            if not 0.1 <= s <= 10 or not np.isfinite([a, dx, dy]).all():
                                raise ValueError("Invalid transform")
                            center = np.asarray(site["floor"]["size_wh"]) / 2
                            lin = s * np.array(
                                [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
                            )
                            t = np.eye(3)
                            t[:2, :2] = lin
                            t[:2, 2] = center - lin @ center + [dx, dy]
                            camera["image_to_floor_h"] = (t @ h).tolist()
                        camera.pop("approved_fingerprint", None)
                        if action == "approve":
                            report = calibration_report(
                                camera,
                                site["floor"].get("pixels_per_metre"),
                                site.get("acceptance"),
                            )
                            if not report["ready"]:
                                raise ValueError("; ".join(report["reasons"]))
                            camera["approved_fingerprint"] = calibration_fingerprint(
                                camera, site["floor"]
                            )
                    else:
                        raise ValueError("Unknown action")
                    site["revision"] = site.get("revision", 0) + 1
                    validate_site(site, False)
                    atomic_json(path, site)
                    self.reply({"ok": True, "revision": site["revision"]})
            except (ValueError, KeyError, OSError, TypeError) as error:
                self.reply({"error": str(error)}, 400)

    print(f"Calibration editor: http://127.0.0.1:{port}/", flush=True)
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
