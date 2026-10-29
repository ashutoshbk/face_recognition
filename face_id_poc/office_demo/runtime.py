"""Bounded multi-camera capture, tracking and independent recognition workers."""
from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import threading
import time
import uuid
from collections import OrderedDict, deque
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlsplit

import cv2
import numpy as np

from domain import HeadTrackIdentity, SingleReferenceReview, best_quality_crops, registered_gate, write_json
from optional_review import REVIEW_DEFAULTS, REVIEW_BOOLS, REVIEW_RANGES, ReviewService, validate_review_settings
from vision_backend import Vision, HeadTracker, associate_faces, encode_jpeg, assess_query_quality


SETTINGS_PROFILE = "office-demo-20261029"
# Reproduce the office evaluation, not a production-calibrated operating point.
DEFAULTS = dict(match_threshold=.30, match_margin=.06, admission_threshold=.68,
                admission_margin=.10, adaptation_mode="review", analysis_fps=10.,
                recognition_hz=1., face_min_width=20, quality_strictness=49, single_reference_mode="review",
                face_detection_mode="full_frame")
DEFAULTS.update(REVIEW_DEFAULTS)


@lru_cache(maxsize=1)
def rtsp_timeout_option():
    help_text = subprocess.run(["ffmpeg", "-hide_banner", "-h", "demuxer=rtsp"],
                               capture_output=True, text=True, timeout=5).stdout
    return "-stimeout" if "-stimeout " in help_text else "-timeout"


def validate_url(url):
    if not isinstance(url, str) or len(url) > 2048 or any(ord(c) < 32 for c in url):
        raise ValueError("Invalid camera address")
    parsed = urlsplit(url)
    if parsed.scheme not in ("rtsp", "rtsps") or not parsed.hostname:
        raise ValueError("A reachable RTSP/RTSPS address is required")
    parsed.port
    return url


def redacted_url(url):
    parsed = urlsplit(url)
    host = parsed.hostname or "camera"
    return f"{parsed.scheme}://{host}:{parsed.port or 554}/[private path]"


def job_current(job, camera, track, epoch, now):
    return bool(track and track.get("tracking_state", "observed") == "observed"
        and track.get("token") == job.get("track_token")
        and camera.generation == job["generation"] and job["epoch"] == epoch
        and 0 <= now-job["captured"] <= 2)


class Camera:
    def __init__(self, name, url, identifier=None):
        self.id = identifier or uuid.uuid4().hex[:12]
        self.name, self.url = name, validate_url(url)
        self.lock = threading.RLock()
        self.capture_lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.process = None
        self.frame = None
        self.frame_time = 0.
        self.sequence = 0
        self.processed = 0
        self.generation = 0
        self.state = "stopped"
        self.error = None
        self.tracker = None
        self.tracks = {}
        self.track_cache = {}
        self.face_observations = []
        self.preview = None
        self.inference_times = deque(maxlen=100)
        self.process_times = deque(maxlen=100)
        self.capture_times = deque(maxlen=100)
        self.last_analysis = 0.
        self.total_processed = 0
        self.processing = False
        self.last_job = {}
        self.dimensions = None
        self.decoder = "not started"
        self.counters = dict(face_detections=0, usable_face_detections=0, small_face_detections=0,
            recognition_queries=0, named_decisions=0, confirmed_decisions=0, unknown_decisions=0,
            stale_recognition_results=0)

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        with self.lock:
            self.process_times.clear()
            self.inference_times.clear()
            self.capture_times.clear()
            self.total_processed = 0
            for key in self.counters:
                self.counters[key] = 0
        self.thread = threading.Thread(target=self._capture, daemon=True, name="capture-"+self.id)
        self.thread.start()

    def stop(self):
        self.stop_event.set()
        if self.process and self.process.poll() is None:
            self.process.terminate()
        if self.thread:
            self.thread.join(timeout=3)
        with self.lock:
            self.generation += 1
            self.frame = self.preview = None
            self.tracks.clear()
            self.track_cache.clear()
            self.face_observations.clear()
            self.state = "stopped"

    def _publish_frame(self, frame, now):
        # Never wait for model work or JPEG rendering to drain the decoder.
        with self.capture_lock:
            self.frame, self.frame_time = frame, now
            self.sequence += 1
            self.capture_times.append(now)
            self.state, self.error = "running", None

    def _capture(self):
        while not self.stop_event.is_set():
            self.state = "connecting"
            self.error = None
            try:
                timeout_option = rtsp_timeout_option()
                probe = subprocess.run(["ffprobe", "-v", "error", timeout_option, "5000000",
                    "-rtsp_transport", "tcp", "-select_streams", "v:0", "-show_entries",
                    "stream=width,height", "-of", "json", self.url], capture_output=True, timeout=12)
                info = json.loads(probe.stdout)["streams"][0]
                width, height = int(info["width"]), int(info["height"])
                if width < 80 or height < 80 or width*height > 16_000_000:
                    raise ValueError("Unsupported stream dimensions")
                gpu_decode = os.environ.get("CAMEX_OFFICE_HWDECODE", "0") == "1"
                command = ["ffmpeg", "-nostdin", "-v", "error", timeout_option, "5000000", "-rtsp_transport", "tcp", "-threads", "2", "-filter_threads", "1"]
                if gpu_decode:
                    command += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
                command += ["-i", self.url, "-an", "-sn", "-dn"]
                filters = "hwdownload,format=nv12," if gpu_decode else ""
                command += ["-vf", filters+"fps=10", "-pix_fmt", "bgr24", "-threads:v", "1", "-f", "rawvideo", "pipe:1"]
                self.process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                                bufsize=65536)
                with self.lock:
                    self.generation += 1
                    self.tracker = HeadTracker(fps=10, threshold=.35, buffer_seconds=2, geometry_fallback=True)
                    self.tracks.clear()
                    self.track_cache.clear()
                    self.face_observations.clear()
                    self.dimensions = [width, height]
                    self.decoder = "FFmpeg NVDEC requested" if gpu_decode else "FFmpeg software decode"
                size = width*height*3
                while not self.stop_event.is_set():
                    frame = np.empty((height, width, 3), dtype=np.uint8)
                    target = memoryview(frame).cast("B")
                    offset = 0
                    while offset < size:
                        count = self.process.stdout.readinto(target[offset:])
                        if not count:
                            break
                        offset += count
                    if offset != size:
                        break
                    now = time.monotonic()
                    self._publish_frame(frame, now)
                if not self.stop_event.is_set():
                    raise RuntimeError("Stream disconnected or decoder stopped")
            except Exception:
                with self.lock:
                    self.state = "reconnecting"
                    self.error = "No fresh camera frames; check address, access and decoder compatibility"
                    self.frame = self.preview = None
                    self.tracks.clear()
                    self.track_cache.clear()
                    self.face_observations.clear()
                    self.generation += 1
            finally:
                if self.process:
                    if self.process.poll() is None:
                        self.process.terminate()
                    try:
                        self.process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait(timeout=2)
                    if self.process.stdout:
                        self.process.stdout.close()
            self.stop_event.wait(2)

    def public(self):
        with self.lock:
            times = [t for t in self.process_times if time.monotonic()-t <= 5]
            fps = (len(times)-1)/(times[-1]-times[0]) if len(times) > 1 and times[-1] > times[0] else 0
            if self.state != "running" or not times or time.monotonic()-times[-1] > 2:
                fps = 0
            tracks = [{k: v for k, v in t.items() if k not in ("evidence", "single_review", "last_seen", "token")} for t in self.tracks.values()]
            with self.capture_lock:
                capture = [t for t in self.capture_times if time.monotonic()-t <= 5]
            capture_fps = (len(capture)-1)/(capture[-1]-capture[0]) if len(capture)>1 else 0
            return dict(id=self.id, name=self.name, url_redacted=redacted_url(self.url), state=self.state,
                counters=dict(self.counters),
                error=self.error, generation=self.generation, dimensions=self.dimensions, tracks=tracks,
                face_counts=dict(detected=len(self.face_observations),
                    usable=sum(f["usable"] for f in self.face_observations),
                    small=sum(f["small"] for f in self.face_observations),
                    unassigned=sum(not f["assigned"] for f in self.face_observations)),
                decoder=self.decoder, metrics=dict(fps=round(fps, 2), capture_fps=round(capture_fps,2), processed_frames=self.total_processed,
                frame_age_ms=round((time.monotonic()-self.frame_time)*1000) if self.frame_time else None,
                inference_ms=round(sum(self.inference_times)/len(self.inference_times), 1) if self.inference_times else None))


class Manager:
    def __init__(self, root, models, provider, gallery):
        self.root, self.models, self.provider = Path(root), Path(models), provider
        (self.root / "config").mkdir(parents=True, exist_ok=True, mode=0o700)
        self.gallery = gallery
        self.settings = dict(DEFAULTS)
        self.cameras = OrderedDict()
        self.pending = OrderedDict()
        self.pending_lock = threading.Lock()
        self.candidates = OrderedDict()
        self.candidate_lock = threading.Lock()
        self.samples = OrderedDict()
        self.samples_lock = threading.Lock()
        self.cooldowns = {}
        self.epoch = 0
        self.admission_lock = threading.RLock()
        self.vision = None
        self.workers_ready = False
        self.engine_error = None
        self.stop_event = threading.Event()
        self.threads = []
        self.recognition_times = deque(maxlen=200)
        self.hardware = {}
        self.started = time.time()
        path = self.root / "config" / "cameras.json"
        if path.exists():
            for c in json.loads(path.read_text()):
                self.cameras[c["id"]] = Camera(c["name"], c["url"], c["id"])
        settings_path = self.root / "config" / "settings.json"
        if settings_path.exists():
            self.settings.update(json.loads(settings_path.read_text()))
        if self.settings.get("adaptation_mode") == "auto" and os.environ.get("CAMEX_OFFICE_ALLOW_AUTO_ADMISSION") != "1":
            self.settings["adaptation_mode"] = "review"
        for key in REVIEW_BOOLS:
            if not isinstance(self.settings.get(key), bool):
                self.settings[key] = REVIEW_DEFAULTS[key]
        for key, (minimum, maximum) in REVIEW_RANGES.items():
            value = self.settings.get(key)
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not minimum <= value <= maximum
                    or key == "unknown_max_records" and value != int(value)):
                self.settings[key] = REVIEW_DEFAULTS[key]
        if not self.settings["review_consent"]:
            self.settings["unknown_capture_enabled"] = False
            self.settings["flagged_review_enabled"] = False
        self.review = ReviewService(self.root, self.gallery, self.settings, self._review_current)

    def _review_current(self, item):
        camera = self.cameras.get(item.get("camera_id"))
        if camera is None or camera.generation != item.get("camera_generation") or item.get("identity_epoch") != self.epoch:
            return False
        track = camera.tracks.get(item.get("track_id"))
        return bool(track and track.get("token") == item.get("track_token") and track.get("tracking_state") == "observed"
                    and track.get("face_visible"))

    def start(self):
        self.review.start()
        self.threads = [threading.Thread(target=self._load, daemon=True, name="model-loader")]
        self.threads[0].start()

    def _load(self):
        try:
            self.vision = Vision(self.models, self.provider, self.root / "reports" / "profiles")
        except Exception as error:
            self.engine_error = str(error)
            return
        while not self.stop_event.is_set():
            try:
                self.gallery.sync()
                break
            except Exception as error:
                self.engine_error = str(error)
                self.stop_event.wait(2)
        if self.stop_event.is_set():
            return
        count = 2 if self.provider == "CUDAExecutionProvider" else 1
        for target in ([self._tracking_loop]*count + [self._recognition_loop, self._resources_loop]):
            t = threading.Thread(target=target, daemon=True)
            self.threads.append(t)
            t.start()
        self.workers_ready = True
        self.engine_error = None

    def _save_cameras(self):
        path = self.root / "config" / "cameras.json"
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        write_json(path, [dict(id=c.id, name=c.name, url=c.url) for c in self.cameras.values()])

    def add_camera(self, name, url, authorized):
        if not authorized:
            raise ValueError("Confirm authorization for this controlled camera feed")
        if not isinstance(name, str) or not name or len(name) > 80 or any(ord(c) < 32 for c in name):
            raise ValueError("Camera name must contain 1-80 characters")
        if len(self.cameras) >= int(os.environ.get("CAMEX_OFFICE_CAMERA_LIMIT", "32")):
            raise ValueError("Configured resource limit reached")
        c = Camera(name, url)
        self.cameras[c.id] = c
        self._save_cameras()
        return c.public()

    def reset_evidence(self):
        with self.admission_lock:
            self._reset_evidence()

    def _reset_evidence(self):
        self.epoch += 1
        self.review.configure(self.settings)
        with self.pending_lock:
            self.pending.clear()
        with self.candidate_lock:
            self.candidates.clear()
        with self.samples_lock:
            self.samples.clear()
        for camera in list(self.cameras.values()):
            with camera.lock:
                for track in camera.track_cache.values():
                    track["evidence"].clear()
                    if track.get("single_review"):
                        track["single_review"].clear()
                    track["samples"] = []
                    track.update(name=None, person_id=None, state="Unknown", votes=0,
                                 observations=0, admission=False, identity_basis=None,
                                 last_face_match_age_seconds=None, token=uuid.uuid4().hex)

    def update_settings(self, changes):
        if not isinstance(changes, dict):
            raise ValueError("Settings must be a JSON object")
        ranges = dict(match_threshold=(0, 1), match_margin=(0, .5), admission_threshold=(0, 1),
            admission_margin=(0, .5), analysis_fps=(1, 30), recognition_hz=(.2, 5), face_min_width=(20, 300),
            quality_strictness=(0, 100))
        proposed = dict(self.settings)
        for key, value in changes.items():
            if key in REVIEW_BOOLS:
                if not isinstance(value, bool):
                    raise ValueError("Review toggles require true or false")
            elif key in REVIEW_RANGES:
                if isinstance(value, bool):
                    raise ValueError("Review numeric settings require a number")
                value = float(value)
                if not REVIEW_RANGES[key][0] <= value <= REVIEW_RANGES[key][1]:
                    raise ValueError("Setting outside allowed range: " + key)
                if key == "unknown_max_records" and value != int(value):
                    raise ValueError("Unknown record limit must be a whole number")
            elif key == "face_detection_mode":
                if value not in ("full_frame", "head_detail"):
                    raise ValueError("Face detection mode must be full_frame or head_detail")
            elif key == "single_reference_mode":
                if value not in ("off", "review"):
                    raise ValueError("Single-reference admission supports Off or manual Review only")
            elif key == "adaptation_mode":
                if value not in ("off", "review", "auto"):
                    raise ValueError("Invalid adaptation mode")
                if value == "auto" and os.environ.get("CAMEX_OFFICE_ALLOW_AUTO_ADMISSION") != "1":
                    raise ValueError("Automatic admission is locked pending threshold validation; use review mode")
            elif key in ranges:
                value = float(value)
                if not ranges[key][0] <= value <= ranges[key][1]:
                    raise ValueError("Setting outside allowed range: " + key)
            else:
                raise ValueError("Unknown setting: " + key)
            proposed[key] = value
        validate_review_settings(proposed)
        if proposed["admission_threshold"] < proposed["match_threshold"] or proposed["admission_margin"] < proposed["match_margin"]:
            raise ValueError("Admission must be at least as strict as recognition")
        with self.admission_lock:
            write_json(self.root / "config" / "settings.json", proposed)
            self.settings = proposed
            self.reset_evidence()
        return self.settings

    def _tracking_loop(self):
        while not self.stop_event.is_set():
            did_work = False
            for camera in list(self.cameras.values()):
                now = time.monotonic()
                with camera.lock:
                    with camera.capture_lock:
                        frame, seq, captured = camera.frame, camera.sequence, camera.frame_time
                    if camera.state != "running" or frame is None:
                        continue
                    if now-captured > 2:
                        camera.state, camera.error = "stale", "Camera frames are stale"
                        camera.tracks.clear()
                        camera.track_cache.clear()
                        camera.face_observations.clear()
                        camera.preview = None
                        continue
                    if camera.processing or seq == camera.processed or now-camera.last_analysis < 1/self.settings["analysis_fps"]:
                        continue
                    generation = camera.generation
                    tracker = camera.tracker
                    camera.processing = True
                    camera.processed, camera.last_analysis = seq, now
                did_work = True
                try:
                    tick = time.perf_counter()
                    heads, faces = self.vision.detect(frame, detail=self.settings["face_detection_mode"] == "head_detail")
                    tracks = tracker.update(heads, frame.shape, int(captured*1000))
                    associations, ambiguous = associate_faces(tracks, faces)
                    duration = (time.perf_counter()-tick)*1000
                    with camera.lock:
                        if generation != camera.generation or camera.state != "running" or time.monotonic()-captured > 2:
                            continue
                        ambiguous_ids = {str(k) for k in tracker.ambiguous_track_ids}
                        assigned_faces = {fi for ti,fi in associations.items()
                            if ti not in ambiguous and str(tracks[ti]["track_id"]) not in ambiguous_ids}
                        camera.face_observations = []
                        face_qualities = []
                        for fi, face in enumerate(faces):
                            quality, reference_quality = assess_query_quality(frame, face,
                                int(self.settings["face_min_width"]), self.settings["quality_strictness"])
                            face_qualities.append((quality, reference_quality))
                            camera.face_observations.append(dict(
                                bbox=[float(face[0]),float(face[1]),float(face[0]+face[2]),float(face[1]+face[3])],
                                landmarks=face[4:14].reshape(5,2).tolist(), width=quality["width"],
                                usable=quality["accepted"] and fi in assigned_faces,
                                assigned=fi in assigned_faces, small=bool(face[2] < self.settings["face_min_width"]),
                                reasons=quality["reasons"] or ([] if fi in assigned_faces else ["Unassigned"])))
                        for key, old in list(camera.track_cache.items()):
                            if now-old["last_seen"] > 2 or key in ambiguous_ids:
                                self._release_samples(old.get("samples", []))
                                del camera.track_cache[key]
                        previous = camera.track_cache
                        current = {}
                        for track_index, item in enumerate(tracks):
                            key = str(item["track_id"])
                            old = previous.get(key, {})
                            t = dict(id=key, bbox=[float(x) for x in item["bbox"]],
                                evidence=old.get("evidence", HeadTrackIdentity()), last_seen=now,
                                single_review=old.get("single_review", SingleReferenceReview()),
                                token=old.get("token", uuid.uuid4().hex),
                                name=old.get("name"), state=old.get("state", "Unknown"),
                                votes=old.get("votes", 0), observations=old.get("observations", 0),
                                score=old.get("score"), admission=False, face_visible=False,
                                samples=self._renew_samples(old.get("samples", []), now),
                                sample_policy="best_quality", samples_seen=old.get("samples_seen", 0),
                                tracking_state="observed", coast_age_seconds=0.,
                                recognition_ms=old.get("recognition_ms"), search_ms=old.get("search_ms"))
                            t.update(t["evidence"].summary(now))
                            face_index = associations.get(track_index)
                            if track_index in ambiguous or key in ambiguous_ids:
                                t["evidence"].clear()
                                t["single_review"].clear()
                                self._release_samples(t["samples"])
                                t["samples"] = []
                                t.update(t["evidence"].summary(now))
                                t.update(state="Association ambiguous", token=uuid.uuid4().hex)
                            elif face_index is None:
                                t.update(t["evidence"].summary(now))
                                if not t["name"]:
                                    t["state"] = "No usable face"
                            else:
                                face = faces[face_index]
                                _, aligned, crop = self.vision.describe(frame, face, int(self.settings["face_min_width"]))
                                quality, reference_quality = face_qualities[face_index]
                                t["quality"], t["face_visible"] = quality, quality["accepted"]
                                if not quality["accepted"]:
                                    t.update(t["evidence"].summary(now))
                                    if not t["name"]:
                                        t["state"] = "Quality rejected"
                                else:
                                    t.update(t["evidence"].summary(now, face_available=True))
                                if quality["accepted"] and now-camera.last_job.get(key, -1e9) >= 1/self.settings["recognition_hz"]:
                                    job = dict(camera=camera, track_id=key, generation=generation, epoch=self.epoch,
                                        track_token=t["token"], sequence=seq, captured=captured,
                                        aligned=aligned, crop=crop, quality=quality, reference_quality=reference_quality)
                                    if self.settings["unknown_capture_enabled"]:
                                        # Keep native surrounding pixels so a saved unknown can
                                        # pass the normal detector again during explicit enrollment.
                                        x, y, w, h = face[:4]
                                        pad = .45*max(w,h)
                                        job["review_crop"] = frame[max(0,int(y-pad)):min(frame.shape[0],int(y+h+pad)),
                                            max(0,int(x-pad)):min(frame.shape[1],int(x+w+pad))].copy()
                                    with self.pending_lock:
                                        self.pending[(camera.id, key)] = job
                                        while len(self.pending) > 128:
                                            self.pending.popitem(last=False)
                                    camera.last_job[key] = now
                            current[key] = t
                        for key, old in list(previous.items()):
                            gap = now-old["last_seen"]
                            if key not in current and key not in ambiguous_ids and 0 <= gap <= .8:
                                box = np.asarray(old["bbox"])
                                competing = any(np.prod(np.maximum(0, np.minimum(box[2:], np.asarray(t["bbox"])[2:])-
                                    np.maximum(box[:2], np.asarray(t["bbox"])[:2]))) > 0 for t in current.values())
                                if competing:
                                    self._release_samples(old.get("samples", []))
                                    del camera.track_cache[key]
                                    continue
                                held = dict(old, tracking_state="coasting", coast_age_seconds=round(gap, 2),
                                            face_visible=False, admission=False)
                                held.update(held["evidence"].summary(now))
                                held["samples"] = self._renew_samples(held.get("samples", []), now)
                                current[key] = held
                        camera.track_cache.update({k:t for k,t in current.items() if t["tracking_state"] == "observed"})
                        camera.tracks = current
                        camera.process_times.append(time.monotonic())
                        camera.total_processed += 1
                        camera.counters["face_detections"] += len(camera.face_observations)
                        camera.counters["usable_face_detections"] += sum(f["usable"] for f in camera.face_observations)
                        camera.counters["small_face_detections"] += sum(f["small"] for f in camera.face_observations)
                        camera.inference_times.append(duration)
                        camera.preview = self._draw(camera, frame)
                except Exception as error:
                    with camera.lock:
                        camera.error = "Analysis error: " + str(error)[:180]
                        camera.tracks.clear()
                        camera.track_cache.clear()
                        camera.face_observations.clear()
                        camera.preview = None
                finally:
                    with camera.lock:
                        camera.processing = False
            if not did_work:
                self.stop_event.wait(.01)

    def _draw(self, camera, frame):
        width = min(960, frame.shape[1])
        ratio = width/frame.shape[1]
        view = cv2.resize(frame, (width, round(frame.shape[0]*ratio)))
        for track in camera.tracks.values():
            if track.get("tracking_state") == "coasting":
                continue
            x1, y1, x2, y2 = [int(v*ratio) for v in track["bbox"]]
            confirmed = track["state"] in ("Confirmed", "Tracked")
            color = (210, 150, 45) if track["state"] == "Tracked" else (80, 180, 40) if confirmed else (20, 180, 240)
            state_label = {"No usable face": "No face", "Quality rejected": "Low quality",
                           "Recognition unavailable": "Offline"}.get(track["state"], track["state"])
            label = track["name"] or state_label
            if track["name"] and not confirmed:
                label = "? " + label
            if track["name"] and track["state"] == "Tracked":
                label += " [head track]"
            text = f'#{track["id"]} {label[:26]}'
            if track["observations"]:
                text += f' {track["votes"]}/{track["observations"]}'
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
            text_x = min(max(1, x1), max(1, width-text_width-3))
            cv2.putText(view, text, (text_x, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
        for face in camera.face_observations:
            x1,y1,x2,y2 = [int(value*ratio) for value in face["bbox"]]
            color = (230,200,0) if face["usable"] else (0,175,255)
            for x in range(x1,x2+1,5):
                cv2.circle(view,(x,y1),1,color,-1)
                cv2.circle(view,(x,y2),1,color,-1)
            for y in range(y1,y2+1,5):
                cv2.circle(view,(x1,y),1,color,-1)
                cv2.circle(view,(x2,y),1,color,-1)
            for x,y in face["landmarks"]:
                cv2.circle(view,(int(x*ratio),int(y*ratio)),1,color,-1)
            text = f'face {round(face["width"])}px'
            if not face["usable"]:
                reasons = {"Face too small":"small", "Head angle":"angle", "Blur":"blur",
                           "Lighting":"light", "Clipped face":"clipped", "Unassigned":"unassigned",
                           "Low detection score":"weak"}
                text += " / " + reasons.get(next(iter(face["reasons"]), ""), "check")
            tx = max(0,min(x1,width-120))
            ty = max(12,min(y2+12,view.shape[0]-3))
            cv2.putText(view,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.34,color,1,cv2.LINE_AA)
        return encode_jpeg(view)

    def _recognition_loop(self):
        while not self.stop_event.is_set():
            with self.pending_lock:
                job = self.pending.popitem(last=False)[1] if self.pending else None
            if job is None:
                self.stop_event.wait(.02)
                continue
            camera = job["camera"]
            if time.monotonic()-job["captured"] > 2 or job["epoch"] != self.epoch:
                continue
            try:
                tick = time.perf_counter()
                vector = self.vision.encode(job["aligned"])
                search_tick = time.perf_counter()
                ranked = self.gallery.search(vector)
                search_ms = (time.perf_counter()-search_tick)*1000
                total_ms = (time.perf_counter()-tick)*1000
                self.recognition_times.append(total_ms)
                with camera.lock:
                    camera.counters["recognition_queries"] += 1
                    track = camera.tracks.get(job["track_id"])
                    if not job_current(job, camera, track, self.epoch, time.monotonic()):
                        camera.counters["stale_recognition_results"] += 1
                        continue
                    result = track["evidence"].add(ranked, job["captured"], job["sequence"], self.settings)
                    camera.counters["named_decisions"] += int(result["name"] is not None)
                    camera.counters["confirmed_decisions"] += int(result["state"] == "Confirmed")
                    camera.counters["unknown_decisions"] += int(result["name"] is None)
                    display = result if track["face_visible"] else track["evidence"].summary(time.monotonic())
                    track.update(display, recognition_ms=round(total_ms, 1), search_ms=round(search_ms, 2))
                    identifier = uuid.uuid4().hex
                    jpeg = encode_jpeg(job["crop"])
                    digest = hashlib.sha256(jpeg).hexdigest()
                    sample = dict(id=identifier, url=f"/api/samples/{identifier}/photo",
                        quality_score=job["quality"]["score"], timestamp=job["captured"],
                        digest=digest, person_id=result["person_id"], candidate=result["name"], state=result["state"])
                    selected = best_quality_crops(track["samples"], sample)
                    keep_ids = {item["id"] for item in selected}
                    with self.samples_lock:
                        for old in track["samples"]:
                            if old["id"] not in keep_ids:
                                self.samples.pop(old["id"], None)
                        if identifier in keep_ids:
                            self.samples[identifier] = (time.monotonic()+10, jpeg)
                        for key in list(self.samples):
                            if self.samples[key][0] < time.monotonic():
                                del self.samples[key]
                        while len(self.samples) > 250:
                            self.samples.popitem(last=False)
                    track["samples"] = selected
                    track["samples_seen"] += 1
                    self._review_observation(job, result, ranked, jpeg, track)
                    proposal = track["single_review"].observe(ranked, result["person_id"], job["reference_quality"],
                        job["captured"], job["sequence"], digest) if self.settings["single_reference_mode"] == "review" else None
                if proposal and result["state"] == "Confirmed" and self.settings["adaptation_mode"] != "off":
                    anchor = next(p["anchor_version"] for p in ranked if p["person_id"] == result["person_id"])
                    self._offer_reference(job, result, vector, anchor, ranked[0]["registered_revision"], proposal)
                if result["admission"] and track["face_visible"] and self.settings["adaptation_mode"] != "off":
                    anchor = next((p["anchor_version"] for p in ranked if p["person_id"] == result["person_id"]), None)
                    revision = ranked[0]["registered_revision"] if ranked else None
                    self._offer_reference(job, result, vector, anchor, revision)
            except Exception as error:
                with camera.lock:
                    t = camera.tracks.get(job["track_id"])
                    if t and t.get("token") == job.get("track_token"):
                        t["evidence"].clear()
                        t.update(name=None, state="Recognition unavailable", token=uuid.uuid4().hex, votes=0, observations=0)
                self.engine_error = str(error)[:180]

    def _review_observation(self, job, result, ranked, jpeg, track):
        if not self.settings["review_consent"] or not track.get("face_visible"):
            return
        person_id = result.get("person_id")
        target = next((p for p in ranked if p["person_id"] == person_id), None)
        threshold = max(self.settings["review_match_threshold"], self.settings["admission_threshold"])
        margin = max(self.settings["review_match_margin"], self.settings["admission_margin"])
        strict = False
        score, gap = 0., 0.
        if target and ranked and ranked[0]["person_id"] == person_id and result.get("state") == "Confirmed":
            score = target["registered_score"]
            rival = max((p["registered_score"] for p in ranked if p["person_id"] != person_id), default=-1.)
            gap = score-rival
            if len(target["registered"]) == 1:
                strict = score >= max(.75, threshold) and gap >= max(.15, margin)
            else:
                strict = registered_gate(ranked, person_id, threshold, margin)[0]
        unknown = (not person_id and not track.get("name") and result.get("state") == "Unknown"
                   and (not ranked or ranked[0]["score"] < self.settings["match_threshold"]))
        if unknown and job.get("review_crop") is not None:
            jpeg = encode_jpeg(job["review_crop"])
        source = f'{job["camera"].id}:{job["generation"]}:{job["track_token"]}'
        self.review.submit(dict(source_key=hashlib.sha256(source.encode()).hexdigest(),
            camera_id=job["camera"].id, camera_name=job["camera"].name, track_id=job["track_id"],
            camera_generation=job["generation"], identity_epoch=job["epoch"], track_token=job["track_token"],
            captured=job["captured"], sequence=job["sequence"], image=jpeg, quality=dict(job["reference_quality"]),
            unknown=unknown, person_id=person_id, strict_match=strict, score=score, margin=gap))

    def _renew_samples(self, samples, now):
        with self.samples_lock:
            valid = []
            for item in samples:
                stored = self.samples.get(item["id"])
                if stored and stored[0] >= now:
                    self.samples[item["id"]] = (now+10, stored[1])
                    valid.append(item)
            return valid

    def _release_samples(self, samples):
        with self.samples_lock:
            for item in samples:
                self.samples.pop(item["id"], None)

    def _offer_reference(self, job, result, vector, anchor_version, registered_revision, single_review=None):
        quality = job["reference_quality"]
        if job["epoch"] != self.epoch or anchor_version is None or not quality["accepted"]:
            return
        person_id = result["person_id"]
        with self.candidate_lock:
            if any(c["person_id"] == person_id and c["expires"] > time.monotonic() for c in self.candidates.values()):
                return
        now = time.monotonic()
        if now-self.cooldowns.get(person_id, -1e9) < 10:
            return
        self.cooldowns[person_id] = now
        c = dict(id=uuid.uuid4().hex, person_id=person_id, person_name=result["name"],
            score=quality["score"], quality=quality, image=encode_jpeg(job["crop"]),
            vector=vector, reason="One-reference proposal; verify identity before saving" if single_review else "Registered-only consensus passed; awaiting review",
            policy="single_reference_review" if single_review else "registered_consensus",
            review_required=bool(single_review),
            epoch=job["epoch"], anchor_version=anchor_version, registered_revision=registered_revision,
            expires=now+600 if single_review else now+120, camera_id=job["camera"].id,
            camera_generation=job["generation"], track_id=job["track_id"],
            evidence=single_review or result["admission_details"])
        refs = [r for r in self.gallery.references() if r["person_id"] == person_id and r["pool"] == "registered"]
        if refs:
            c["anchor_reference_id"] = refs[0]["id"]
            c["anchor_reference_url"] = "/api/photo/"+refs[0]["id"]
        c["url"] = "/api/candidates/"+c["id"]+"/photo"
        if self.settings["adaptation_mode"] == "auto" and not single_review:
            self._admit(c)
        else:
            with self.candidate_lock:
                self.candidates[c["id"]] = c
                while len(self.candidates) > 50:
                    self.candidates.popitem(last=False)

    def _admit(self, candidate, identity_verified=False):
        with self.admission_lock:
            return self._admit_locked(candidate, identity_verified)

    def decide_candidate(self, identifier, action, identity_verified=False):
        if action not in ("approve", "reject"):
            raise ValueError("Unknown action")
        with self.admission_lock:
            with self.candidate_lock:
                candidate = self.candidates.get(identifier)
                if candidate is None:
                    raise ValueError("Candidate expired or already decided")
            result = self._admit_locked(candidate, identity_verified) if action == "approve" else {"ok": True}
            with self.candidate_lock:
                self.candidates.pop(identifier, None)
            return result

    def _admit_locked(self, candidate, identity_verified=False):
        if candidate.get("review_required") and identity_verified is not True:
            raise ValueError("Explicit operator identity verification is required for a one-reference proposal")
        if candidate["epoch"] != self.epoch or time.monotonic() > candidate["expires"]:
            raise ValueError("Candidate is stale; collect fresh registered-only evidence")
        camera = self.cameras.get(candidate["camera_id"])
        if not camera or camera.generation != candidate["camera_generation"]:
            raise ValueError("Candidate camera connection changed")
        return self.gallery.put(candidate["person_id"], "live", candidate["image"], candidate["vector"],
            candidate["quality"], dict(source="live", camera_id=candidate["camera_id"],
                admission=candidate["evidence"], track_id=candidate["track_id"],
                policy=candidate.get("policy", "registered_consensus"), operator_verified=identity_verified),
                expected_anchor_version=candidate["anchor_version"],
                expected_registered_revision=candidate["registered_revision"])

    def candidate_list(self):
        with self.candidate_lock:
            for key in list(self.candidates):
                if self.candidates[key]["expires"] < time.monotonic():
                    del self.candidates[key]
            return [{k: v for k, v in c.items() if k not in ("image", "vector", "epoch", "expires")}
                    for c in self.candidates.values()]

    def sample_photo(self, identifier):
        with self.samples_lock:
            sample = self.samples.get(identifier)
            if not sample or sample[0] < time.monotonic():
                raise ValueError("Track observation expired")
            return sample[1]

    def state(self):
        return dict(people=self.gallery.people(), cameras=[c.public() for c in list(self.cameras.values())],
            settings=self.settings, review=self.review.snapshot(), index=dict(ready=self.gallery.healthy, error=self.gallery.error,
            reference_count=len(self.gallery.references()), pending_file_cleanup=self.gallery.pending_cleanup_count()), engine=dict(self.vision.status(), ready=self.workers_ready) if self.vision else
            dict(ready=False, error=self.engine_error or "Loading models"),
            metrics=dict(uptime_seconds=round(time.time()-self.started), pending_recognition=len(self.pending),
                recognition_ms=round(sum(self.recognition_times)/len(self.recognition_times), 1) if self.recognition_times else None,
                error=self.engine_error, hardware=self.hardware,
                auto_admission_enabled=os.environ.get("CAMEX_OFFICE_ALLOW_AUTO_ADMISSION") == "1"))

    def _resources_loop(self):
        import psutil
        process = psutil.Process()
        while not self.stop_event.is_set():
            if not self.gallery.healthy:
                try:
                    self.gallery.sync()
                    self.engine_error = None
                except Exception:
                    pass
            self.gallery.cleanup_pending()
            self.candidate_list()
            with self.samples_lock:
                for identifier, (expires, _) in list(self.samples.items()):
                    if expires < time.monotonic():
                        del self.samples[identifier]
            memory = psutil.virtual_memory()
            result = dict(cpu_percent=psutil.cpu_percent(), ram_used_gb=round(memory.used/1e9, 2),
                ram_available_gb=round(memory.available/1e9, 2), process_rss_mb=round(process.memory_info().rss/1e6, 1),
                scope="Host GPU totals include other applications")
            try:
                output = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,temperature.gpu",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=3)
                row = output.stdout.strip().splitlines()[0].split(",")
                result.update(gpu=row[0].strip(), gpu_percent=float(row[1]), gpu_memory_mib=float(row[2]),
                              gpu_temperature_c=float(row[3]))
            except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
                result["gpu_percent"] = None
            self.hardware = result
            if result.get("gpu_temperature_c", 0) >= 85:
                self.engine_error = "Demo sources stopped by the 85 C software thermal guard"
                for camera in list(self.cameras.values()):
                    if camera.state != "stopped":
                        camera.stop()
            self.stop_event.wait(2)

    def close(self):
        self.stop_event.set()
        for camera in list(self.cameras.values()):
            camera.stop()
        for thread in self.threads:
            thread.join(timeout=3)
        self.review.close()
