"""Bounded multi-camera capture, tracking and independent recognition workers."""
from __future__ import annotations

import json
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

from domain import TrackEvidence, write_json
from vision_backend import Vision, HeadTracker, associate_faces, encode_jpeg


DEFAULTS = dict(match_threshold=.45, match_margin=.06, admission_threshold=.68,
                admission_margin=.10, adaptation_mode="review", analysis_fps=10.,
                recognition_hz=1., face_min_width=70)


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
    return bool(track and track.get("face_visible") and track.get("token") == job.get("track_token")
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

    def start(self):
        if self.thread and self.thread.is_alive():
            return
        self.stop_event.clear()
        with self.lock:
            self.process_times.clear()
            self.inference_times.clear()
            self.capture_times.clear()
            self.total_processed = 0
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
            tracks = [{k: v for k, v in t.items() if k not in ("evidence", "last_seen", "token")} for t in self.tracks.values()]
            with self.capture_lock:
                capture = [t for t in self.capture_times if time.monotonic()-t <= 5]
            capture_fps = (len(capture)-1)/(capture[-1]-capture[0]) if len(capture)>1 else 0
            return dict(id=self.id, name=self.name, url_redacted=redacted_url(self.url), state=self.state,
                error=self.error, generation=self.generation, dimensions=self.dimensions, tracks=tracks,
                decoder=self.decoder, metrics=dict(fps=round(fps, 2), capture_fps=round(capture_fps,2), processed_frames=self.total_processed,
                frame_age_ms=round((time.monotonic()-self.frame_time)*1000) if self.frame_time else None,
                inference_ms=round(sum(self.inference_times)/len(self.inference_times), 1) if self.inference_times else None))


class Manager:
    def __init__(self, root, models, provider, gallery):
        self.root, self.models, self.provider = Path(root), Path(models), provider
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

    def start(self):
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
        with self.pending_lock:
            self.pending.clear()
        with self.candidate_lock:
            self.candidates.clear()
        with self.samples_lock:
            self.samples.clear()
        for camera in list(self.cameras.values()):
            with camera.lock:
                for track in camera.tracks.values():
                    track["evidence"].clear()
                    track.update(name=None, state="Unknown", votes=0, admission=False)

    def update_settings(self, changes):
        ranges = dict(match_threshold=(0, 1), match_margin=(0, .5), admission_threshold=(0, 1),
            admission_margin=(0, .5), analysis_fps=(1, 30), recognition_hz=(.2, 5), face_min_width=(40, 300))
        proposed = dict(self.settings)
        for key, value in changes.items():
            if key == "adaptation_mode":
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
        if proposed["admission_threshold"] < proposed["match_threshold"] or proposed["admission_margin"] < proposed["match_margin"]:
            raise ValueError("Admission must be at least as strict as recognition")
        with self.admission_lock:
            self.settings = proposed
            self.reset_evidence()
            write_json(self.root / "config" / "settings.json", self.settings)
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
                    heads, faces = self.vision.detect(frame)
                    tracks = tracker.update(heads, frame.shape, int(captured*1000))
                    associations, ambiguous = associate_faces(tracks, faces)
                    duration = (time.perf_counter()-tick)*1000
                    with camera.lock:
                        if generation != camera.generation or camera.state != "running" or time.monotonic()-captured > 2:
                            continue
                        previous = camera.tracks
                        current = {}
                        for track_index, item in enumerate(tracks):
                            key = str(item["track_id"])
                            old = previous.get(key, {})
                            t = dict(id=key, bbox=[float(x) for x in item["bbox"]],
                                evidence=old.get("evidence", TrackEvidence()), last_seen=now,
                                token=old.get("token", uuid.uuid4().hex),
                                name=old.get("name"), state=old.get("state", "Unknown"),
                                votes=old.get("votes", 0), observations=old.get("observations", 0),
                                score=old.get("score"), admission=False, face_visible=False,
                                samples=old.get("samples", []),
                                recognition_ms=old.get("recognition_ms"), search_ms=old.get("search_ms"))
                            t.update(t["evidence"].summary(now))
                            face_index = associations.get(track_index)
                            if track_index in ambiguous or face_index is None:
                                t["evidence"].clear()
                                t.update(name=None, state="No usable face", token=uuid.uuid4().hex, votes=0, observations=0)
                            else:
                                face = faces[face_index]
                                quality, aligned, crop = self.vision.describe(frame, face, int(self.settings["face_min_width"]))
                                t["quality"], t["face_visible"] = quality, quality["accepted"]
                                if not quality["accepted"]:
                                    t["evidence"].clear()
                                    t.update(name=None, state="Quality rejected", token=uuid.uuid4().hex, votes=0, observations=0)
                                elif now-camera.last_job.get(key, -1e9) >= 1/self.settings["recognition_hz"]:
                                    job = dict(camera=camera, track_id=key, generation=generation, epoch=self.epoch,
                                        track_token=t["token"], sequence=seq, captured=captured,
                                        aligned=aligned, crop=crop, quality=quality)
                                    with self.pending_lock:
                                        self.pending[(camera.id, key)] = job
                                        while len(self.pending) > 128:
                                            self.pending.popitem(last=False)
                                    camera.last_job[key] = now
                            current[key] = t
                        camera.tracks = current
                        camera.process_times.append(time.monotonic())
                        camera.total_processed += 1
                        camera.inference_times.append(duration)
                        camera.preview = self._draw(camera, frame)
                except Exception as error:
                    with camera.lock:
                        camera.error = "Analysis error: " + str(error)[:180]
                        camera.tracks.clear()
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
            x1, y1, x2, y2 = [int(v*ratio) for v in track["bbox"]]
            confirmed = track["state"] == "Confirmed"
            color = (80, 180, 40) if confirmed else (20, 180, 240)
            state_label = {"No usable face": "No face", "Quality rejected": "Low quality",
                           "Recognition unavailable": "Offline"}.get(track["state"], track["state"])
            label = track["name"] or state_label
            if track["name"] and not confirmed:
                label = "? " + label
            text = f'#{track["id"]} {label[:26]}'
            if track["observations"]:
                text += f' {track["votes"]}/{track["observations"]}'
            cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)
            text_width = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, .45, 1)[0][0]
            text_x = min(max(1, x1), max(1, width-text_width-3))
            cv2.putText(view, text, (text_x, max(18, y1-5)), cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
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
                    track = camera.tracks.get(job["track_id"])
                    if not job_current(job, camera, track, self.epoch, time.monotonic()):
                        continue
                    result = track["evidence"].add(ranked, job["captured"], job["sequence"], self.settings)
                    track.update(result, recognition_ms=round(total_ms, 1), search_ms=round(search_ms, 2))
                    identifier = uuid.uuid4().hex
                    with self.samples_lock:
                        self.samples[identifier] = (time.monotonic()+10, encode_jpeg(job["crop"]))
                        for key in list(self.samples):
                            if self.samples[key][0] < time.monotonic():
                                del self.samples[key]
                        while len(self.samples) > 250:
                            self.samples.popitem(last=False)
                    track["samples"] = [s for s in track["samples"] if s["timestamp"] > time.monotonic()-8][-4:]
                    track["samples"].append(dict(id=identifier, url=f"/api/samples/{identifier}/photo",
                        quality_score=job["quality"]["score"], timestamp=job["captured"],
                        candidate=result["name"], state=result["state"]))
                if result["admission"] and self.settings["adaptation_mode"] != "off":
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

    def _offer_reference(self, job, result, vector, anchor_version, registered_revision):
        if job["epoch"] != self.epoch or anchor_version is None:
            return
        person_id = result["person_id"]
        now = time.monotonic()
        if now-self.cooldowns.get(person_id, -1e9) < 10:
            return
        self.cooldowns[person_id] = now
        c = dict(id=uuid.uuid4().hex, person_id=person_id, person_name=result["name"],
            score=job["quality"]["score"], quality=job["quality"], image=encode_jpeg(job["crop"]),
            vector=vector, reason="Registered-only consensus passed; awaiting review",
            epoch=job["epoch"], anchor_version=anchor_version, registered_revision=registered_revision,
            expires=now+120, camera_id=job["camera"].id,
            camera_generation=job["generation"], track_id=job["track_id"],
            evidence=result["admission_details"])
        c["url"] = "/api/candidates/"+c["id"]+"/photo"
        if self.settings["adaptation_mode"] == "auto":
            self._admit(c)
        else:
            with self.candidate_lock:
                self.candidates[c["id"]] = c
                while len(self.candidates) > 50:
                    self.candidates.popitem(last=False)

    def _admit(self, candidate):
        with self.admission_lock:
            return self._admit_locked(candidate)

    def _admit_locked(self, candidate):
        if candidate["epoch"] != self.epoch or time.monotonic() > candidate["expires"]:
            raise ValueError("Candidate is stale; collect fresh registered-only evidence")
        camera = self.cameras.get(candidate["camera_id"])
        if not camera or camera.generation != candidate["camera_generation"]:
            raise ValueError("Candidate camera connection changed")
        return self.gallery.put(candidate["person_id"], "live", candidate["image"], candidate["vector"],
            candidate["quality"], dict(source="live", camera_id=candidate["camera_id"],
                admission=candidate["evidence"], track_id=candidate["track_id"]),
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
            settings=self.settings, index=dict(ready=self.gallery.healthy, error=self.gallery.error,
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
