"""Single-consenting-tester capture and selected-template verification only."""
import base64
import io
import json
import sqlite3
import threading
import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from vision.networks import Networks
from vision.head_tracker import HeadTracker
from vision.association import associate_faces, TurnAwayCheck
from vision.vendor.face_align import norm_crop

MODEL_ID = 'arcface-r50-w600k-512d-4c06341c'
HEAD_CONTINUITY_SECONDS = 2.0
CAPTURE_WINDOW_SECONDS = 30.0
CAPTURE_FRESHNESS_SECONDS = 2.5

cv2.setNumThreads(2)


def unit(vector):
    vector = np.asarray(vector, dtype=np.float32).reshape(-1)
    length = np.linalg.norm(vector)
    if not np.isfinite(length) or length < 1e-8:
        raise ValueError("Invalid face descriptor")
    return vector / length


def jpeg(image):
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("Image encoding failed")
    return encoded.tobytes()


def data_url(image_bytes):
    return "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()


def decode_image(data):
    if not data or len(data) > 2_000_000:
        raise ValueError("Frame must be a JPEG smaller than 2 MB")
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format != "JPEG" or max(image.size) > 1920 or min(image.size) < 80:
                raise ValueError("Expected a JPEG frame between 80 and 1920 pixels per side")
    except (OSError, Image.DecompressionBombError) as error:
        raise ValueError("Invalid JPEG") from error
    decoded = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError("Invalid JPEG")
    return decoded


class Engine:
    def __init__(self, models):
        self.lock = threading.Lock()
        self.recognition_lock = threading.Lock()
        self.networks = Networks(models, provider='CPUExecutionProvider', face_threshold=0.7,load_recognition=False)

    def detect(self, frame, crop_faces=False):
        with self.lock:
            tick = time.perf_counter()
            heads, bodies = self.networks.heads_and_bodies(frame, threshold=.15)
            head_ms = (time.perf_counter()-tick)*1000
            tick = time.perf_counter()
            dx=dy=0
            face_image=frame
            confident=heads[heads[:,4]>=.4]
            if crop_faces and len(confident)==1:
                x1,y1,x2,y2=confident[0,:4]
                pad_x=max(16,(x2-x1)*.35);pad_y=max(16,(y2-y1)*.35)
                dx=max(0,int(x1-pad_x));dy=max(0,int(y1-pad_y))
                right=min(frame.shape[1],int(x2+pad_x));bottom=min(frame.shape[0],int(y2+pad_y))
                face_image=frame[dy:bottom,dx:right]
            raw_faces = self.networks.faces(face_image)
            for face in raw_faces:
                face['bbox']+=np.array([dx,dy,dx,dy])
                face['landmarks']+=np.array([dx,dy])
            face_ms = (time.perf_counter()-tick)*1000
        faces = [np.r_[f['bbox'][:2], f['bbox'][2:]-f['bbox'][:2], f['landmarks'].reshape(-1), f['score']].astype(np.float32) for f in raw_faces]
        return {'heads':heads,'faces':faces,'bodies':bodies,'head_ms':head_ms,'face_ms':face_ms,
                'face_checked':True,'face_input_pixels':int(face_image.shape[0]*face_image.shape[1])}

    def detect_roi(self,frame):
        return self.detect(frame,crop_faces=True)

    def detect_heads(self,frame):
        with self.lock:
            tick=time.perf_counter()
            heads,bodies=self.networks.heads_and_bodies(frame,threshold=.15)
        return {'heads':heads,'faces':[],'bodies':bodies,'head_ms':(time.perf_counter()-tick)*1000,
                'face_ms':0,'face_checked':False}

    def encode(self, frame, face):
        with self.recognition_lock:
            feature, aligned = self.networks.embedding(frame, face[4:14].reshape(5,2))
        return aligned, feature

    def encode_crop(self,crop):
        with self.recognition_lock:
            return self.networks.embedding_crop(crop)


def align_face(frame,face):
    return norm_crop(frame,np.asarray(face[4:14],np.float32).reshape(5,2),image_size=112)


def assess_quality(frame, face, minimum_width=70, minimum_sharpness=45):
    height, width = frame.shape[:2]
    x, y, w, h = [float(v) for v in face[:4]]
    x0, y0 = max(0, int(x)), max(0, int(y))
    x1, y1 = min(width, int(x + w)), min(height, int(y + h))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return {"accepted": False, "score": 0.0, "reasons": ["Outside frame"], "width": w, "sharpness": 0, "brightness": 0, "frontal": 0}
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Fixed-size sharpness measurement avoids rewarding larger crops by resolution alone.
    sharpness = float(cv2.Laplacian(cv2.resize(gray, (112, 112)), cv2.CV_32F).var())
    brightness = float(gray.mean())
    eye1, eye2, nose = face[4:6], face[6:8], face[8:10]
    eye_distance = max(float(np.linalg.norm(eye1 - eye2)), 1.0)
    yaw_proxy = abs(float(nose[0] - (eye1[0] + eye2[0]) / 2)) / eye_distance
    roll_proxy = abs(float(eye1[1] - eye2[1])) / eye_distance
    frontal = max(0, 1 - yaw_proxy - roll_proxy)
    reasons = []
    if w < minimum_width:
        reasons.append("Face too small")
    if sharpness < minimum_sharpness:
        reasons.append("Blur")
    if brightness < 40 or brightness > 220:
        reasons.append("Lighting")
    if yaw_proxy > 0.4 or roll_proxy > 0.35:
        reasons.append("Head angle")
    if x < 2 or y < 2 or x + w > width - 2 or y + h > height - 2:
        reasons.append("Clipped face")
    if face[-1] < 0.7:
        reasons.append("Low detection score")
    score = 100 * (0.3 * min(w / 180, 1) + 0.3 * min(sharpness / 250, 1) + 0.25 * frontal + 0.15 * max(0, 1 - abs(brightness - 125) / 125))
    return {"accepted": not reasons, "score": round(score, 1), "reasons": reasons, "width": round(w), "sharpness": round(sharpness, 1), "brightness": round(brightness), "frontal": round(frontal, 2)}


class Gallery:
    def __init__(self, directory, max_testers=10):
        self.max_testers=max_testers
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / "consenting_testers_arcface_r50.sqlite3"
        self.legacy_path = directory / "consenting_testers.sqlite3"
        self.lock = threading.Lock()
        self.cache = {}
        with self.connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS testers (id TEXT PRIMARY KEY, name TEXT NOT NULL, created REAL NOT NULL, template BLOB NOT NULL, photo BLOB NOT NULL, samples INTEGER NOT NULL, consent INTEGER NOT NULL)")
        self.path.chmod(0o600)

    def connect(self):
        return sqlite3.connect(self.path)

    def legacy_count(self):
        if not self.legacy_path.exists():
            return 0
        with sqlite3.connect(f'file:{self.legacy_path}?mode=ro',uri=True) as db:
            return db.execute('SELECT COUNT(*) FROM testers').fetchone()[0]

    def list(self):
        with self.connect() as db:
            return [dict(id=r[0], name=r[1], created=r[2], samples=r[3]) for r in db.execute("SELECT id,name,created,samples FROM testers ORDER BY created DESC")]

    def get(self, identifier):
        with self.lock:
            if identifier not in self.cache:
                with self.connect() as db:
                    row = db.execute("SELECT id,name,template,photo,samples FROM testers WHERE id=? AND consent=1", (identifier,)).fetchone()
                if row is None:
                    raise ValueError("Tester not found")
                if len(row[2]) != 512*4:
                    raise ValueError('Incompatible template model; re-enrollment is required')
                self.cache[identifier] = {"id": row[0], "name": row[1], "template": np.frombuffer(row[2], dtype=np.float32).copy(), "photo": row[3], "samples": row[4]}
            return self.cache[identifier]

    def add(self, name, candidates, consent):
        chosen = sorted(candidates, key=lambda c: c["quality"], reverse=True)[:3]
        if not consent:
            raise ValueError("Explicit tester consent is required")
        if len(chosen) < 3:
            raise ValueError("Three accepted samples are required")
        if any(np.asarray(c['feature']).size != 512 for c in chosen):
            raise ValueError('Enrollment requires 512D ArcFace samples')
        template = unit(np.average([c["feature"] for c in chosen], axis=0, weights=[c["quality"] for c in chosen]))
        return self._store(name, template, chosen[0]['jpeg'], consent, len(chosen))

    def add_photo(self, name, feature, photo, consent):
        if np.asarray(feature).size != 512:
            raise ValueError('Enrollment requires a 512D ArcFace sample')
        return self._store(name, unit(feature), photo, consent, 1)

    def _store(self, name, template, photo, consent, sample_count):
        name = str(name).strip()
        if not consent:
            raise ValueError("Explicit tester consent is required")
        if not name or len(name) > 60:
            raise ValueError("Enter a tester name of 1-60 characters")
        identifier = str(uuid.uuid4())
        with self.lock, self.connect() as db:
            if db.execute("SELECT COUNT(*) FROM testers").fetchone()[0] >= self.max_testers:
                raise ValueError(f"This local gallery permits {self.max_testers} tester(s); delete the existing profile before enrolling another")
            db.execute("INSERT INTO testers VALUES (?,?,?,?,?,?,1)", (identifier, name, time.time(), template.tobytes(), photo, sample_count))
        return identifier

    def delete(self, identifier):
        with self.lock, self.connect() as db:
            db.execute("PRAGMA secure_delete=ON")
            if not db.execute("DELETE FROM testers WHERE id=?", (identifier,)).rowcount:
                raise ValueError("Tester not found")
            self.cache.pop(identifier, None)
        # Database deletion is not a guarantee of physical SSD/snapshot erasure.


class Evidence:
    def __init__(self):
        self.samples = deque(maxlen=5)

    def clear(self):
        self.samples.clear()

    def add(self, timestamp, score, threshold):
        if self.samples and timestamp - self.samples[-1]["timestamp"] < 0.35:
            return
        self.samples.append({"timestamp": timestamp, "score": float(score), "pass": bool(score >= threshold)})

    def summary(self, now):
        while self.samples and now - self.samples[0]["timestamp"] > 4:
            self.samples.popleft()
        values = list(self.samples)
        votes = sum(item["pass"] for item in values)
        ready = len(values) == 5 and values[-1]["timestamp"] - values[0]["timestamp"] >= 1.3
        return {"state": ("Match" if votes >= 3 else "No match") if ready else "Collecting evidence", "votes": votes, "required": 3, "count": len(values), "window": 5, "scores": [round(v["score"], 4) for v in values]}


class Session:
    def __init__(self, report_directory=None):
        self.id = str(uuid.uuid4())
        self.lock = threading.RLock()
        self.selected = None
        self.minimum_width = 70
        self.minimum_sharpness = 45
        self.threshold = 0.45
        self.last_used = time.monotonic()
        self.events = deque(maxlen=200)
        self.report_directory = Path(report_directory) if report_directory else None
        self.reset()

    def reset(self):
        self.tracker = HeadTracker(fps=10, threshold=.4, buffer_seconds=HEAD_CONTINUITY_SECONDS, geometry_fallback=True)
        self._capture_epoch = getattr(self, '_capture_epoch', 0) + 1
        self.capture_events = deque(maxlen=100)
        self.capture_reset_count = 0
        self.last_capture_reset = 'New capture session'
        self.candidates = []
        self.evidence = Evidence()
        self.active_id = None
        self.last_seen = 0
        self.last_sample = 0
        self.last_feature = None
        self.last_frame = None
        self.sequence = 0
        self.generation = uuid.uuid4().hex[:5]
        self.last_tracker_ms = -1
        self.head_since = 0
        self.turnaway = TurnAwayCheck()
        self.saved_turnaway = None
        self.current_face = None

    def clear_capture(self, reason='Capture reset'):
        if self.candidates:
            self.capture_reset_count += 1
            self.capture_events.append({'time':time.time(),'reason':reason,'samples_removed':len(self.candidates)})
        if reason != 'New head track' or self.last_capture_reset == 'New capture session':
            self.last_capture_reset = reason
        self._capture_epoch += 1
        self.candidates.clear()
        self.evidence.clear()
        self.last_sample = 0
        self.last_feature = None

    def configure(self, settings, gallery):
        with self.lock:
            if settings.get("selected"):
                gallery.get(settings["selected"])
            selected = settings.get('selected') or None
            if (selected == self.selected and settings['minimum_width'] == self.minimum_width
                    and settings['minimum_sharpness'] == self.minimum_sharpness and settings['threshold'] == self.threshold):
                return
            self.selected = selected
            self.minimum_width = settings["minimum_width"]
            self.minimum_sharpness = settings["minimum_sharpness"]
            self.threshold = settings["threshold"]
            self.clear_capture('Capture settings or profile changed')

    def capture_status(self, result):
        reason = ', '.join((result.get('quality') or {}).get('reasons', []))
        return {'samples':len(self.candidates),'required':3,'ready':result['enrollment_ready'],
            'state':'Ready' if result['enrollment_ready'] else 'Collecting' if result.get('quality',{} ) and result['quality']['accepted'] else 'Paused',
            'pause_reason':reason or (result['head_status'] if not result.get('face_associated') else ''),
            'last_reset':self.last_capture_reset,'reset_count':self.capture_reset_count,
            'sample_window_seconds':CAPTURE_WINDOW_SECONDS,'head_gap_limit_seconds':HEAD_CONTINUITY_SECONDS}

    def process(self, engine, frame, gallery, now=None, encode_samples=True, frame_id=None):
        now = time.monotonic() if now is None else now
        with self.lock:
            started = time.perf_counter()
            self.last_used = now
            if self.last_seen and now - self.last_seen >= HEAD_CONTINUITY_SECONDS:
                self.clear_capture('No continuous head observation for 2 seconds')
                self.active_id = None
            self.sequence += 1
            self.current_face=None
            detect_start = time.perf_counter()
            observations = engine.detect(frame)
            face_checked=observations.get('face_checked',True)
            faces, heads = observations['faces'], observations['heads']
            detect_ms = (time.perf_counter() - detect_start) * 1000
            timestamp_ms = max(int(now*1000), self.last_tracker_ms+1)
            tracks = self.tracker.update(heads,frame.shape,timestamp_ms)
            self.last_tracker_ms = timestamp_ms
            matches, ambiguous_heads = associate_faces(tracks,faces)
            ambiguous = bool(ambiguous_heads or self.tracker.association_ambiguous)
            head_count = max(len(tracks), int(sum(h[4] >= .4 for h in heads)))
            overlays = []
            for index, track in enumerate(tracks):
                box = np.asarray(track['bbox'])
                face = faces[matches[index]] if index in matches else None
                overlays.append({'id':f"H-{self.generation}-{track['track_id']:03d}",
                                 'box':[round(float(v),1) for v in np.r_[box[:2],box[2:]-box[:2]]],
                                 'face_box':[round(float(v),1) for v in face[:4]] if face is not None else None,
                                 'detector_score':round(float(track['score']),3)})
            retained = [c for c in self.candidates if now-c['timestamp'] < CAPTURE_WINDOW_SECONDS]
            if len(retained) != len(self.candidates):
                self.capture_events.append({'time':time.time(),'reason':'Accepted sample expired','samples_removed':len(self.candidates)-len(retained)})
            self.candidates = retained
            if now-self.last_sample > CAPTURE_WINDOW_SECONDS:
                self.last_feature = None
            result = {'sequence':self.sequence,'frame_id':frame_id,'width':frame.shape[1],'height':frame.shape[0],
                      'face_count':len(faces),'head_count':head_count,'tracks':overlays,'quality':None,'samples':[],
                      'active_id':None,'selected':self.selected,'head_status':'No head observed','head_age_seconds':0,
                      'verification':{'state':'No head observed','votes':0,'count':0,'window':5,'required':3,'scores':[]},
                      'enrollment_ready':False,'face_associated':False,
                      'timing':{'detection_ms':round(detect_ms,2),'head_detection_ms':round(observations.get('head_ms',0),2),
                                'face_detection_ms':round(observations.get('face_ms',0),2),'embedding_ms':0.0,'comparison_ms':0.0}}
            if head_count > 1 or len(faces) > 1 or ambiguous:
                self.clear_capture('Ambiguous head/face association' if ambiguous else 'Multiple heads or faces detected')
                self.active_id = None
                result['verification']['state'] = 'Ambiguous association: paused' if ambiguous else 'Multiple heads or faces: paused'
                result['head_status'] = result['verification']['state']
            elif len(tracks) != 1:
                self.evidence.clear()
                result['head_status'] = 'Head temporarily unobserved' if self.active_id else 'Acquiring head track' if head_count else 'No head observed'
                result['verification']['state'] = result['head_status']
            else:
                track_id = overlays[0]["id"]
                if track_id != self.active_id:
                    self.clear_capture('Head track changed' if self.active_id else 'New head track')
                    self.active_id = track_id
                    self.head_since = now
                self.last_seen = now
                result["active_id"] = track_id
                result['head_status'] = 'Head tracked'
                result['head_age_seconds'] = round(now-self.head_since,1)
                face = faces[matches[0]] if 0 in matches else None
                self.current_face=face
                result['face_associated'] = face is not None
                quality = assess_quality(frame,face,self.minimum_width,self.minimum_sharpness) if face is not None else None
                result["quality"] = quality
                if quality and quality["accepted"]:
                    if encode_samples and now - self.last_sample >= 0.35:
                        encode_start = time.perf_counter()
                        aligned, feature = engine.encode(frame, face)
                        result["timing"]["embedding_ms"] = round((time.perf_counter() - encode_start) * 1000, 2)
                        if self.last_feature is not None and float(np.dot(self.last_feature, feature)) < 0.35:
                            self.clear_capture('Face appearance consistency check changed')
                        self.last_feature = feature
                        self.last_sample = now
                        sample = {"timestamp": now, "quality": quality["score"], "feature": feature, "jpeg": jpeg(aligned)}
                        self.candidates = sorted(self.candidates + [sample], key=lambda c: c["quality"], reverse=True)[:5]
                        if self.selected:
                            compare_start = time.perf_counter()
                            try:
                                selected = gallery.get(self.selected)
                                score = float(np.dot(feature, selected["template"]))
                                self.evidence.add(now, score, self.threshold)
                            except ValueError:
                                self.selected = None
                                self.evidence.clear()
                            result["timing"]["comparison_ms"] = round((time.perf_counter() - compare_start) * 1000, 3)
                    result["verification"] = self.evidence.summary(now)
                    if not self.selected:
                        result["verification"]["state"] = "No tester selected"
                else:
                    self.evidence.clear()
                    result['verification']['state'] = 'Quality gate: paused' if face is not None else 'Head tracked / face unavailable'
                candidates = self.candidates
                result["samples"] = [{"quality": c["quality"], "age_seconds": round(now-c["timestamp"], 1), "image": data_url(c["jpeg"])} for c in candidates]
                span = max((c["timestamp"] for c in candidates), default=0) - min((c["timestamp"] for c in candidates), default=0)
                result["enrollment_ready"] = bool(quality and quality['accepted'] and len(candidates) >= 3 and span >= 0.65)
            if face_checked or head_count!=1 or ambiguous:
                self.turnaway.update(result['active_id'],bool(result['quality'] and result['quality']['accepted']),head_count,ambiguous or len(faces)>1,now)
            result['turnaway'] = self.turnaway.snapshot()
            if self.turnaway.result and self.report_directory and self.saved_turnaway != self.turnaway.test_id:
                self.report_directory.mkdir(parents=True,exist_ok=True,mode=0o700)
                report = {'timestamp':time.time(),'model_id':MODEL_ID,
                          'source':'Operator-initiated local check; camera contents not recorded or independently verified',
                          **self.turnaway.result}
                path = self.report_directory / f'head_continuity_{self.turnaway.test_id}.json'
                path.write_text(json.dumps(report,indent=2)+'\n')
                path.chmod(0o600)
                self.saved_turnaway = self.turnaway.test_id
            result["timing"]["total_ms"] = round((time.perf_counter() - started) * 1000, 2)
            result["selected"] = self.selected
            result['capture'] = self.capture_status(result)
            self.last_frame = result
            event = {"time": time.time(), "track": result["active_id"], "state": result["verification"]["state"], "quality": result["quality"]["score"] if result["quality"] else None, "processing_ms": result["timing"]["total_ms"]}
            if not self.events or self.events[-1]["state"] != event["state"]:
                self.events.append(event)
            return result

    def start_turnaway(self):
        with self.lock:
            if not self.last_frame or not self.last_frame['active_id'] or not self.last_frame['quality'] or not self.last_frame['quality']['accepted'] or time.monotonic()-self.last_seen > CAPTURE_FRESHNESS_SECONDS:
                raise ValueError('A current single head with a clear associated face is required')
            self.turnaway.start(self.active_id,time.monotonic())
            return self.turnaway.snapshot()

    def enroll(self, name, consent, track_id, gallery):
        with self.lock:
            if not self.last_frame or not self.last_frame["enrollment_ready"] or time.monotonic() - self.last_seen > CAPTURE_FRESHNESS_SECONDS:
                raise ValueError("A current, clear single-person capture is required")
            if self.active_id != track_id:
                raise ValueError("The active track changed; capture again")
            identifier = gallery.add(name, self.candidates, consent)
            self.selected = identifier
            self.clear_capture('Enrollment completed')
            return identifier
