"""Separate detection and recognition sessions, using the existing model exports."""
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

RESEARCH = Path(__file__).resolve().parents[1] / "research_demo"
if str(RESEARCH) not in sys.path:
    sys.path.append(str(RESEARCH))
from vision.networks import MODEL_HASHES, sha256, nms, encode_arcface_crop
from vision.vendor.scrfd import SCRFD
from vision.vendor.face_align import norm_crop
from vision.head_tracker import HeadTracker
from vision.association import associate_faces
from core import assess_quality

FACE_MODELS = {
    "scrfd_2.5g": ("scrfd-kps-2.5g-640x640.onnx", MODEL_HASHES["scrfd-kps-2.5g-640x640.onnx"]),
    "scrfd_10g": ("det_10g.onnx", "5838f7fe053675b1c7a08b633df49e7af5495cee0493c7dcf6697200b85b5b91"),
}


def face_model_spec(name):
    if name not in FACE_MODELS:
        raise ValueError("Unsupported face detector: " + str(name))
    return FACE_MODELS[name]


def assess_query_quality(frame, face, minimum_width=70, strictness=50):
    """Tune query eligibility without weakening reference-photo admission."""
    reference = assess_quality(frame, face, minimum_width=max(70, minimum_width))
    query = dict(reference, reasons=list(reference["reasons"]))
    if "Outside frame" in query["reasons"]:
        return query, reference
    level = float(strictness) / 100
    eye1, eye2, nose = face[4:6], face[6:8], face[8:10]
    eye_distance = max(float(np.linalg.norm(eye1-eye2)), 1.)
    yaw = abs(float(nose[0]-(eye1[0]+eye2[0])/2)) / eye_distance
    roll = abs(float(eye1[1]-eye2[1])) / eye_distance
    reasons = [r for r in query["reasons"] if r not in ("Face too small", "Blur", "Lighting", "Head angle")]
    if face[2] < minimum_width:
        reasons.append("Face too small")
    if query["sharpness"] < 10 + 70*level:
        reasons.append("Blur")
    if query["brightness"] < 20 + 40*level or query["brightness"] > 240 - 40*level:
        reasons.append("Lighting")
    if yaw > .6 - .4*level or roll > .5 - .3*level:
        reasons.append("Head angle")
    # Preserve the exact existing result at the default, including boundary rounding.
    if strictness == 50 and minimum_width >= 70:
        return dict(reference, reasons=list(reference["reasons"])), reference
    query.update(accepted=not reasons, reasons=reasons)
    return query, reference


def decode_upload(data):
    if not data or len(data) > 12_000_000:
        raise ValueError("Image must be between 1 byte and 12 MB")
    with Image.open(io.BytesIO(data)) as image:
        if image.format not in ("JPEG", "PNG") or image.width * image.height > 24_000_000:
            raise ValueError("Use JPEG/PNG, at most 24 megapixels")
        if min(image.size) < 80:
            raise ValueError("Image is too small")
        return cv2.cvtColor(np.asarray(ImageOps.exif_transpose(image).convert("RGB")), cv2.COLOR_RGB2BGR)


def encode_jpeg(image):
    ok, data = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 94])
    if not ok:
        raise ValueError("Cannot encode image")
    return data.tobytes()


def face_crop(frame, face):
    x, y, w, h = face[:4]
    pad = max(w, h) * .16
    return frame[max(0, int(y-pad)):min(frame.shape[0], int(y+h+pad)),
                 max(0, int(x-pad)):min(frame.shape[1], int(x+w+pad))].copy()


def padded_enrollment_faces(frame, detector):
    """Retry small uploads at two scales without inventing native face pixels."""
    height, width = frame.shape[:2]
    longest = max(height, width)
    if longest >= 640:
        return []
    faces = []
    sides = sorted({min(640, max(256, int(np.ceil(longest*scale/64))*64))
                    for scale in (1.3, 1.6)})
    for side in sides:
        padded = np.zeros((side, side, 3), dtype=frame.dtype)
        padded[:height, :width] = frame
        boxes, landmarks = detector.detect(padded, input_size=(640, 640))
        if landmarks is None:
            continue
        for box, points in zip(boxes, landmarks):
            if (not np.isfinite(box).all() or not np.isfinite(points).all()
                    or box[0] < 0 or box[1] < 0 or box[2] > width or box[3] > height
                    or box[2] <= box[0] or box[3] <= box[1]
                    or (points < 0).any() or (points[:, 0] > width).any()
                    or (points[:, 1] > height).any()):
                continue
            faces.append(np.r_[box[:2], box[2:4]-box[:2], points.reshape(-1), box[4]].astype(np.float32))
    return faces


def head_detail_regions(heads, faces, shape, cursor=0, limit=4):
    """Bounded native-pixel crops; never alter the reported native face size."""
    height, width = shape[:2]
    candidates = []
    for head in heads:
        x1, y1, x2, y2 = map(float, head[:4])
        if x2 <= x1 or y2 <= y1:
            continue
        padding = .15*np.array([x2-x1, y2-y1])
        # A high-confidence sufficiently large full-frame face needs no extra pass.
        adequate = any(x1-padding[0] <= f[0]+f[2]/2 <= x2+padding[0] and
                       y1-padding[1] <= f[1]+f[3]/2 <= y2+padding[1] and
                       f[2] >= 70 and f[-1] >= .85 for f in faces)
        if adequate:
            continue
        side = min(640, max(128, int(max(x2-x1, y2-y1)*2.2)))
        cx, cy = (x1+x2)/2, (y1+y2)/2
        left, top = max(0, int(cx-side/2)), max(0, int(cy-side/2))
        right, bottom = min(width, int(cx+side/2)), min(height, int(cy+side/2))
        region = (left, top, right, bottom)
        if min(right-left, bottom-top) >= 32 and region not in candidates:
            candidates.append(region)
    if not candidates:
        return []
    start = cursor % len(candidates)
    return (candidates[start:] + candidates[:start])[:limit]


class Vision:
    def __init__(self, model_dir, provider, profile_dir):
        import onnxruntime as ort
        cv2.setNumThreads(2)
        if provider not in ort.get_available_providers():
            raise RuntimeError("Requested ONNX execution provider is unavailable: " + provider)
        self.provider = provider
        self.detect_lock = threading.BoundedSemaphore(2)
        self.local = threading.local()
        self.encode_lock = threading.Lock()
        self.sessions = {}
        self.face_model = os.environ.get("CAMEX_OFFICE_FACE_MODEL", "scrfd_2.5g")
        face_filename, face_hash = face_model_spec(self.face_model)
        self.profiling = os.environ.get("CAMEX_OFFICE_PROFILE", "0") == "1"
        profile_dir = Path(profile_dir)
        profile_dir.mkdir(parents=True, exist_ok=True)
        hashes = dict(MODEL_HASHES)
        hashes.pop("scrfd-kps-2.5g-640x640.onnx")
        hashes[face_filename] = face_hash
        for name, expected in hashes.items():
            path = Path(model_dir) / name
            if not path.exists() or sha256(path) != expected:
                raise RuntimeError("Missing or incompatible model: " + name)
            options = ort.SessionOptions()
            options.intra_op_num_threads = 2
            options.inter_op_num_threads = 1
            options.enable_profiling = self.profiling
            options.profile_file_prefix = str(profile_dir / name.replace(".onnx", ""))
            options.log_severity_level = 3
            options.add_session_config_entry("session.intra_op.allow_spinning", "0")
            session = ort.InferenceSession(str(path), sess_options=options, providers=[provider])
            if session.get_providers()[0] != provider:
                raise RuntimeError("Requested provider initialization failed: " + name)
            self.sessions[name] = session
        self.head = self.sessions["yolox-tiny-crowdhuman-416x416.onnx"]
        self.face_session = self.sessions[face_filename]
        self.face = SCRFD(session=self.face_session)
        self.face.det_thresh = .7
        self.face.nms_thresh = .4
        self.rec = self.sessions["arcface-r50-w600k-112x112.onnx"]
        grids, strides = [], []
        for stride in (8, 16, 32):
            yy, xx = np.meshgrid(np.arange(416//stride), np.arange(416//stride), indexing="ij")
            grids.append(np.stack((xx, yy), -1).reshape(-1, 2))
            strides.append(np.full((len(grids[-1]), 1), stride))
        self.grid = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)
        self.calls = {"head": 0, "face": 0, "face_detail": 0, "recognition": 0}
        self.saved_profile = None

    def detect(self, frame, enrollment=False, detail=False):
        with self.detect_lock:
            detector = getattr(self.local, "face", None)
            if detector is None:
                detector = SCRFD(session=self.face_session)
                detector.det_thresh, detector.nms_thresh = .7, .4
                self.local.face = detector
            h, w = frame.shape[:2]
            ratio = min(416/h, 416/w)
            canvas = np.full((416, 416, 3), 114, np.uint8)
            resized = cv2.resize(frame, (int(w*ratio), int(h*ratio)))
            canvas[:resized.shape[0], :resized.shape[1]] = resized
            pred = self.head.run(None, {self.head.get_inputs()[0].name:
                canvas.transpose(2, 0, 1)[None].astype(np.float32)})[0][0].copy()
            self.calls["head"] += 1
            pred[:, :2] = (pred[:, :2] + self.grid) * self.strides
            pred[:, 2:4] = np.exp(np.clip(pred[:, 2:4], -20, 20)) * self.strides
            scores = pred[:, 4] * pred[:, 6]
            chosen = (scores >= .35) & np.isfinite(pred).all(axis=1)
            p, scores = pred[chosen], scores[chosen]
            boxes = np.c_[p[:, :2]-p[:, 2:4]/2, p[:, :2]+p[:, 2:4]/2] / ratio
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w-1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h-1)
            keep = nms(boxes, scores)
            heads = np.c_[boxes[keep], scores[keep]].astype(np.float32).reshape(-1, 5)
            # SCRFD is a multi-face detector. One pass avoids serial inference
            # for every head; returned landmarks still map to native pixels.
            rois = [(0, 0, w, h)]
            faces = []
            def run_roi(region, is_detail=False):
                x1, y1, x2, y2 = region
                if min(x2-x1, y2-y1) < 12:
                    return
                detected, landmarks = detector.detect(frame[y1:y2, x1:x2], input_size=(640, 640))
                self.calls["face"] += 1
                self.calls["face_detail"] += int(is_detail)
                if landmarks is None:
                    return
                for box, points in zip(detected, landmarks):
                    if is_detail and (box[0] < 2 or box[1] < 2 or box[2] > x2-x1-2 or box[3] > y2-y1-2):
                        continue
                    xy = box[:4] + np.array([x1, y1, x1, y1])
                    points = points + np.array([x1, y1])
                    faces.append(np.r_[xy[:2], xy[2:]-xy[:2], points.reshape(-1), box[4]].astype(np.float32))
            run_roi(rois[0])
            # Small native video crops can fail when stretched to fill 640px.
            # Only retry zero-face uploads; normal multi-face rejection stays intact.
            if enrollment and not faces:
                faces.extend(padded_enrollment_faces(frame, detector))
            if detail and not enrollment:
                cursor = getattr(self.local, "detail_cursor", 0)
                regions = head_detail_regions(heads, faces, frame.shape, cursor=cursor, limit=1)
                self.local.detail_cursor = cursor+len(regions)
                for region in regions:
                    run_roi(region, is_detail=True)
            if len(faces) > 1:
                f = np.stack(faces)
                keep = nms(np.c_[f[:, :2], f[:, :2]+f[:, 2:4]], f[:, -1], .35)
                faces = [faces[k] for k in keep]
            return heads, faces

    def describe(self, frame, face, minimum_width=70):
        quality = assess_quality(frame, face, minimum_width=minimum_width)
        aligned = norm_crop(frame, face[4:14].reshape(5, 2), image_size=112)
        return quality, aligned, face_crop(frame, face)

    def encode(self, aligned):
        with self.encode_lock:
            self.calls["recognition"] += 1
            return encode_arcface_crop(self.rec, aligned)

    def profile_summary(self):
        if not self.profiling:
            return {"enabled": False, "note": "Run scripts/check_runtime.py --exercise-models for isolated CUDA profiling"}
        if self.saved_profile is not None:
            return self.saved_profile
        result = {}
        for name, session in self.sessions.items():
            path = session.end_profiling()
            try:
                events = json.loads(Path(path).read_text())
                providers = {}
                for event in events:
                    provider = event.get("args", {}).get("provider")
                    if provider:
                        providers[provider] = providers.get(provider, 0) + 1
                result[name] = {"provider_node_events": providers, "profile": Path(path).name}
            except (ValueError, OSError):
                result[name] = {"error": "Profile could not be read"}
        self.saved_profile = result
        return result

    def status(self):
        return {"ready": True, "requested_provider": self.provider,
                "face_model": self.face_model, "face_input": [640, 640],
                "providers": {name: s.get_providers() for name, s in self.sessions.items()},
                "profiling": self.profiling, "calls": dict(self.calls)}
