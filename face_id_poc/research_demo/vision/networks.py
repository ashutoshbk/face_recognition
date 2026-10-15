"""ONNX model adapters; no model downloads or external service calls."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .vendor.scrfd import SCRFD
from .vendor.face_align import norm_crop

MODEL_HASHES = {
    "arcface-r50-w600k-112x112.onnx": "4c06341c33c2ca1f86781dab0e829f88ad5b64be9fba56e56bc9ebdefc619e43",
    "scrfd-kps-2.5g-640x640.onnx": "a2a876b5ad23e66a8b3973112ff137f98414c214b17c31a45955d26e54b58cbf",
    "yolox-tiny-crowdhuman-416x416.onnx": "7880da82285f9cedd92c46c3c8edece6e751ce8558f4be080ea6325d2a19d56e",
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def session(path, provider, threads=2):
    if provider not in ort.get_available_providers():
        raise RuntimeError(f"Requested provider unavailable: {provider}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.add_session_config_entry('session.intra_op.allow_spinning','0')
    opts.add_session_config_entry('session.inter_op.allow_spinning','0')
    opts.log_severity_level = 3
    sess = ort.InferenceSession(str(path), sess_options=opts, providers=[provider])
    if provider not in sess.get_providers():
        raise RuntimeError("GPU initialization failed; refusing silent CPU fallback")
    return sess


def nms(boxes, scores, threshold=0.45):
    if not len(boxes):
        return []
    order = np.argsort(-scores)
    areas = np.maximum(0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0, boxes[:, 3] - boxes[:, 1]
    )
    keep = []
    while len(order):
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        lo = np.maximum(boxes[i, :2], boxes[rest, :2])
        hi = np.minimum(boxes[i, 2:], boxes[rest, 2:])
        inter = np.prod(np.maximum(0, hi - lo), axis=1)
        iou = inter / np.maximum(areas[i] + areas[rest] - inter, 1e-6)
        order = rest[iou <= threshold]
    return keep


class Networks:
    def __init__(self, model_dir, provider="CUDAExecutionProvider", face_threshold=0.7, load_recognition=True):
        self.model_dir = Path(model_dir)
        self.provider=provider
        for name, expected in MODEL_HASHES.items():
            if sha256(self.model_dir / name) != expected:
                raise ValueError(f"Model checksum mismatch: {name}")
        self.head = session(
            self.model_dir / "yolox-tiny-crowdhuman-416x416.onnx", provider
        )
        self.face_session = session(
            self.model_dir / "scrfd-kps-2.5g-640x640.onnx", provider
        )
        self.face = SCRFD(session=self.face_session)
        self.face.det_thresh = face_threshold
        self.face.nms_thresh = 0.4
        self.rec = session(self.model_dir / "arcface-r50-w600k-112x112.onnx", provider) if load_recognition else None
        self.providers = {
            "head": self.head.get_providers(),
            "face": self.face_session.get_providers(),
            "recognition": self.rec.get_providers() if self.rec else [],
        }
        self.grid, self.strides = [], []
        for stride in [8, 16, 32]:
            h = w = 416 // stride
            yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
            self.grid.append(np.stack((xx, yy), -1).reshape(-1, 2))
            self.strides.append(np.full((h * w, 1), stride))
        self.grid = np.concatenate(self.grid).astype(np.float32)
        self.strides = np.concatenate(self.strides).astype(np.float32)

    def heads_and_bodies(self, frame, threshold=0.15):
        h, w = frame.shape[:2]
        ratio = min(416 / h, 416 / w)
        image = np.full((416, 416, 3), 114, dtype=np.uint8)
        resized = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
        image[: resized.shape[0], : resized.shape[1]] = resized
        blob = image.transpose(2, 0, 1)[None].astype(np.float32)
        pred = self.head.run(None, {self.head.get_inputs()[0].name: blob})[0][0].copy()
        # This YOLOX export uses raw grid-relative centers and log sizes.
        pred[:, :2] = (pred[:, :2] + self.grid) * self.strides
        pred[:, 2:4] = np.exp(np.clip(pred[:, 2:4], -20, 20)) * self.strides
        outputs = []
        for label in [1, 0]:
            scores = pred[:, 4] * pred[:, 5 + label]
            chosen = (scores >= threshold) & np.isfinite(pred).all(axis=1)
            p, scores = pred[chosen], scores[chosen]
            boxes = np.c_[p[:, :2] - p[:, 2:4] / 2, p[:, :2] + p[:, 2:4] / 2] / ratio
            boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, w - 1)
            boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, h - 1)
            keep = nms(boxes, scores)
            outputs.append(
                np.c_[boxes[keep], scores[keep]].astype(np.float32).reshape(-1, 5)
            )
        return outputs[0], outputs[1]

    def heads(self, frame, threshold=0.15):
        return self.heads_and_bodies(frame, threshold)[0]

    def faces(self, frame):
        boxes, landmarks = self.face.detect(frame)
        if landmarks is None:
            return []
        return [
            {
                "bbox": b[:4].astype(float),
                "score": float(b[4]),
                "landmarks": k.astype(np.float32),
            }
            for b, k in zip(boxes, landmarks)
        ]

    def embedding(self, frame, landmarks):
        crop = norm_crop(frame, np.asarray(landmarks, np.float32), image_size=112)
        return self.embedding_crop(crop), crop

    def embedding_crop(self,crop):
        if self.rec is None:
            self.rec=session(self.model_dir/'arcface-r50-w600k-112x112.onnx',self.provider)
            self.providers['recognition']=self.rec.get_providers()
        return encode_arcface_crop(self.rec,crop)


def encode_arcface_crop(rec,crop):
    if crop.shape!=(112,112,3):
        raise ValueError('Recognition requires an aligned 112x112 color crop')
    blob = cv2.dnn.blobFromImage(
        crop, 1 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True
    )
    vector = (
        rec.run(None, {rec.get_inputs()[0].name: blob})[0]
        .reshape(-1)
        .astype(np.float32)
    )
    length = float(np.linalg.norm(vector))
    if not np.isfinite(vector).all() or length < 1e-8 or vector.size != 512:
        raise ValueError("Invalid face embedding")
    return vector / length


def quality(frame, face, cfg):
    box = np.asarray(face["bbox"], float)
    k = np.asarray(face["landmarks"], float)
    if (
        box.shape != (4,)
        or k.shape != (5, 2)
        or not np.isfinite(box).all()
        or not np.isfinite(k).all()
        or (box[2:] <= box[:2]).any()
    ):
        return {"usable": False, "reasons": ["invalid_geometry"], "rank": 0.0}
    width, height = box[2:] - box[:2]
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = np.rint(box).astype(int)
    x1, x2 = np.clip([x1, x2], 0, w)
    y1, y2 = np.clip([y1, y2], 0, h)
    intereye = float(np.linalg.norm(k[1] - k[0]))
    mid = (k[1] + k[0]) / 2
    axis = (k[1] - k[0]) / max(intereye, 1e-6)
    asymmetry = abs(float(np.dot(k[2] - mid, axis))) / max(intereye, 1e-6)
    crop = frame[y1:y2, x1:x2]
    sharpness = (
        0.0
        if crop.size == 0
        else float(
            cv2.Laplacian(
                cv2.cvtColor(cv2.resize(crop, (112, 112)), cv2.COLOR_BGR2GRAY),
                cv2.CV_64F,
            ).var()
        )
    )
    reasons = []
    if width < cfg["face_min_width"] or height < cfg["face_min_width"]:
        reasons.append("small")
    if intereye < cfg["face_min_intereye"]:
        reasons.append("landmarks_small")
    if sharpness < cfg["face_min_sharpness"]:
        reasons.append("blurred")
    if asymmetry > cfg["face_max_pose_asymmetry"]:
        reasons.append("profile")
    if box[0] < 2 or box[1] < 2 or box[2] > w - 2 or box[3] > h - 2:
        reasons.append("clipped")
    if (k < box[:2] - np.array([width, height]) * 0.1).any() or (
        k > box[2:] + np.array([width, height]) * 0.1
    ).any():
        reasons.append("invalid_landmarks")
    rank = float(width * min(sharpness / 100, 2) / (1 + asymmetry))
    return {
        "usable": not reasons,
        "reasons": reasons,
        "width_px": float(width),
        "intereye_px": intereye,
        "sharpness": sharpness,
        "pose_asymmetry": asymmetry,
        "rank": rank,
    }
