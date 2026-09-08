import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from .models import MODEL_HASHES, sha256, quality
from .io_utils import claim_directory


def enroll(networks, photos, output, config):
    photos, output = Path(photos), Path(output)
    files = sorted(
        p for p in photos.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not files:
        raise ValueError("No enrollment photographs found")
    sources = [{"filename": p.name, "sha256": sha256(p)} for p in files]
    fingerprint = hashlib.sha256(
        json.dumps([sources, MODEL_HASHES, config, "align-v1"], sort_keys=True).encode()
    ).hexdigest()
    if (output / "gallery.json").exists():
        old = json.loads((output / "gallery.json").read_text())
        if old["fingerprint"] == fingerprint and (output / "vectors.npy").exists():
            return old
        raise ValueError(
            "Gallery output already contains different inputs; choose a new directory"
        )
    run_lock = claim_directory(output, fingerprint, protected=[photos])
    (output / "reference").mkdir(exist_ok=True)
    people, vectors, rejected = [], [], []
    enroll_cfg = dict(
        config,
        face_min_width=70,
        face_min_intereye=20,
        face_min_sharpness=15,
        face_max_pose_asymmetry=0.65,
    )
    for path in files:
        image = cv2.imread(str(path))
        if image is None:
            rejected.append({"filename": path.name, "reason": "unreadable"})
            continue
        faces = networks.faces(image)
        if len(faces) != 1:
            rejected.append(
                {
                    "filename": path.name,
                    "reason": f"expected one face, got {len(faces)}",
                }
            )
            continue
        q = quality(image, faces[0], enroll_cfg)
        if not q["usable"]:
            rejected.append({"filename": path.name, "reason": q["reasons"]})
            continue
        vector, crop = networks.embedding(image, faces[0]["landmarks"])
        person_id = hashlib.sha256(path.stem.encode()).hexdigest()[:12]
        relative = f"reference/{person_id}.jpg"
        if not cv2.imwrite(str(output / relative), crop):
            raise IOError("Failed to save reference crop")
        people.append(
            {
                "person_id": person_id,
                "display_name": path.stem,
                "photo": path.name,
                "reference": relative,
                "quality": q,
            }
        )
        vectors.append(vector)
    if not people:
        raise ValueError("No usable reference images")
    manifest = {
        "schema_version": "1.0",
        "fingerprint": fingerprint,
        "models": MODEL_HASHES,
        "preprocessing": "align-v1",
        "people": people,
        "sources": sources,
        "rejected": rejected,
    }
    np.save(output / "vectors.npy", np.asarray(vectors, np.float32), allow_pickle=False)
    manifest["vectors_sha256"] = sha256(output / "vectors.npy")
    (output / "gallery.json").write_text(json.dumps(manifest, indent=2))
    run_lock.close()
    return manifest


class Gallery:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.manifest = json.loads((self.folder / "gallery.json").read_text())
        if (
            self.manifest["models"] != MODEL_HASHES
            or self.manifest["preprocessing"] != "align-v1"
        ):
            raise ValueError("Enrollment and inference model/preprocessing mismatch")
        self.people = self.manifest["people"]
        if sha256(self.folder / "vectors.npy") != self.manifest["vectors_sha256"]:
            raise ValueError("Gallery vector checksum mismatch")
        self.vectors = np.load(self.folder / "vectors.npy", allow_pickle=False)
        if (
            self.vectors.shape != (len(self.people), 512)
            or not np.isfinite(self.vectors).all()
        ):
            raise ValueError("Malformed gallery embeddings")
        self.vectors = self.vectors / np.maximum(
            np.linalg.norm(self.vectors, axis=1, keepdims=True), 1e-8
        )
        self.by_id = {p["person_id"]: p for p in self.people}
        self.references = {
            p["person_id"]: cv2.imread(str(self.folder / p["reference"]))
            for p in self.people
        }

    def match(self, embedding):
        scores = self.vectors @ embedding
        # Rank distinct persons, not individual templates.
        best = {}
        for p, score in zip(self.people, scores):
            pid = p["person_id"]
            best[pid] = max(best.get(pid, -1.0), float(score))
        ranking = sorted(best.items(), key=lambda x: x[1], reverse=True)
        pid, score = ranking[0]
        second = ranking[1][1] if len(ranking) > 1 else -1.0
        return pid, score, score - second
