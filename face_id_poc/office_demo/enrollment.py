"""Explicit, previewed enrollment from photos or named folders."""
import threading
import time
import uuid
from pathlib import PurePosixPath

from domain import safe_name
from vision_backend import decode_upload, encode_jpeg


class Enrollment:
    def __init__(self, manager):
        self.manager = manager
        self.previews = {}
        self.lock = threading.Lock()

    def prepare(self, files, name=None, person_id=None, consent=False):
        if not consent:
            raise ValueError("Confirm authorization and consent for these enrollment photos")
        if not self.manager.vision:
            raise ValueError("Models are not ready")
        if not files or len(files) > 200 or sum(len(data) for _, data in files) > 128_000_000:
            raise ValueError("Import 1-200 images, at most 128 MB in total")
        selected = next((p for p in self.manager.gallery.people() if p["id"] == person_id), None) if person_id else None
        if person_id and not selected:
            raise ValueError("Selected person no longer exists")
        if selected:
            name = selected["name"]
        existing = {p["name"]: p["id"] for p in self.manager.gallery.people()}
        if name:
            name = safe_name(name)
        groups = {}
        with self.lock:
            for path, data in files:
                parts = PurePosixPath(path.replace("\\", "/")).parts
                if not parts or PurePosixPath(path).is_absolute() or ".." in parts:
                    raise ValueError("Invalid relative image path")
                if name:
                    group_name = name
                elif len(parts) >= 2:
                    # Browser folder uploads include the selected parent folder.
                    group_name = safe_name(parts[-2])
                else:
                    group_name = safe_name(PurePosixPath(parts[-1]).stem)
                effective_id = person_id or existing.get(group_name)
                group = groups.setdefault(group_name, dict(name=group_name, person_id=effective_id, items=[]))
                item = dict(filename=parts[-1], accepted=False, reason=None)
                try:
                    if PurePosixPath(path).suffix.lower() not in (".jpg", ".jpeg", ".png"):
                        raise ValueError("Only JPEG and PNG images are supported")
                    frame = decode_upload(data)
                    heads, faces = self.manager.vision.detect(frame, enrollment=True)
                    if len(faces) != 1:
                        raise ValueError("Enrollment image must contain exactly one clearly visible face")
                    quality, aligned, crop = self.manager.vision.describe(frame, faces[0],
                        int(self.manager.settings["face_min_width"]))
                    item["quality"] = quality
                    if not quality["accepted"]:
                        raise ValueError("; ".join(quality["reasons"]))
                    item.update(accepted=True, vector=self.manager.vision.encode(aligned),
                        image=encode_jpeg(frame))
                except Exception as error:
                    item["reason"] = str(error)[:220]
                group["items"].append(item)
            for token in list(self.previews):
                if self.previews[token]["expires"] < time.monotonic():
                    del self.previews[token]
            if len(self.previews) >= 3:
                del self.previews[next(iter(self.previews))]
            token = uuid.uuid4().hex
            self.previews[token] = dict(expires=time.monotonic()+600, groups=list(groups.values()))
            return dict(token=token, groups=[dict(name=g["name"], person_id=g["person_id"],
                items=[{k: v for k, v in item.items() if k not in ("image", "vector")} for item in g["items"]])
                for g in groups.values()])

    def commit(self, token):
        with self.lock:
            preview = self.previews.pop(token, None)
        if not preview or preview["expires"] < time.monotonic():
            raise ValueError("Import preview expired; review the images again")
        results = []
        for group in preview["groups"]:
            accepted = [i for i in group["items"] if i["accepted"]]
            if not accepted:
                results.append(dict(name=group["name"], accepted=0, rejected=len(group["items"])))
                continue
            person_id = group["person_id"] or self.manager.gallery.create_person(group["name"])
            count = 0
            details = []
            for item in sorted(accepted, key=lambda x: -x["quality"]["score"]):
                try:
                    result = self.manager.gallery.put(person_id, "registered", item["image"], item["vector"],
                        item["quality"], dict(source="explicit_registration", original_filename=item["filename"], consent=True))
                finally:
                    self.manager.reset_evidence()
                count += int(result["accepted"])
                details.append(dict(filename=item["filename"], **result))
            results.append(dict(name=group["name"], person_id=person_id, accepted=count,
                                rejected=len(group["items"])-count, details=details))
        self.manager.reset_evidence()
        return dict(results=results, accepted=sum(r["accepted"] for r in results),
                    rejected=sum(r["rejected"] for r in results))
