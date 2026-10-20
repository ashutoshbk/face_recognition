import importlib
import json
import io
import sys
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domain import Gallery, unit


class Index:
    def replace(self, refs):
        self.refs = list(refs)

    def scores(self, vector, refs):
        return {r["id"]: float(r["vector"] @ vector) for r in refs}


class Vision:
    def detect(self, frame, enrollment=False):
        return [], [np.zeros(15)]

    def describe(self, frame, face, minimum_width):
        return {"accepted": True, "score": 80, "width": 100}, frame, frame

    def encode(self, frame):
        return unit(np.arange(512)+1)

    def status(self):
        return {"ready": True, "providers": {"fake": ["UnitTest"]}}


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("CAMEX_OFFICE_HOME", str(tmp_path))
    module = importlib.import_module("app")
    gallery = Gallery(tmp_path, Index())
    gallery.sync()
    module.gallery = gallery
    module.manager.gallery = gallery
    module.manager.root = tmp_path
    (tmp_path / "config").mkdir(exist_ok=True)
    module.manager.cameras.clear()
    module.manager.vision = Vision()
    module.manager.settings["adaptation_mode"] = "review"
    module.enrollment.manager = module.manager
    yield TestClient(module.app)
    module.manager.cameras.clear()


def image():
    return cv2.imencode(".jpg", np.zeros((120, 120, 3), np.uint8))[1].tobytes()


def test_camera_count_is_not_fixed_to_five(client):
    for i in range(6):
        response = client.post("/api/cameras", json={"name": f"Camera {i}",
            "url": f"rtsp://127.0.0.1:18554/test{i}", "authorized": True})
        assert response.status_code == 200, response.text
    assert len(client.get("/api/state").json()["cameras"]) == 6


def test_enrollment_preview_commit_and_privacy(client):
    result = client.post("/api/enroll/preview", data={"paths": json.dumps(["Staff/Example Person/photo.jpg"]),
        "consent": "true"}, files=[("files", ("photo.jpg", image(), "image/jpeg"))])
    assert result.status_code == 200, result.text
    assert result.json()["groups"][0]["name"] == "Example Person"
    assert "vector" not in result.text and "image" not in result.json()["groups"][0]["items"][0]
    assert client.get("/api/people").json()["people"] == []
    response = client.post("/api/enroll/commit", json={"token": result.json()["token"]})
    assert response.status_code == 200 and response.json()["accepted"] == 1
    person = client.get("/api/people").json()["people"][0]
    assert len(person["registered"]) == 1 and person["live"] == []
    assert client.get(person["registered"][0]["url"]).status_code == 200


def test_enrollment_needs_authorization(client):
    response = client.post("/api/enroll/preview", data={"name": "Tester", "consent": "false"},
                           files=[("files", ("photo.jpg", image(), "image/jpeg"))])
    assert response.status_code == 400

def test_traversal_rejected(client):
    response = client.post("/api/enroll/preview", data={"paths": '["../Tester/photo.jpg"]', "consent": "true"},
                           files=[("files", ("photo.jpg", image(), "image/jpeg"))])
    assert response.status_code == 400


def test_protection_and_auto_lock(client):
    assert client.get("/api/state", headers={"Origin": "https://evil.example"}).status_code == 403
    assert client.post("/api/settings", json={"adaptation_mode": "auto"}).status_code == 400
    assert client.post("/api/settings", json={"admission_threshold": .1}).status_code == 400


def test_camera_url_redacted(client):
    r = client.post("/api/cameras", json={"name": "Private", "url": "rtsp://name:secret@127.0.0.1/private", "authorized": True})
    assert r.status_code == 200 and "secret" not in r.text
    assert "secret" not in client.get("/api/state").text


def test_zip_folder_preview(client):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Staff/Example Person/front.jpg", image())
    response = client.post("/api/enroll/preview", data={"consent": "true"},
        files=[("files", ("staff.zip", archive.getvalue(), "application/zip"))])
    assert response.status_code == 200, response.text
    assert response.json()["groups"][0]["name"] == "Example Person"


def test_zip_traversal_rejected(client):
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("../Example Person/front.jpg", image())
    response = client.post("/api/enroll/preview", data={"consent": "true"},
        files=[("files", ("staff.zip", archive.getvalue(), "application/zip"))])
    assert response.status_code == 400
