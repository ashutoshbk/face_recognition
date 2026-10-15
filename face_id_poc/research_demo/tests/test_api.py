import importlib
import sys

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv("LOCAL_LAB_DATA",str(tmp_path))
    sys.modules.pop("app",None)
    app = importlib.import_module("app")
    with TestClient(app.app) as client:
        yield client


def test_local_boundary_and_validation(client):
    assert client.get("/api/health").status_code == 200
    assert client.post("/api/sessions",headers={"Origin":"https://other.example"}).status_code == 403
    assert client.get("/api/health",headers={"Host":"other.example"}).status_code == 400
    identifier = client.post("/api/sessions").json()["id"]
    assert client.post(f"/api/sessions/{identifier}/turnaway").status_code == 400
    assert client.get(f"/api/sessions/{identifier}/turnaway/export").status_code == 400
    assert client.put(f"/api/sessions/{identifier}/settings",json={"threshold":-1}).status_code == 422
    assert client.put(f"/api/sessions/{identifier}/settings",json={"selected":"not-registered"}).status_code == 400
    assert client.post(f"/api/sessions/{identifier}/enroll",json={"name":"Tester","consent":True,"track_id":"missing"}).status_code == 400
    assert client.post(f"/api/sessions/{identifier}/frame",content=b"x"*2_000_001).status_code == 413
    assert client.post(f"/api/sessions/{identifier}/frame",content=b"invalid JPEG").status_code == 400
    assert client.post("/api/benchmark",json={"count":42}).status_code == 400
    assert client.post('/api/self-test/enroll-photo',content=b'x'*5_400_001).status_code == 413
    invalid = client.post('/api/self-test/enroll-photo',json={'name':'Tester','consent':True,'image_base64':'private-marker'*500_000})
    assert invalid.status_code == 413
    assert 'private-marker' not in invalid.text
    assert client.post('/api/self-test/enroll-photo',content=b'{broken').status_code == 422
    assert client.post('/api/self-test/enroll-photo',json={'name':'Tester','consent':False,'image_base64':'AAAA'}).status_code == 400
    assert client.get('/api/testers?mode=self').json() == []
    assert client.get("/api/testers").json() == []
    assert client.delete(f"/api/sessions/{identifier}").status_code == 200
    assert client.post(f"/api/sessions/{identifier}/reset").status_code == 404
