import json
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

import cv2
import numpy as np
import pytest

from camex_floor.adapter import from_mct_result
from camex_floor.runtime import FloorTrackingService
from test_runtime import site, packet


def test_mct_schema_adapter_preserves_frame_time():
    result = {
        "site_id": "test",
        "camera_id": "north",
        "timestamp_ms": 1234,
        "tracks": [
            {
                "local_id": 9,
                "global_id": 6,
                "bbox_xyxy": [1, 2, 3, 4],
                "detection_confidence": 0.7,
            }
        ],
    }
    p = from_mct_result(
        result,
        np.zeros((480, 640, 3), np.uint8),
        lambda frame, tracks: [{"embedding": [1, 0]}],
    )
    assert (
        p["timestamp_ms"] == 1234
        and p["frame_size_wh"] == [640, 480]
        and p["tracks"][0]["embedding"] == [1, 0]
    )


def test_two_same_camera_tracks_cannot_share_global_id():
    svc = FloorTrackingService(site())
    p = packet("north", 0, "same")
    p["tracks"].append(
        {**p["tracks"][0], "local_id": 2, "bbox_xyxy": [300, 100, 350, 300]}
    )
    result = svc.process_batch([p])
    assert len({t["global_id"] for t in result["tracks"]}) == 2


def test_adapter_preserves_live_identity_corrections():
    event = {"source_global_id": "fragment", "canonical_global_id": "stable"}
    result = {"camera_id": "new_camera", "timestamp_ms": 100, "tracks": []}
    packet = from_mct_result(
        result, np.zeros((40, 60, 3), np.uint8), identity_updates=[event]
    )
    assert packet["upstream_identity_updates"] == [event]


def test_conflicting_upstream_alias_is_reported_without_stopping_other_cameras():
    svc = FloorTrackingService(site())
    p = packet("north", 0, "a")
    p["tracks"].append({**p["tracks"][0], "local_id": 2, "global_id": "b"})
    svc.process_batch([p])
    p = packet("south", 100, "c")
    p["upstream_identity_updates"] = [
        {
            "site_id": "test",
            "timestamp_ms": 100,
            "source_global_id": "a",
            "canonical_global_id": "b",
        }
    ]
    r = svc.process_batch([p])
    assert r["metrics"]["rejected_identity_updates"] == 1
    assert r["identity_updates"][0]["event"] == "identity_update_rejected"
    assert svc.resolve_global_id("a") != svc.resolve_global_id("b")


def test_examples_parse_without_optional_models():
    import ast
    from pathlib import Path

    examples = list((Path(__file__).resolve().parents[1] / "examples").glob("*.py"))
    assert examples, "Integration example must be included in the distribution"
    for p in examples:
        ast.parse(p.read_text(), filename=str(p))


def test_http_assets_revision_and_fail_closed_approval(tmp_path):
    s = site()
    s["revision"] = 0
    for cid, c in s["cameras"].items():
        c["reference"] = cid + ".png"
        cv2.imwrite(str(tmp_path / c["reference"]), np.zeros((480, 640, 3), np.uint8))
    cv2.imwrite(str(tmp_path / "floor.png"), np.full((480, 640, 3), 255, np.uint8))
    path = tmp_path / "site.json"
    path.write_text(json.dumps(s))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "camex_floor.cli",
            "editor",
            "--site",
            str(path),
            "--port",
            str(port),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(80):
            try:
                with urllib.request.urlopen(base + "/api/site") as r:
                    assert json.load(r)["site_id"] == "test"
                break
            except urllib.error.URLError:
                time.sleep(0.1)
        else:
            raise AssertionError("Editor failed to start")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/../../site.json")
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(base + "/asset?kind=/etc/passwd")
        body = {"revision": 0, "camera_id": "north", "anchors": [], "action": "approve"}
        req = urllib.request.Request(
            base + "/api/save",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 400 and json.loads(path.read_text())["revision"] == 0
        body["action"] = "save"
        req = urllib.request.Request(
            base + "/api/save",
            json.dumps(body).encode(),
            {"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as r:
            assert json.load(r)["revision"] == 1
        with pytest.raises(urllib.error.HTTPError) as e:
            urllib.request.urlopen(req)
        assert e.value.code == 409
        req = urllib.request.Request(
            base + "/api/save",
            json.dumps({**body, "revision": 1}).encode(),
            {"Content-Type": "application/json", "Origin": "https://example.com"},
        )
        with pytest.raises(urllib.error.HTTPError):
            urllib.request.urlopen(req)
    finally:
        proc.terminate()
        proc.wait(timeout=10)
