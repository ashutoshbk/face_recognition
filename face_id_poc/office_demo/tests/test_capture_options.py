import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import runtime


def test_old_ffmpeg_socket_timeout(monkeypatch):
    runtime.rtsp_timeout_option.cache_clear()
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=" -stimeout <int> socket timeout"))
    assert runtime.rtsp_timeout_option() == "-stimeout"
    runtime.rtsp_timeout_option.cache_clear()


def test_new_ffmpeg_socket_timeout(monkeypatch):
    runtime.rtsp_timeout_option.cache_clear()
    monkeypatch.setattr(runtime.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=" -timeout <int64> socket timeout"))
    assert runtime.rtsp_timeout_option() == "-timeout"
    runtime.rtsp_timeout_option.cache_clear()
