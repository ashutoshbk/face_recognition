import json
import fcntl
from pathlib import Path
import numpy as np


def claim_directory(path, fingerprint, protected=()):
    path = Path(path).resolve()
    for p in protected:
        p = Path(p).resolve()
        if p == path or path in p.parents:
            raise ValueError("Output must not contain an input")
    marker = path / ".camex_run.json"
    if path.exists() and any(path.iterdir()):
        if (
            not marker.exists()
            or json.loads(marker.read_text()).get("fingerprint") != fingerprint
        ):
            raise FileExistsError("Output contains unrelated or different-run files")
    else:
        path.mkdir(parents=True, exist_ok=True)
        with marker.open("x") as f:
            json.dump({"fingerprint": fingerprint}, f)
    lock = (path / ".camex_run.lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        raise RuntimeError("Another process is writing this output directory") from None
    return lock


class FileClock:
    """Normalize a CFR file clock; reject discontinuous or stuck positive PTS."""

    def __init__(self, fps):
        self.fps = fps
        self.origin = None
        self.previous = None
        self.fallback_frames = 0

    def timestamp(self, index, raw_ms):
        expected = index * 1000 / self.fps
        if not np.isfinite(raw_ms) or raw_ms < 0:
            raise ValueError("Invalid file PTS; prepare an indexed CFR clip")
        if raw_ms == 0 and (self.previous is None or self.previous == 0):
            self.fallback_frames += 1
            self.previous = 0
            return round(expected)
        if self.origin is None:
            self.origin = raw_ms - expected
        value = raw_ms - self.origin
        if self.previous is not None and raw_ms <= self.previous:
            raise ValueError("Repeated/reversed PTS; prepare an indexed CFR clip")
        if abs(value - expected) > 1000 / self.fps * 0.6:
            raise ValueError(
                "Non-CFR or discontinuous PTS; prepare an indexed CFR clip"
            )
        self.previous = raw_ms
        return round(value)
