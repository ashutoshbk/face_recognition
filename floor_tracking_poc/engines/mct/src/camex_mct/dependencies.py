from __future__ import annotations

import os
import sys
from pathlib import Path


def repository_root() -> Path:
    configured = os.environ.get("CAMEX_MCT_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def configure_third_party() -> Path:
    root = (
        Path(os.environ.get("CAMEX_MCT_THIRD_PARTY", repository_root() / "third_party"))
        .expanduser()
        .resolve()
    )
    required = (root / "ByteTrack", root / "BoT-SORT", root / "deep-person-reid")
    missing = [str(path) for path in required if not path.is_dir()]
    if missing:
        raise RuntimeError(
            "Missing pinned third-party dependencies: " + ", ".join(missing) + ". Run `make setup`."
        )
    for path in reversed(required):
        value = str(path)
        if value not in sys.path:
            sys.path.insert(0, value)
    return root
