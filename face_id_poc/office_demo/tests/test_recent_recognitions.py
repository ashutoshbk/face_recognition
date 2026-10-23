import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js required for UI history tests")
def test_recent_recognition_history():
    subprocess.run(
        [shutil.which("node"), "--test", str(Path(__file__).with_name("recent_recognitions.test.cjs")),
         str(Path(__file__).with_name("recognition_snapshots.test.cjs")),
         str(Path(__file__).with_name("settings_defaults.test.cjs"))],
        check=True, capture_output=True, text=True, timeout=30,
    )
