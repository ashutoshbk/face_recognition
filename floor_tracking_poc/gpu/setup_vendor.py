"""Fetch the pinned tracker dependencies during the image build."""

from pathlib import Path
import subprocess

root = Path("/opt/camex")
target = root / "third_party"
target.mkdir(exist_ok=True)
for line in (root / "engines/mct/third_party.lock").read_text().splitlines():
    if not line.strip():
        continue
    name, url, commit = line.split()
    path = target / name
    if not path.exists():
        subprocess.run(["git", "init", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth", "1", url, commit], check=True
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", "FETCH_HEAD"], check=True
        )
    actual = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != commit:
        raise RuntimeError(f"Wrong dependency revision: {name}")
patch = root / "engines/mct/patches/bot-sort-external-reid.patch"
subprocess.run(["git", "-C", str(target / "BoT-SORT"), "apply", str(patch)], check=True)
