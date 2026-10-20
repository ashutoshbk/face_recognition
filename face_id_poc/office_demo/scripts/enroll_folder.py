#!/usr/bin/env python3
"""Preview or explicitly commit an authorized local folder of reference photos."""
import argparse
import json
from pathlib import Path
import urllib.request
from urllib.parse import urlsplit
import uuid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--api", default="http://127.0.0.1:9170")
    parser.add_argument("--flat-names", action="store_true", help="One person per filename, for the old staff folder")
    parser.add_argument("--consent", action="store_true", help="Confirm approved, consenting enrollment material")
    parser.add_argument("--commit", action="store_true", help="Commit eligible references after preview")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    if not args.consent:
        parser.error("Explicit --consent authorization is required")
    origin = urlsplit(args.api)
    if origin.scheme != "http" or origin.hostname not in ("localhost", "127.0.0.1"):
        parser.error("Use the local loopback API or an SSH tunnel")
    root = args.folder.resolve(strict=True)
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and p.is_file())
    if not paths or len(paths) > 200 or sum(p.stat().st_size for p in paths) > 128_000_000:
        parser.error("Expected 1-200 images, at most 128 MB")
    relative = [str(Path("Staff") / p.stem / p.name) if args.flat_names else str(p.relative_to(root)) for p in paths]
    if not args.flat_names and any(len(Path(p).parts) < 2 for p in relative):
        parser.error("Use one subfolder per person, or explicitly select --flat-names")
    boundary = "camex-" + uuid.uuid4().hex
    body = bytearray()
    for name, value in (("consent", "true"), ("paths", json.dumps(relative))):
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode())
    for i, path in enumerate(paths):
        suffix = path.suffix.lower()
        mime = "image/png" if suffix == ".png" else "image/jpeg"
        body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="photo-{i}{suffix}"\r\nContent-Type: {mime}\r\n\r\n'.encode())
        body.extend(path.read_bytes())
        body.extend(b"\r\n")
    body.extend(f"--{boundary}--\r\n".encode())
    request = urllib.request.Request(args.api+"/api/enroll/preview", data=bytes(body),
        headers={"Content-Type": "multipart/form-data; boundary="+boundary})
    with urllib.request.urlopen(request, timeout=180) as response:
        result = {"preview": json.load(response)}
    if args.commit:
        request = urllib.request.Request(args.api+"/api/enroll/commit",
            data=json.dumps({"token": result["preview"]["token"]}).encode(),
            headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=180) as response:
            result["commit"] = json.load(response)
    if args.report:
        if args.report.exists():
            raise ValueError("Choose a new report filename")
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, indent=2)+"\n")
        args.report.chmod(0o600)
    preview = result["preview"]
    print("People:", len(preview["groups"]), "Eligible images:", sum(
        i["accepted"] for g in preview["groups"] for i in g["items"]))
    if "commit" in result:
        print("Committed references:", result["commit"]["accepted"], "Rejected:", result["commit"]["rejected"])
    else:
        print("Preview only; nothing registered. Token:", preview["token"])


if __name__ == "__main__":
    main()
