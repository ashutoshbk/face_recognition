#!/usr/bin/env python3
"""Opt-in camera-only import for a ready, local office backend. Never enrolls people."""
import argparse
import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import ProxyHandler, Request, build_opener


def camera_rows(path):
    if path.stat().st_mode & 0o077:
        raise ValueError("Camera config is not private; chmod 600 it before importing")
    value = json.loads(path.read_text())
    rows = value.get("cameras", [])
    expected = {"camera_a": None, "camera_b": None, "camera_c": None,
                "camera_d_replay_a": "camera_a", "camera_e_replay_b": "camera_b"}
    if len(rows) != 5 or {row.get("id") for row in rows} != set(expected):
        raise ValueError("Expected the five-camera fixture manifest")
    for row in rows:
        url = urlsplit(row.get("url", ""))
        if (row.get("fixture") is not True or row.get("replay_of") != expected[row["id"]]
                or url.scheme != "rtsp" or url.hostname not in {"127.0.0.1", "mediamtx"}
                or url.username or url.password or url.query or url.fragment
                or url.path != "/" + row["id"] or not row.get("name")):
            raise ValueError("Invalid fixture metadata or non-local fixture URL")
        if row["replay_of"] and "REPLAY" not in row["name"]:
            raise ValueError("Duplicate streams must retain an explicit REPLAY label")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="Private cameras.docker.json or cameras.host.json")
    parser.add_argument("--api", default="http://127.0.0.1:9170")
    parser.add_argument("--authorized", action="store_true",
                        help="Assert authorization for these recordings and this evaluation")
    parser.add_argument("--start", action="store_true", help="Start only the five newly imported cameras")
    args = parser.parse_args()
    os.umask(0o077)
    if not args.authorized:
        parser.error("Import requires --authorized after recording/evaluation authorization is confirmed")
    api = urlsplit(args.api)
    if (api.scheme != "http" or api.hostname not in {"127.0.0.1", "localhost"}
            or api.username or api.password or api.query or api.fragment or api.path not in {"", "/"}):
        parser.error("API must be an HTTP loopback origin without credentials or a path")
    rows = camera_rows(args.config)
    opener = build_opener(ProxyHandler({}))

    def request(path, data=None):
        body = json.dumps(data).encode() if data is not None else None
        message = Request(args.api.rstrip("/") + path, data=body,
                          headers={"Content-Type": "application/json"})
        try:
            with opener.open(message, timeout=15) as response:
                return json.load(response)
        except (HTTPError, URLError) as error:
            raise RuntimeError(f"Office API request failed for {path}; inspect local backend logs") from error

    if request("/api/health").get("ready") is not True:
        raise RuntimeError("Office backend models/index are not ready; no cameras imported")
    existing = {camera["name"] for camera in request("/api/state").get("cameras", [])}
    if existing.intersection(row["name"] for row in rows):
        raise RuntimeError("Fixture names already exist; inspect the UI before retrying a partial import")
    imported = []
    try:
        for row in rows:
            result = request("/api/cameras", {"name": row["name"], "url": row["url"], "authorized": True})
            imported.append(result["id"])
        if args.start:
            for identifier in imported:
                request(f"/api/cameras/{quote(identifier, safe='')}/start", {})
    except Exception:
        print(f"Import interrupted after adding {len(imported)} cameras. Inspect the UI; no automatic deletion.")
        raise
    print(f"Imported {len(imported)} fixture cameras (3 sources + 2 labelled replays); "
          + ("started only these cameras." if args.start else "left stopped for review."))


if __name__ == "__main__":
    main()
