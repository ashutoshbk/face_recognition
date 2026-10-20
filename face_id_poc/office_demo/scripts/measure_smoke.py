#!/usr/bin/env python3
"""Capture a short local timing report without exporting identities or photos."""
import argparse
import json
import statistics
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://127.0.0.1:9170")
    parser.add_argument("--seconds", type=int, default=60)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wait-ready", action="store_true", help="Wait up to 45 seconds for all configured cameras")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Choose a new output path")
    rows = []
    waiting = time.monotonic()
    if args.wait_ready:
        while time.monotonic()-waiting < 45:
            with urllib.request.urlopen(args.api+"/api/state", timeout=10) as response:
                state = json.load(response)
            if state["cameras"] and all(c["state"] == "running" and c["metrics"]["processed_frames"] >= 3 for c in state["cameras"]):
                break
            time.sleep(.5)
        else:
            raise RuntimeError("Not all configured cameras became ready")
    warmup = time.monotonic()-waiting
    started = time.monotonic()
    while time.monotonic()-started < args.seconds:
        tick = time.monotonic()
        with urllib.request.urlopen(args.api+"/api/state", timeout=10) as response:
            state = json.load(response)
        row = dict(elapsed=round(time.monotonic()-started, 2), hardware=state["metrics"]["hardware"],
            cameras=[dict(id=c["id"], name=c["name"], state=c["state"], metrics=c["metrics"],
                error=c["error"], tracks=len(c["tracks"]),
                usable_faces=sum(t.get("face_visible", False) for t in c["tracks"]),
                confirmed=sum(t.get("state")=="Confirmed" for t in c["tracks"])) for c in state["cameras"]],
            recognition_ms=state["metrics"]["recognition_ms"], pending=state["metrics"]["pending_recognition"])
        rows.append(row)
        print(f'{row["elapsed"]:5.1f}s '+" | ".join(f'{c["name"]}: {c["metrics"]["fps"]} FPS' for c in row["cameras"]), flush=True)
        time.sleep(max(0, 1-(time.monotonic()-tick)))
    summary = {}
    for camera in rows[-1]["cameras"]:
        observations = [c for r in rows for c in r["cameras"] if c["id"]==camera["id"]]
        summary[camera["id"]] = dict(name=camera["name"],
            mean_fps=statistics.mean(c["metrics"]["fps"] for c in observations),
            minimum_fps=min(c["metrics"]["fps"] for c in observations),
            max_tracks=max(c["tracks"] for c in observations),
            max_usable_faces=max(c["usable_faces"] for c in observations),
            max_confirmed=max(c["confirmed"] for c in observations))
    result = dict(seconds=args.seconds, warmup_seconds=round(warmup,2), timestamp=time.time(), summary=summary, rows=rows,
        scope="Short replay smoke test. Three physical views plus explicitly labelled repeats; not identity accuracy or long-duration capacity.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2)+"\n")
    args.output.chmod(0o600)
    print("REPORT:", args.output)


if __name__ == "__main__":
    main()
