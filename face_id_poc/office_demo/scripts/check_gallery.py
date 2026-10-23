#!/usr/bin/env python3
"""Read-only exact-vector index sanity check; not a face-recognition accuracy test."""
import argparse
import json
import os
from pathlib import Path
import sqlite3
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from domain import QdrantIndex


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    database = Path(os.environ["CAMEX_OFFICE_HOME"]) / "database/office.sqlite3"
    with sqlite3.connect(f"file:{database}?mode=ro", uri=True) as db:
        rows = db.execute("SELECT id,person_id,vector FROM refs WHERE pool='registered'").fetchall()
    if not rows:
        raise RuntimeError("No registered references to check")
    refs = [dict(id=r[0], person_id=r[1], vector=np.frombuffer(r[2], np.float32)) for r in rows]
    index = QdrantIndex(os.environ.get("CAMEX_OFFICE_QDRANT_URL", "http://127.0.0.1:16343"))
    by_id = {r["id"]: r for r in refs}
    times, correct = [], 0
    for ref in refs:
        started = time.perf_counter()
        scores = index.scores(ref["vector"], refs)
        times.append((time.perf_counter()-started)*1000)
        first = max(scores, key=scores.get)
        correct += by_id[first]["person_id"] == ref["person_id"]
    result = dict(kind="Exact stored-vector self-match sanity; NOT held-out face recognition accuracy",
        references=len(refs), correct_person_self_matches=correct,
        query_median_ms=round(statistics.median(times),3),
        query_p95_ms=round(sorted(times)[min(len(times)-1,int(.95*len(times)))],3))
    if args.output:
        args.output.write_text(json.dumps(result,indent=2)+"\n")
        args.output.chmod(0o600)
    print(json.dumps(result,indent=2))


if __name__ == "__main__":
    main()
