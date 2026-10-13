"""Isolated non-biometric synthetic-vector Qdrant benchmark."""
import json
import platform
import threading
import time
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models
from settings import QDRANT_PORT,QDRANT_GRPC_PORT


def normalize_rows(array):
    return array / np.maximum(np.linalg.norm(array, axis=1, keepdims=True), 1e-8)


def recall_at_k(approximate, reference):
    return len(set(approximate) & set(reference)) / len(reference) if reference else 0.0


class Benchmark:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.cancel_event = threading.Event()
        self.state = {"status": "idle", "progress": 0, "message": "Ready", "result": None}
        previous = self.directory / "latest.json"
        if previous.exists():
            self.state["result"] = json.loads(previous.read_text())

    def client(self):
        return QdrantClient(host="127.0.0.1", port=QDRANT_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True, timeout=120)

    def snapshot(self):
        with self.lock:
            return dict(self.state)

    def update(self, **values):
        with self.lock:
            self.state.update(values)

    def start(self, count=100000, dimensions=128, queries=100, ef=128):
        if count not in (10000, 100000, 500000, 1000000) or dimensions not in (128, 512) or queries not in (20, 100, 300) or ef not in (64, 128, 256):
            raise ValueError("Unsupported benchmark configuration")
        with self.lock:
            if self.state["status"] in ("loading", "indexing", "querying"):
                raise ValueError("A benchmark is already running")
            self.cancel_event.clear()
            self.state = {"status": "loading", "progress": 0, "message": "Connecting to local Qdrant", "result": self.state.get("result"), "count": count, "dimensions": dimensions, "queries": queries, "ef": ef}
            threading.Thread(target=self.run, args=(count, dimensions, queries, ef), daemon=True).start()

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise InterruptedError("Benchmark cancelled; existing vectors retained")

    def run(self, count, dimensions, queries, ef):
        client = self.client()
        collection = f"local_lab_nonbiometric_v1_{count}_{dimensions}"
        started = time.perf_counter()
        try:
            version = client.info().version
            if not client.collection_exists(collection):
                client.create_collection(
                    collection_name=collection,
                    vectors_config=models.VectorParams(size=dimensions, distance=models.Distance.COSINE, on_disk=True),
                    hnsw_config=models.HnswConfigDiff(m=16, ef_construct=100, max_indexing_threads=3, on_disk=True),
                    optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0, default_segment_number=2),
                )
            info = client.get_collection(collection)
            reused = info.points_count == count
            rng = np.random.default_rng(20261013)
            centers = normalize_rows(rng.normal(size=(1024, dimensions)).astype(np.float32))
            if not reused:
                # Deterministic batches can be safely rerun after an interrupted load.
                for start in range(0, count, 2048):
                    self.check_cancel()
                    end = min(start + 2048, count)
                    noise = rng.normal(size=(end-start, dimensions)).astype(np.float32)
                    vectors = normalize_rows(centers[np.arange(start, end) % len(centers)] + noise * (0.3 / np.sqrt(dimensions)))
                    client.upsert(collection, points=models.Batch(ids=list(range(start, end)), vectors=vectors.tolist()), wait=True)
                    self.update(progress=round(65 * end/count, 1), message=f"Loaded {end:,} / {count:,} generated vectors")
            load_seconds = time.perf_counter() - started
            client.update_collection(collection, optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000))
            self.update(status="indexing", progress=66, message="Building HNSW index")
            deadline = time.monotonic() + 900
            while True:
                self.check_cancel()
                info = client.get_collection(collection)
                indexed = info.indexed_vectors_count or 0
                self.update(message=f"Indexed {indexed:,} / {count:,} vectors", progress=66 + 19 * indexed / count)
                if indexed >= count * 0.95 and str(info.status).lower().endswith("green"):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError("Index build has not completed; no finished-search performance claim was produced. Retry later to reuse the collection.")
                time.sleep(2)
            index_seconds = time.perf_counter() - started - load_seconds
            self.update(status="querying", progress=85, message="Measuring held-out queries and exact-search recall")
            query_rng = np.random.default_rng(120261013)
            query_vectors = normalize_rows(centers[query_rng.integers(0, len(centers), size=queries + 10)] + query_rng.normal(size=(queries + 10, dimensions)).astype(np.float32) * (0.3 / np.sqrt(dimensions)))
            params = models.SearchParams(hnsw_ef=ef, exact=False)
            for vector in query_vectors[:10]:
                self.check_cancel()
                client.query_points(collection, query=vector.tolist(), limit=10, search_params=params, with_payload=False, with_vectors=False)
            latencies, recalls, exact_latencies, examples = [], [], [], []
            for index, vector in enumerate(query_vectors[10:]):
                self.check_cancel()
                query = vector.tolist()
                tick = time.perf_counter()
                matches = client.query_points(collection, query=query, limit=10, search_params=params, with_payload=False, with_vectors=False).points
                latencies.append((time.perf_counter() - tick) * 1000)
                tick = time.perf_counter()
                exact = client.query_points(collection, query=query, limit=10, search_params=models.SearchParams(exact=True), with_payload=False, with_vectors=False).points
                exact_latencies.append((time.perf_counter() - tick) * 1000)
                recalls.append(recall_at_k([p.id for p in matches], [p.id for p in exact]))
                if index < 5:
                    examples.append({"query": index + 1, "milliseconds": round(latencies[-1], 3), "recall_at_10": recalls[-1], "top_ids": [p.id for p in matches]})
                self.update(progress=85 + 15 * (index+1)/queries, message=f"Measured {index+1} / {queries} queries")
            result = {
                "timestamp": time.time(), "engine": "Qdrant server", "version": version,
                "transport": "localhost gRPC", "hardware": platform.machine(),
                "data": "Generated clustered numerical vectors; no faces, images, identities or webcam samples",
                "dimensions": dimensions, "vectors": count, "queries": queries, "warmup_queries": 10,
                "distribution": "1024 unit Gaussian centers + normalized Gaussian noise, seed 20261013",
                "query_distribution": "Held-out noisy queries, seed 120261013; not exact copies of inserted points",
                "hnsw_ef": ef, "m": 16, "ef_construct": 100, "top_k": 10,
                "indexed_vectors": indexed, "reused_collection": reused,
                "load_seconds": round(load_seconds, 2), "index_wait_seconds": round(index_seconds, 2),
                "p50_ms": round(float(np.percentile(latencies, 50)), 3),
                "p95_ms": round(float(np.percentile(latencies, 95)), 3),
                "p99_ms": round(float(np.percentile(latencies, 99)), 3),
                "mean_ms": round(float(np.mean(latencies)), 3),
                "serial_query_rate": round(1000 / float(np.mean(latencies)), 1),
                "recall_at_10": round(float(np.mean(recalls)), 4),
                "exact_p50_ms": round(float(np.percentile(exact_latencies, 50)), 3),
                "latencies_ms": latencies, "examples": examples,
                "limitations": "Warm, single-client local search; not maximum concurrent throughput, face-recognition accuracy or a CCTV capacity benchmark.",
            }
            filename = f"benchmark_{count}_{dimensions}_{int(result['timestamp'])}.json"
            (self.directory / filename).write_text(json.dumps(result, indent=2) + "\n")
            (self.directory / "latest.json").write_text(json.dumps(result, indent=2) + "\n")
            self.update(status="complete", progress=100, message="Benchmark complete", result=result)
        except InterruptedError as error:
            self.update(status="cancelled", message=str(error))
        except Exception as error:
            self.update(status="error", message=str(error))
        finally:
            client.close()
