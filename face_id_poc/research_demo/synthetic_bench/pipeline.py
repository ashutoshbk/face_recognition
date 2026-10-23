"""Reproducible, resumable synthetic-image embedding and Qdrant evaluation."""

import argparse
import concurrent.futures
from contextlib import contextmanager
import fcntl
import hashlib
import json
import multiprocessing
import os
import platform
import signal
import sqlite3
import time
import zipfile
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from qdrant_client import QdrantClient, models

from vision.networks import MODEL_HASHES, sha256
from vision.vendor.face_align import norm_crop
from vision.vendor.scrfd import SCRFD
from settings import ROOT,DATA as APP_DATA,DATASET,MODELS,QDRANT_PORT,QDRANT_GRPC_PORT

DATA = APP_DATA / 'digiface_research'
VERSION = 'digiface_scrfd640_arcface_r50_v1'
ENCODER = None
ARCHIVES = {}
STOP = False


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def database(directory):
    directory.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(directory / 'embeddings.sqlite3', timeout=60)
    connection.execute('PRAGMA journal_mode=WAL')
    connection.execute('''CREATE TABLE IF NOT EXISTS samples (
        identity INTEGER NOT NULL, role TEXT NOT NULL, archive TEXT NOT NULL,
        member TEXT, vector BLOB, milliseconds REAL, reason TEXT,
        PRIMARY KEY(identity, role))''')
    return connection


def catalog(dataset=DATASET):
    manifest = json.loads((dataset / 'download_manifest.json').read_text())
    if manifest.get('status') != 'complete':
        raise ValueError('Verified DigiFace download manifest required')
    records = []
    seen = set()
    for archive in manifest['archives']:
        path = dataset / archive['file']
        if path.stat().st_size != archive['bytes']:
            raise ValueError(f'Archive size differs from verified manifest: {path.name}')
        with zipfile.ZipFile(path) as z:
            members = set(z.namelist())
        ids = sorted(int(p.split('/')[0]) for p in members if p.endswith('/0.png') and p.split('/')[0].isdigit())
        if len(ids) != archive['identity_folder_count']:
            raise ValueError(f'Identity count mismatch in {path.name}')
        for identity in ids:
            if identity in seen:
                raise ValueError('Duplicate synthetic identity across archives')
            seen.add(identity)
            if f'{identity}/1.png' not in members:
                raise ValueError('Missing disjoint probe image')
            candidates = [f'{identity}/{i}.png' for i in (0, 2, 3) if f'{identity}/{i}.png' in members]
            records.append({'identity': identity, 'archive': path.name, 'members': candidates})
    records.sort(key=lambda r: r['identity'])
    return records, manifest


def plan(records, limit, probes, seed=20261013):
    rng = np.random.default_rng(seed)
    selected = sorted(rng.choice(len(records), min(limit, len(records)), replace=False).tolist())
    enrollment = [records[i] for i in selected]
    probe_indices = sorted(rng.choice(len(enrollment), min(probes, len(enrollment)), replace=False).tolist())
    queries = [{'identity': enrollment[i]['identity'], 'archive': enrollment[i]['archive'],
                'members': [f"{enrollment[i]['identity']}/1.png"]} for i in probe_indices]
    return enrollment, queries


class Encoder:
    def __init__(self, threads=2):
        cv2.setNumThreads(1)
        def create(name):
            path = MODELS / name
            if sha256(path) != MODEL_HASHES[name]:
                raise ValueError(f'Model checksum mismatch: {name}')
            options = ort.SessionOptions()
            options.intra_op_num_threads = threads
            options.inter_op_num_threads = 1
            options.log_severity_level = 3
            return ort.InferenceSession(str(path), sess_options=options, providers=['CPUExecutionProvider'])
        self.detector = SCRFD(session=create('scrfd-kps-2.5g-640x640.onnx'))
        self.detector.det_thresh = 0.5
        self.detector.nms_thresh = 0.4
        self.rec = create('arcface-r50-w600k-112x112.onnx')
        self.input_name = self.rec.get_inputs()[0].name

    def encode(self, image):
        if image is None or image.shape != (112, 112, 3):
            raise ValueError('Expected a DigiFace 112x112 color image')
        # Padding prevents a tightly cropped face from being truncated during detection.
        frame = cv2.copyMakeBorder(cv2.resize(image, (224, 224)), 48, 48, 48, 48, cv2.BORDER_CONSTANT, value=(127,127,127))
        boxes, landmarks = self.detector.detect(frame, input_size=(640,640))
        if landmarks is None or len(boxes) != 1:
            raise ValueError('No unambiguous face detected')
        if not np.isfinite(landmarks).all():
            raise ValueError('Invalid face landmarks')
        crop = norm_crop(frame, landmarks[0].astype(np.float32), image_size=112)
        blob = cv2.dnn.blobFromImage(crop, 1/127.5, (112,112), (127.5,127.5,127.5), swapRB=True)
        vector = self.rec.run(None, {self.input_name:blob})[0].reshape(-1).astype(np.float32)
        norm = np.linalg.norm(vector)
        if vector.size != 512 or not np.isfinite(vector).all() or norm < 1e-8:
            raise ValueError('Invalid descriptor')
        return vector / norm


def initialize_worker(threads):
    global ENCODER
    ENCODER = Encoder(threads)
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def read_member(archive, member):
    if archive not in ARCHIVES:
        ARCHIVES[archive] = zipfile.ZipFile(DATASET / archive)
    return cv2.imdecode(np.frombuffer(ARCHIVES[archive].read(member), np.uint8), cv2.IMREAD_COLOR)


def embed_record(record):
    started = time.perf_counter()
    errors = []
    for member in record['members']:
        try:
            vector = ENCODER.encode(read_member(record['archive'], member))
            return (record['identity'], record['role'], record['archive'], member,
                    vector.tobytes(), (time.perf_counter()-started)*1000, None)
        except ValueError as error:
            errors.append(f'{member}: {error}')
    return (record['identity'], record['role'], record['archive'], None, None,
            (time.perf_counter()-started)*1000, '; '.join(errors))


def embeddings(connection, directory, enrollment, queries, workers, threads, update):
    global STOP
    completed = set(connection.execute('SELECT identity,role FROM samples').fetchall())
    jobs = [{**r, 'role':role} for role, rows in [('gallery',enrollment),('probe',queries)] for r in rows]
    pending = [r for r in jobs if (r['identity'],r['role']) not in completed]
    done = len(jobs) - len(pending)
    started = time.monotonic()
    baseline = done
    # Bounded batches avoid queuing 110,000 futures or large images in memory.
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
            initializer=initialize_worker, initargs=(threads,)) as pool:
        for offset in range(0, len(pending), 128):
            if STOP:
                raise InterruptedError('Paused after saving completed batches')
            results = pool.map(embed_record, pending[offset:offset+128], chunksize=4)
            for result in results:
                connection.execute('INSERT OR REPLACE INTO samples VALUES (?,?,?,?,?,?,?)', result)
                done += 1
            connection.commit()
            elapsed = time.monotonic()-started
            rate = (done-baseline)/max(elapsed,1e-6)
            counts = dict(connection.execute('SELECT role, COUNT(*) FROM samples WHERE vector IS NOT NULL GROUP BY role'))
            rejected = connection.execute('SELECT COUNT(*) FROM samples WHERE vector IS NULL').fetchone()[0]
            update(status='embedding', completed=done, total=len(jobs), gallery_vectors=counts.get('gallery',0),
                   probes=counts.get('probe',0), rejected=rejected, images_per_second=rate,
                   eta_seconds=(len(jobs)-done)/max(rate,1e-6))
    return time.monotonic()-started, len(pending)


@contextmanager
def client():
    connection = QdrantClient(host='127.0.0.1', port=QDRANT_PORT, grpc_port=QDRANT_GRPC_PORT, prefer_grpc=True, timeout=120)
    try:
        yield connection
    finally:
        connection.close()


def populate(connection, collection, update):
    started = time.monotonic()
    with client() as q:
        if not q.collection_exists(collection):
            q.create_collection(collection, vectors_config=models.VectorParams(size=512,distance=models.Distance.COSINE,on_disk=True),
                hnsw_config=models.HnswConfigDiff(m=16,ef_construct=100,max_indexing_threads=3,on_disk=True),
                optimizers_config=models.OptimizersConfigDiff(indexing_threshold=0,default_segment_number=2))
        total = connection.execute("SELECT COUNT(*) FROM samples WHERE role='gallery' AND vector IS NOT NULL").fetchone()[0]
        loaded = 0
        cursor = connection.execute("SELECT identity,archive,member,vector FROM samples WHERE role='gallery' AND vector IS NOT NULL ORDER BY identity")
        while rows := cursor.fetchmany(512):
            if STOP:
                raise InterruptedError('Stopped during idempotent Qdrant upload')
            q.upsert(collection, points=models.Batch(ids=[r[0] for r in rows],
                vectors=[np.frombuffer(r[3],np.float32).tolist() for r in rows],
                payloads=[{'source':'DigiFace1M synthetic','synthetic_identity':r[0], 'archive':r[1],'member':r[2]} for r in rows]),wait=True)
            loaded += len(rows)
            update(status='loading', loaded=loaded, gallery_vectors=total, collection=collection)
        if q.count(collection,exact=True).count != total:
            raise RuntimeError('Collection count mismatch; no benchmark emitted')
        loaded_at = time.monotonic()
        q.update_collection(collection,optimizers_config=models.OptimizersConfigDiff(indexing_threshold=1000))
        deadline = time.monotonic()+1800
        while True:
            if STOP:
                raise InterruptedError('Stopped during indexing; rerun to resume')
            info = q.get_collection(collection)
            indexed = info.indexed_vectors_count or 0
            update(status='indexing', indexed_vectors=indexed, gallery_vectors=total)
            if str(info.status).lower().endswith('green') and (indexed >= total*0.95 or total < 1000):
                break
            if time.monotonic() > deadline:
                raise RuntimeError('HNSW build timeout; repeat the run to resume')
            time.sleep(2)
        return {'upload_seconds':loaded_at-started,'index_wait_seconds':time.monotonic()-loaded_at,
                'gallery_vectors':total,'indexed_vectors':indexed,'qdrant_version':q.info().version}


def stats(values):
    if not values:
        return None
    return {key:float(np.percentile(values,p)) for key,p in [('p50',50),('p95',95),('p99',99)]} | {
        'mean':float(np.mean(values)), 'maximum':float(max(values)), 'count':len(values)}


def evaluate(connection, collection, ef, update, max_queries=300):
    rows = connection.execute("""SELECT p.identity,p.archive,p.member,p.vector,p.milliseconds FROM samples p
        JOIN samples g ON p.identity=g.identity AND g.role='gallery' AND g.vector IS NOT NULL
        WHERE p.role='probe' AND p.vector IS NOT NULL ORDER BY p.identity""").fetchall()
    if not rows:
        raise RuntimeError('No successfully encoded disjoint probes with enrolled identities')
    rng = np.random.default_rng(20261014)
    measured = [rows[i] for i in sorted(rng.choice(len(rows),min(max_queries,len(rows)),replace=False).tolist())]
    params = models.SearchParams(hnsw_ef=ef,exact=False)
    latency, exact_latency, recall, correct, examples = [], [], [], [], []
    with client() as q:
        for row in rows[:min(10,len(rows))]:
            q.query_points(collection,query=np.frombuffer(row[3],np.float32).tolist(),limit=10,search_params=params,with_payload=False)
        for index,row in enumerate(measured):
            if STOP:
                raise InterruptedError('Stopped during search evaluation')
            vector = np.frombuffer(row[3],np.float32).tolist()
            start = time.perf_counter()
            matches = q.query_points(collection,query=vector,limit=10,search_params=params,with_payload=False,with_vectors=False).points
            latency.append((time.perf_counter()-start)*1000)
            start = time.perf_counter()
            exact = q.query_points(collection,query=vector,limit=10,search_params=models.SearchParams(exact=True),with_payload=False,with_vectors=False).points
            exact_latency.append((time.perf_counter()-start)*1000)
            reference = {p.id for p in exact}
            recall.append(len(reference & {p.id for p in matches})/len(reference))
            correct.append(bool(matches and matches[0].id == row[0]))
            examples.append({'identity':row[0],'query_archive':row[1],'query_member':row[2],
                'correct_top1':correct[-1],'search_ms':latency[-1],'ann_recall_at10':recall[-1],
                'matches':[{'identity':p.id,'cosine':p.score} for p in matches]})
            if index % 10 == 0:
                update(status='querying',queries_done=index+1,query_count=len(measured))
    encoder = Encoder(threads=2)
    full_latency = []
    with client() as q:
        for row in measured[:25]:
            if STOP:
                raise InterruptedError('Stopped during end-to-end measurement')
            start = time.perf_counter()
            vector = encoder.encode(read_member(row[1],row[2]))
            q.query_points(collection,query=vector.tolist(),limit=10,search_params=params,with_payload=False,with_vectors=False)
            full_latency.append((time.perf_counter()-start)*1000)
    return {'queries':len(measured),'eligible_probes':len(rows),'warmup_queries':min(10,len(rows)),'top_k':10,'hnsw_ef':ef,
        'search_latency_ms':stats(latency),'exact_latency_ms':stats(exact_latency),
        'ann_recall_at10':float(np.mean(recall)), 'synthetic_closed_set_top1':float(np.mean(correct)),
        'latencies_ms':latency,'examples':examples,'image_to_results_latency_ms':stats(full_latency),
        'image_to_results_definition':'Sequential ZIP member read, PNG decode, SCRFD, alignment, ArcFace and local gRPC lookup; model startup excluded; at most 25 fixed probes.',
        'query_definition':'Image 1.png only; never used for gallery enrollment; encoded probe and gallery identities must both succeed.',
        'timing_definition':'Warm sequential localhost gRPC round trip; excludes face encoding, disk image read, UI and exact reference query.',
        'limitations':'Synthetic closed-set experiment, not real-person recognition accuracy, open-set FAR/TAR, maximum throughput or live surveillance capacity.'}


def run(args):
    global STOP
    DATA.mkdir(parents=True,exist_ok=True)
    directory = DATA / args.name
    directory.mkdir(exist_ok=True)
    with (directory/'run.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        status = {'status':'preparing','started_at':time.time(),'pid':os.getpid(),'name':args.name,'collection':None}
        def update(**values):
            status.update(values,updated_at=time.time())
            atomic_json(directory/'status.json',status)
        def stop(*_):
            global STOP
            STOP=True
        signal.signal(signal.SIGTERM,stop)
        signal.signal(signal.SIGINT,stop)
        connection = database(directory)
        try:
            records, manifest = catalog()
            enrollment, queries = plan(records,args.identities,args.probes)
            configuration = {'version':VERSION,'dataset_identities':len(records),'dataset_images':manifest['total_png_count'],
                'identity_target':len(enrollment),'probe_target':len(queries),'seed':20261013,'dimensions':512,
                'gallery_candidates':[0,2,3], 'probe_image':1,'model_hashes':{k:v for k,v in MODEL_HASHES.items() if not k.startswith('yolox')},
                'archive_sha256':{a['file']:a['sha256'] for a in manifest['archives']},
                'scope':'Standalone synthetic-only non-commercial research. No webcam, RTSP, real identities or deployment connectors.'}
            configpath = directory/'configuration.json'
            if configpath.exists() and json.loads(configpath.read_text()) != configuration:
                raise ValueError('Configuration changed; choose a new run name to avoid mixing data')
            atomic_json(configpath,configuration)
            collection = f"digiface_research_{hashlib.sha256(json.dumps(configuration,sort_keys=True).encode()).hexdigest()[:16]}"
            update(total=len(enrollment)+len(queries),identity_target=len(enrollment),probe_target=len(queries),collection=collection,
                workers=args.workers,threads_per_worker=args.threads)
            elapsed, processed = embeddings(connection,directory,enrollment,queries,args.workers,args.threads,update)
            sizes = populate(connection,collection,update)
            result = evaluate(connection,collection,args.ef,update,args.queries)
            counts = dict(connection.execute('SELECT role,COUNT(*) FROM samples WHERE vector IS NOT NULL GROUP BY role'))
            failures = [{'identity':r[0],'role':r[1],'reason':r[2]} for r in connection.execute('SELECT identity,role,reason FROM samples WHERE vector IS NULL')]
            encoding = [r[0] for r in connection.execute("SELECT milliseconds FROM samples WHERE vector IS NOT NULL")]
            result.update(configuration)
            result.update(sizes,collection=collection,completed_at=time.time(),
                hardware={'machine':platform.machine(),'system':platform.platform(),'provider':'CPUExecutionProvider'},
                encoded_counts=counts, rejected=failures, embedding_ms=stats(encoding),
                this_run_encoding_seconds=elapsed, this_run_new_samples=processed,
                note='Embedding timings include parallel CPU contention. Resume reuses cached descriptors; upload is idempotent.')
            atomic_json(directory/'result.json',result)
            atomic_json(DATA/'latest_completed.json',{'name':args.name,'completed_at':result['completed_at']})
            update(status='complete',result_available=True,eta_seconds=0,gallery_vectors=sizes['gallery_vectors'])
            print(json.dumps({k:result[k] for k in ['collection','gallery_vectors','queries','search_latency_ms','ann_recall_at10','synthetic_closed_set_top1']},indent=2),flush=True)
        except BaseException as error:
            update(status='paused' if isinstance(error,InterruptedError) else 'error',error=str(error))
            raise
        finally:
            connection.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',default='all_identities')
    parser.add_argument('--identities',type=int,default=109999)
    parser.add_argument('--probes',type=int,default=400)
    parser.add_argument('--queries',type=int,default=300)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--ef',type=int,default=256)
    args = parser.parse_args()
    if not args.name.replace('_','').isalnum() or not 1 <= args.identities <= 109999 or not 1 <= args.probes <= 2000 or not 1 <= args.queries <= 2000 or not 1 <= args.workers <= 4 or not 1 <= args.threads <= 4 or args.ef not in (64,128,256,512):
        parser.error('Invalid benchmark configuration')
    run(args)


if __name__ == '__main__':
    main()
