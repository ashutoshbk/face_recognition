import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from synthetic_bench import pipeline, service


def test_fixed_disjoint_split():
    records = [{'identity':i,'archive':'fixture.zip','members':[f'{i}/0.png',f'{i}/2.png']} for i in range(100)]
    gallery, probes = pipeline.plan(records,80,20)
    assert (gallery,probes) == pipeline.plan(records,80,20)
    assert len({r['identity'] for r in gallery}) == 80
    assert len({r['identity'] for r in probes}) == 20
    assert all(r['members']==[f"{r['identity']}/1.png"] for r in probes)
    assert all(not any(m.endswith('/1.png') for m in r['members']) for r in gallery)


def test_resume_storage_and_reject_count(tmp_path):
    db=pipeline.database(tmp_path)
    db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',(1,'gallery','fixture.zip','1/0.png',np.ones(512,np.float32).tobytes(),10,None))
    db.execute('INSERT INTO samples VALUES (?,?,?,?,?,?,?)',(2,'gallery','fixture.zip',None,None,10,'No face'))
    db.commit();db.close()
    reopened=pipeline.database(tmp_path)
    assert reopened.execute('SELECT COUNT(*) FROM samples').fetchone()[0] == 2
    assert reopened.execute('SELECT COUNT(*) FROM samples WHERE vector IS NOT NULL').fetchone()[0] == 1
    reopened.close()


def test_atomic_json_finite_only(tmp_path):
    pipeline.atomic_json(tmp_path/'status.json',{'value':1})
    assert json.loads((tmp_path/'status.json').read_text()) == {'value':1}
    with pytest.raises(ValueError):
        pipeline.atomic_json(tmp_path/'status.json',{'value':float('nan')})
    assert json.loads((tmp_path/'status.json').read_text()) == {'value':1}


def test_summary_statistics():
    assert pipeline.stats([]) is None
    result=pipeline.stats([1,2,3,100])
    assert result['p50']==2.5 and result['maximum']==100 and result['count']==4


def test_service_limits(monkeypatch,tmp_path):
    monkeypatch.setattr(service,'DATA',tmp_path)
    c=TestClient(service.app)
    assert c.get('/api/status').status_code==200
    assert c.get('/api/status',headers={'host':'evil.example'}).status_code==400
    assert c.post('/api/build/start',headers={'origin':'https://other.example'}).status_code==403
    assert c.post('/api/runs/missing/probe/123').status_code==404
    assert c.post('/api/upload',content=b'arbitrary face image').status_code==404
    assert c.post('/api/webcam',content=b'arbitrary frame').status_code==404
    with pytest.raises(Exception):
        service.folder('../test')


def test_only_recorded_synthetic_probe(monkeypatch,tmp_path):
    monkeypatch.setattr(service,'DATA',tmp_path)
    root=tmp_path/'fixture';root.mkdir()
    pipeline.atomic_json(root/'configuration.json',{'scope':'synthetic'})
    pipeline.atomic_json(root/'result.json',{'collection':'unused','hnsw_ef':128})
    db=pipeline.database(root);db.close()
    c=TestClient(service.app)
    assert c.post('/api/runs/fixture/probe/999').status_code==404
    assert c.get('/api/runs/fixture/image/unknown/999').status_code==400
    assert c.get('/api/runs/fixture/image/probe/999').status_code==404


def test_context_closes_client(monkeypatch):
    class Fake:
        closed=False
        def __init__(self,**kwargs): pass
        def close(self): self.closed=True
    monkeypatch.setattr(pipeline,'QdrantClient',Fake)
    with pipeline.client() as client:
        assert not client.closed
    assert client.closed


def test_completed_resume_preserves_sample_counts(monkeypatch,tmp_path):
    monkeypatch.setattr(service,'DATA',tmp_path)
    root=tmp_path/'fixture';root.mkdir()
    pipeline.atomic_json(root/'configuration.json',{})
    pipeline.atomic_json(root/'status.json',{'status':'complete'})
    pipeline.atomic_json(root/'result.json',{'identity_target':256,'probe_target':64,
        'encoded_counts':{'probe':63},'rejected':[{'identity':12,'role':'probe'}]})
    value=service.state('fixture')
    assert value['completed']==320 and value['rejected']==1 and value['probes']==63
