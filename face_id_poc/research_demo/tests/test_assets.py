import hashlib
import zipfile
from types import SimpleNamespace

import pytest

import lab
import vision.networks


def test_cached_download_checks_digest_without_network(tmp_path,monkeypatch):
    target=tmp_path/'archive';target.write_bytes(b'verified')
    monkeypatch.setattr(lab.subprocess,'run',lambda *a,**k:pytest.fail('Unexpected network'))
    lab.download('https://example.invalid/file',target,hashlib.sha256(b'verified').hexdigest())


def test_download_rejects_bad_checksum(tmp_path,monkeypatch):
    target=tmp_path/'archive'
    def fake_download(*a,**k):target.with_suffix('.part').write_bytes(b'corrupt')
    monkeypatch.setattr(lab.subprocess,'run',fake_download)
    with pytest.raises(ValueError,match='Checksum mismatch'):
        lab.download('https://example.invalid/file',target,'0'*64)
    assert not target.exists()


def test_models_require_explicit_research_acknowledgment():
    with pytest.raises(ValueError,match='Review'):
        lab.assets(SimpleNamespace(accept_research_terms=False))


def test_bundle_rejects_wrong_model_without_installing(tmp_path,monkeypatch):
    bundle=tmp_path/'models.zip';models=tmp_path/'models'
    with zipfile.ZipFile(bundle,'w') as archive:archive.writestr('../../model.onnx',b'bad')
    monkeypatch.setattr(lab,'MODELS',models)
    monkeypatch.setattr(vision.networks,'MODEL_HASHES',{'model.onnx':'0'*64})
    with pytest.raises(ValueError,match='checksum'):
        lab.assets(SimpleNamespace(accept_research_terms=True,model_bundle=str(bundle)))
    assert not (models/'model.onnx').exists()
    assert not (tmp_path.parent/'model.onnx').exists()
