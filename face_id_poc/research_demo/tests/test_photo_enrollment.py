import base64
import io

import numpy as np
import pytest
from PIL import Image

from core import jpeg, unit
from photo_enrollment import PhotoEnrollment, decode_photo
from test_core import FakeEngine, FRAME, face
from test_self_test import setup


class PhotoEncoder:
    def encode_photo(self,crop):
        assert crop.shape==(112,112,3)
        return unit(np.ones(512))


def service(tmp_path):
    gallery,search,q=setup(tmp_path);search.status=lambda:{'ready':True}
    engine=FakeEngine()
    return PhotoEnrollment(engine,PhotoEncoder(),search),gallery,search,q,engine


def encoded(frame=FRAME):return base64.b64encode(jpeg(frame)).decode()


def test_explicit_photo_registration_and_lookup(tmp_path):
    operator,gallery,search,q,engine=service(tmp_path)
    result=operator.enroll('Consenting tester',True,encoded())
    assert result['testers'][0]['samples']==1
    assert gallery.get(result['id'])['template'].shape==(512,)
    assert search.search(unit(np.ones(512)),result['id'])['self_is_first']
    assert all(x=='self-test' for x in q.writes)
    with pytest.raises(ValueError,match='one self profile'):operator.enroll('Second',True,encoded())


@pytest.mark.parametrize('case',['no_consent','no_face','multiple_faces','multiple_heads','blur','invalid_image'])
def test_photo_rejection_never_stores_a_profile(tmp_path,case):
    operator,gallery,search,q,engine=service(tmp_path);data=encoded();consent=True
    if case=='no_consent':consent=False
    if case=='no_face':engine.faces=[]
    if case=='multiple_faces':engine.faces=[face(),face()]
    if case=='multiple_heads':engine.heads=np.array([[0,0,100,100,.9],[200,0,300,100,.9]])
    if case=='blur':data=encoded(np.zeros_like(FRAME))
    if case=='invalid_image':data=base64.b64encode(b'not an image').decode()
    with pytest.raises(ValueError):operator.enroll('Tester',consent,data)
    assert not gallery.list() and not q.writes


def test_unavailable_qdrant_rolls_back_photo_registration(tmp_path):
    operator,gallery,search,q,_=service(tmp_path)
    def fail(*a,**k):raise RuntimeError('Qdrant unavailable')
    q.upsert=fail
    with pytest.raises(RuntimeError):operator.enroll('Tester',True,encoded())
    assert not gallery.list()


def test_png_and_invalid_photo_sizes():
    buffer=io.BytesIO();Image.new('RGB',(100,100),(100,120,130)).save(buffer,format='PNG')
    assert decode_photo(base64.b64encode(buffer.getvalue()).decode()).shape==(100,100,3)
    with pytest.raises(ValueError):decode_photo('%%%')
    with pytest.raises(ValueError,match='4 MB'):decode_photo(base64.b64encode(b'x'*4_000_001).decode())
    large=io.BytesIO();Image.new('RGB',(4097,80)).save(large,format='PNG')
    with pytest.raises(ValueError,match='dimensions'):decode_photo(base64.b64encode(large.getvalue()).decode())


def test_photo_registration_requires_completed_gallery(tmp_path):
    operator,gallery,search,q,_=service(tmp_path);search.status=lambda:{'ready':False}
    with pytest.raises(ValueError,match='gallery setup'):operator.enroll('Tester',True,encoded())
    assert not gallery.list()
