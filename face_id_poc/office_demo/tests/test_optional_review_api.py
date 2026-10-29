import importlib
import json

from test_api import client, image


def enroll(client):
    preview = client.post('/api/enroll/preview', data={'name':'Consented participant','consent':'true'},
                          files=[('files', ('face.jpg', image(), 'image/jpeg'))]).json()
    client.post('/api/enroll/commit', json={'token':preview['token']})
    return client.get('/api/people').json()['people'][0]


def test_optional_controls_and_complete_reset(client):
    response=client.post('/api/settings',json={'unknown_capture_enabled':True})
    assert response.status_code == 400
    response=client.post('/api/settings',json={'review_consent':True,'unknown_capture_enabled':True,'flagged_review_enabled':True})
    assert response.status_code == 200 and response.json()['unknown_capture_enabled']
    defaults=client.get('/api/settings/defaults').json()['settings']
    response=client.post('/api/settings',json=defaults)
    assert response.status_code == 200
    assert not response.json()['unknown_capture_enabled'] and not response.json()['flagged_review_enabled']


def test_flag_management_requires_enrollment_and_explicit_consent(client):
    person=enroll(client)
    endpoint='/api/people/'+person['id']+'/flag'
    assert client.post(endpoint,json={'reason':'Test'}).status_code == 400
    response=client.post(endpoint,json={'reason':'Consented test','participant_consented':True,'expires_hours':24})
    assert response.status_code == 200
    state=client.get('/api/state').json()
    assert state['review']['flags'][0]['person_id'] == person['id']
    assert not state['settings']['flagged_review_enabled']
    assert client.delete(endpoint).status_code == 200
    assert client.get('/api/state').json()['review']['flags'] == []


def test_unknown_photos_stay_out_of_gallery_and_use_normal_enrollment(client):
    module=importlib.import_module('app')
    store=module.manager.review.store
    item=dict(source_key='synthetic-track',camera_id='c',camera_name='Fixture',track_id='t',image=image(),
              quality={'accepted':True,'score':80,'width':100})
    store.save_unknown(item,module.manager.settings)
    result=client.get('/api/review/unknowns').json()
    assert result['total'] == 1 and client.get('/api/people').json()['people'] == []
    photo=result['items'][0]['photos'][0]
    image_response=client.get(photo['url'])
    assert image_response.status_code == 200 and image_response.headers['content-type'] == 'image/jpeg'
    preview=client.post('/api/enroll/preview', data={'name':'Explicitly enrolled','consent':'true'},
                        files=[('files', ('review.jpg', image_response.content, 'image/jpeg'))])
    assert preview.status_code == 200 and client.get('/api/people').json()['people'] == []
    client.post('/api/enroll/commit',json={'token':preview.json()['token']})
    assert len(client.get('/api/people').json()['people']) == 1
    assert client.delete('/api/review/unknowns/'+result['items'][0]['id']).status_code == 200
    assert client.get(photo['url']).status_code == 400


def test_review_paging_and_cross_origin_rejected(client):
    assert client.get('/api/review/unknowns?offset=-1').status_code == 400
    assert client.get('/api/review/unknowns?limit=101').status_code == 400
    assert client.get('/api/review/unknowns',headers={'Origin':'https://untrusted.test'}).status_code == 403
