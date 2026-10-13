from pathlib import Path
import pytest
from rtsp_input import validate_url,RTSPInput
from settings import ROOT

@pytest.mark.parametrize('url',['file:///etc/passwd','http://example.com/video','rtsp://','rtsp://camera:bad/feed','rtsp://camera/\nsecret'])
def test_bad_stream_addresses_are_rejected_without_echo(url):
    with pytest.raises(ValueError) as error:validate_url(url)
    assert url not in str(error.value)

def test_controlled_stream_requires_authorization():
    with pytest.raises(ValueError,match='authorization'):RTSPInput('rtsp://localhost:8554/demo',False)

def test_rtsp_parser_accepts_valid_addresses():
    assert validate_url('rtsp://localhost:8554/demo').startswith('rtsp://')

def test_no_developer_absolute_paths_in_runtime():
    for name in ['app.py','core.py','self_test.py','recognition_process.py','settings.py','rtsp_input.py','synthetic_bench/pipeline.py']:
        content=(ROOT/name).read_text()
        assert '/Users/fovea' not in content and '/Volumes/SamsungT7' not in content
