from pathlib import Path
import pytest
from settings import MODELS,ROOT

def pytest_collection_modifyitems(items):
    have_models=all((MODELS/name).exists() for name in ['arcface-r50-w600k-112x112.onnx','scrfd-kps-2.5g-640x640.onnx','yolox-tiny-crowdhuman-416x416.onnx'])
    for item in items:
        name=Path(str(item.fspath)).name
        if name in ('test_api.py','test_real_models.py') and not have_models:
            item.add_marker(pytest.mark.skip(reason='Separate model bundle required for model/API integration tests'))
        elif name=='test_real_models.py' and not (ROOT/'qa/synthetic_1.jpg').exists():
            item.add_marker(pytest.mark.skip(reason='Run lab.py fixtures after downloading the research dataset'))
