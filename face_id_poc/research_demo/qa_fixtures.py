"""Small synthetic fixtures for local QA, not a gallery importer."""
import json
import subprocess
import zipfile
from pathlib import Path

import cv2
import numpy as np

from core import Engine, assess_quality
from settings import MODELS,DATASET

ROOT = Path(__file__).resolve().parent
QA = ROOT / "qa"
QA.mkdir(exist_ok=True)
engine = Engine(MODELS)
source = DATASET/"subjects_0-1999_72_imgs.zip"
fixtures = []
with zipfile.ZipFile(source) as archive:
    # Read a few fictional subjects only; never import the full face dataset.
    names = [name for name in archive.namelist() if name.endswith(".png")]
    folders = list(dict.fromkeys(str(Path(name).parent) for name in names))[:12]
    for folder in folders:
        for name in [name for name in names if str(Path(name).parent) == folder][:8]:
            small = cv2.imdecode(np.frombuffer(archive.read(name),np.uint8),cv2.IMREAD_COLOR)
            frame = np.full((540,960,3),55,dtype=np.uint8)
            resized = cv2.resize(small,(280,280),interpolation=cv2.INTER_CUBIC)
            frame[130:410,340:620] = resized
            observations = engine.detect(frame)
            faces = observations['faces']
            if len(faces) == 1 and assess_quality(frame,faces[0])["accepted"]:
                filename = f"synthetic_{len(fixtures)+1}.jpg"
                cv2.imwrite(str(QA / filename),frame)
                fixtures.append({"file":filename,"source":name,"quality":assess_quality(frame,faces[0])})
                break
        if len(fixtures) == 3:
            break
assert len(fixtures) == 3, "Could not obtain three clear synthetic QA fixtures"
(QA / "fixtures.json").write_text(json.dumps(fixtures,indent=2))
subprocess.run(["ffmpeg","-y","-hide_banner","-loglevel","error","-loop","1","-i",str(QA/fixtures[0]["file"]),"-t","5","-vf","fps=15,format=yuv420p","-f","yuv4mpegpipe",str(QA/"synthetic_camera.y4m")],check=True)
print(json.dumps(fixtures,indent=2))
