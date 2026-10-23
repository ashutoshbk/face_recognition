"""Frame-varying synthetic QA marker; no physical camera or real-person images."""
import json
from pathlib import Path

import cv2

ROOT=Path(__file__).resolve().parent
source=cv2.imread(str(ROOT/'qa/synthetic_1.jpg'))
path=ROOT/'qa/synchronized_camera.y4m'
with path.open('wb') as stream:
    stream.write(b'YUV4MPEG2 W960 H540 F15:1 Ip A1:1 C420jpeg\n')
    for index in range(90):
        frame=source.copy()
        # The changing marker reveals drawing the latest video under old detections.
        frame[480:535,:]=((index*37)%200+25,(index*71)%200+25,(index*113)%200+25)
        stream.write(b'FRAME\n')
        stream.write(cv2.cvtColor(frame,cv2.COLOR_BGR2YUV_I420).tobytes())
print(json.dumps({'file':str(path),'frames':90,'fps':15,'synthetic_only':True}))
