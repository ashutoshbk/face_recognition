"""Explicit single-self-profile photo enrollment, never bulk gallery import."""
import base64
import binascii
import io
import threading

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from core import align_face, assess_quality, jpeg

MAX_PHOTO_BYTES=4_000_000


def decode_photo(encoded):
    try:data=base64.b64decode(encoded,validate=True)
    except (ValueError,binascii.Error):raise ValueError('Choose a valid JPEG or PNG photo')
    if not data or len(data)>MAX_PHOTO_BYTES:raise ValueError('Photo must be smaller than 4 MB')
    try:
        with Image.open(io.BytesIO(data)) as original:
            if original.format not in ('JPEG','PNG'):raise ValueError('Only JPEG and PNG photos are accepted')
            if min(original.size)<80 or max(original.size)>4096 or original.width*original.height>16_000_000:
                raise ValueError('Photo dimensions must be between 80 and 4096 pixels, at most 16 megapixels')
            image=ImageOps.exif_transpose(original).convert('RGB')
            image.thumbnail((1920,1920))
            return cv2.cvtColor(np.asarray(image),cv2.COLOR_RGB2BGR)
    except (UnidentifiedImageError,OSError,Image.DecompressionBombError) as error:
        raise ValueError('Cannot decode this JPEG or PNG photo') from error


class PhotoEnrollment:
    def __init__(self,engine,recognizer,search):
        self.engine,self.recognizer,self.search=engine,recognizer,search
        self.lock=threading.Lock()

    def enroll(self,name,consent,encoded):
        if not consent:raise ValueError('Explicit consent to enroll your own photo is required')
        name=name.strip()
        if not name or len(name)>60:raise ValueError('Enter a tester name of 1-60 characters')
        if not self.lock.acquire(blocking=False):raise ValueError('A photo enrollment is already running')
        try:
            if self.search.gallery.list():raise ValueError('Only one self profile is allowed; delete the existing profile first')
            if not self.search.status()['ready']:raise ValueError('Complete the quick or full synthetic gallery setup before enrollment')
            frame=decode_photo(encoded)
            observations=self.engine.detect(frame)
            faces=observations['faces']
            if len(faces)!=1:raise ValueError('Photo must contain exactly one clearly visible face')
            if sum(h[4]>=.4 for h in observations['heads'])>1:raise ValueError('Photo must contain only the consenting tester')
            quality=assess_quality(frame,faces[0])
            if not quality['accepted']:raise ValueError('Photo rejected: '+', '.join(quality['reasons']))
            crop=align_face(frame,faces[0])
            feature=self.recognizer.encode_photo(crop)
            identifier=self.search.gallery.add_photo(name,feature,jpeg(crop),True)
            self.search.register_or_rollback(identifier)
            return {'id':identifier,'testers':self.search.gallery.list(),'source':'single_uploaded_photo','quality':quality}
        finally:self.lock.release()
