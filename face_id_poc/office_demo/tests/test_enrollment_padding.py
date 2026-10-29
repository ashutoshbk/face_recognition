import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_backend import padded_enrollment_faces


class Detector:
    def __init__(self, box=None, points=None):
        self.shapes = []
        self.box = box if box is not None else [30, 20, 110, 130, .85]
        self.points = points if points is not None else [[50, 50], [90, 50], [70, 75], [55, 100], [85, 100]]

    def detect(self, image, input_size):
        assert input_size == (640, 640)
        self.shapes.append(image.shape)
        assert np.array_equal(image[:188, :176], np.full((188, 176, 3), 42, np.uint8))
        return np.array([self.box]), np.array([self.points])


def test_small_crop_retries_keep_native_width_and_landmarks():
    detector = Detector()
    faces = padded_enrollment_faces(np.full((188, 176, 3), 42, np.uint8), detector)
    assert detector.shapes == [(256, 256, 3), (320, 320, 3)]
    assert len(faces) == 2
    assert all(face[2] == 80 and face[3] == 110 for face in faces)
    assert all(np.array_equal(face[4:14], np.array(detector.points).reshape(-1)) for face in faces)


def test_padding_does_not_create_valid_faces_outside_original_image():
    detector = Detector(box=[30, 20, 200, 130, .9])
    assert padded_enrollment_faces(np.full((188, 176, 3), 42, np.uint8), detector) == []


def test_padding_rejects_landmarks_outside_original_image():
    detector = Detector(points=[[50, 50], [190, 50], [70, 75], [55, 100], [85, 100]])
    assert padded_enrollment_faces(np.full((188, 176, 3), 42, np.uint8), detector) == []


def test_large_images_do_not_run_padding_retry():
    detector = Detector()
    assert padded_enrollment_faces(np.zeros((720, 1280, 3), np.uint8), detector) == []
    assert detector.shapes == []
