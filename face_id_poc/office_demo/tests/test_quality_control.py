import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vision_backend import Vision, assess_query_quality, assess_quality, encode_jpeg
from enrollment import Enrollment
from runtime import DEFAULTS, Manager


def fixture():
    rng = np.random.default_rng(3)
    frame = rng.integers(80, 170, (300, 300, 3), dtype=np.uint8)
    face = np.array([80, 80, 100, 120, 100, 110, 150, 110, 150, 140,
                     110, 170, 140, 170, .95])
    return frame, face


def test_quality_function_defaults_preserve_original_reference_checks():
    frame, face = fixture()
    query, reference = assess_query_quality(frame, face)
    assert query == reference == assess_quality(frame, face, minimum_width=70)


def test_relaxed_query_does_not_relax_reference_learning():
    frame, face = fixture()
    query, reference = assess_query_quality(frame, face, strictness=0)
    assert query['accepted']
    assert not reference['accepted'] and 'Head angle' in reference['reasons']
    assert query['score'] == reference['score']


@pytest.mark.parametrize('kind', ['clipped', 'confidence'])
def test_invariant_guards_are_not_relaxed(kind):
    frame, face = fixture()
    if kind == 'clipped':
        face[0] = 0
    else:
        face[-1] = .6
    query, _ = assess_query_quality(frame, face, strictness=0)
    assert not query['accepted']


def test_reducing_query_width_keeps_reference_minimum():
    frame, face = fixture()
    face[2] = 50
    query, reference = assess_query_quality(frame, face, minimum_width=40, strictness=0)
    assert query['accepted']
    assert 'Face too small' in reference['reasons']


def test_twenty_pixel_query_does_not_qualify_for_reference_learning():
    frame, _ = fixture()
    face = np.array([80, 80, 20, 25, 85, 87, 95, 87, 90, 93,
                     85, 98, 95, 98, .95])
    query, reference = assess_query_quality(frame, face, minimum_width=20, strictness=0)
    assert query['accepted']
    assert not reference['accepted']
    assert 'Face too small' in reference['reasons']


def test_office_query_default_is_twenty_and_values_below_twenty_are_rejected(tmp_path):
    (tmp_path / 'config').mkdir()
    manager = Manager(tmp_path, tmp_path, 'CPUExecutionProvider', SimpleNamespace())
    assert manager.settings['face_min_width'] == 20
    manager.update_settings(dict(face_min_width=20))
    assert manager.settings['face_min_width'] == 20
    with pytest.raises(ValueError):
        manager.update_settings(dict(face_min_width=19))


def test_manual_registration_stays_strict_with_relaxed_query_settings():
    frame, _ = fixture()
    face = np.array([80, 80, 20, 25, 85, 87, 95, 87, 90, 93,
                     85, 98, 95, 98, .95])
    vision = Vision.__new__(Vision)
    vision.detect = lambda *args, **kwargs: ([], [face])
    manager = SimpleNamespace(vision=vision, gallery=SimpleNamespace(people=lambda: []),
        settings=dict(face_min_width=20, quality_strictness=0))
    result = Enrollment(manager).prepare([('Tester.jpg', encode_jpeg(frame))], consent=True)
    item = result['groups'][0]['items'][0]
    assert not item['accepted']
    assert 'Face too small' in item['reason']


def test_rejected_reference_cannot_be_offered_even_for_confirmed_name(tmp_path):
    manager = Manager(tmp_path, tmp_path, 'CPUExecutionProvider', SimpleNamespace())
    job = dict(epoch=manager.epoch, reference_quality=dict(accepted=False))
    manager._offer_reference(job, dict(state='Confirmed'), None, 1, 1)
    assert not manager.candidates


@pytest.mark.parametrize('value', [-1, 101, float('nan')])
def test_strictness_validation(tmp_path, value):
    manager = Manager(tmp_path, tmp_path, 'CPUExecutionProvider', SimpleNamespace())
    with pytest.raises(ValueError):
        manager.update_settings(dict(quality_strictness=value))


def test_explicit_strict_quality_patch_does_not_change_matching(tmp_path):
    (tmp_path / 'config').mkdir()
    manager = Manager(tmp_path, tmp_path, 'CPUExecutionProvider', SimpleNamespace())
    manager.update_settings(dict(match_threshold=.55, quality_strictness=10, face_min_width=45))
    manager.update_settings(dict(quality_strictness=50, face_min_width=70))
    assert manager.settings['match_threshold'] == .55
    assert manager.settings['quality_strictness'] == 50
    assert manager.settings['face_min_width'] == 70


def test_saved_site_settings_override_new_install_defaults(tmp_path):
    (tmp_path / 'config').mkdir()
    original = Manager(tmp_path, tmp_path, 'CPUExecutionProvider', SimpleNamespace())
    original.update_settings(dict(match_threshold=.55, face_min_width=90, quality_strictness=75,
                                 single_reference_mode='off'))
    restarted = Manager(tmp_path, tmp_path, 'CPUExecutionProvider', SimpleNamespace())
    assert restarted.settings == original.settings
    assert restarted.settings != DEFAULTS
