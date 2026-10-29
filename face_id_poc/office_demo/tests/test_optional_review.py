import importlib
import io
import json
import sqlite3
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from optional_review import REVIEW_DEFAULTS, ReviewStore, ReviewService, validate_review_settings
from runtime import DEFAULTS, Manager
from domain import Gallery, unit


def jpeg(value=0):
    out = io.BytesIO()
    Image.new("RGB", (100, 100), ((value*31)%255, 80, 110)).save(out, "JPEG")
    return out.getvalue()


def item(value=0, source="track-1", captured=100., unknown=True):
    return dict(source_key=source, camera_id="camera", camera_name="Consented test", track_id="7",
                captured=captured, sequence=value, image=jpeg(value), quality=dict(accepted=True, score=75+value, width=100),
                unknown=unknown, person_id=None if unknown else "person", strict_match=not unknown, score=.90, margin=.25)


def settings(**changes):
    return dict(DEFAULTS, review_consent=True, **changes)


class TestReviewStore:
    def test_no_gallery_or_vectors_created(self, tmp_path):
        store = ReviewStore(tmp_path)
        store.save_unknown(item(), settings())
        assert not (tmp_path / "database/office.sqlite3").exists()
        with store.db() as db:
            cols = [r[1] for r in db.execute("PRAGMA table_info(photos)")]
            assert "vector" not in cols and "person_id" not in cols

    def test_five_best_not_latest(self, tmp_path):
        store = ReviewStore(tmp_path)
        for value in range(7):
            store.save_unknown(item(value), settings())
        rows = store.list("unknowns")["items"]
        assert len(rows) == 1
        assert [p["score"] for p in rows[0]["photos"]] == [81, 80, 79, 78, 77]
        assert not store.save_unknown(item(1), settings())
        assert len(list(store.images.glob("*.jpg"))) == 5

    def test_duplicates_not_saved(self, tmp_path):
        store = ReviewStore(tmp_path)
        assert store.save_unknown(item(), settings())
        assert not store.save_unknown(item(), settings())

    def test_quota_and_expiry_remove_files(self, tmp_path):
        now = [1000.]
        store = ReviewStore(tmp_path, clock=lambda: now[0])
        for value in range(3):
            now[0] += 1
            store.save_unknown(item(value, source=str(value)), settings(unknown_max_records=2))
        assert store.list("unknowns")["total"] == 2
        assert len(list(store.images.glob("*.jpg"))) == 2
        now[0] += 25*3600
        store.cleanup()
        assert store.counts()["unknowns"] == 0
        assert not list(store.images.glob("*.jpg"))

    def test_persistence_and_delete(self, tmp_path):
        store = ReviewStore(tmp_path)
        store.save_unknown(item(), settings())
        other = ReviewStore(tmp_path)
        row = other.list("unknowns")["items"][0]
        identifier = row["photos"][0]["id"]
        assert other.photo(identifier) == jpeg()
        other.delete("unknowns", row["id"])
        with pytest.raises(ValueError):
            other.photo(identifier)

    def test_notice_cooldown_review_and_person_removal(self, tmp_path):
        now = [1000.]
        store = ReviewStore(tmp_path, clock=lambda: now[0])
        person = dict(id="person", name="Test participant", registered=[dict(id="reference",score=90)])
        store.set_flag("person", "Consented notice demonstration", 24)
        flag = store.flags()[0]
        assert store.save_notice(item(unknown=False), person, flag, settings(), 5)
        assert not store.save_notice(item(unknown=False), person, flag, settings(), 6)
        row = store.list("notices")["items"][0]
        store.acknowledge(row["id"])
        assert store.counts()["pending_notices"] == 0
        store.remove_person("person")
        assert store.flags() == [] and store.counts()["notices"] == 0
        assert not list(store.images.glob("*.jpg"))

    def test_flag_validation(self,tmp_path):
        store=ReviewStore(tmp_path)
        with pytest.raises(ValueError): store.set_flag("p", "", 24)
        with pytest.raises(ValueError): store.set_flag("p", "reason", -1)

    def test_removed_reference_clears_its_notice(self,tmp_path):
        store=ReviewStore(tmp_path)
        person=dict(id='person',name='Participant',registered=[dict(id='reference',score=90)])
        store.save_notice(item(unknown=False),person,dict(reason='Consented test'),settings(),5)
        store.remove_reference('reference')
        assert store.counts()['notices']==0 and not list(store.images.glob('*.jpg'))


class TestReviewService:
    def service(self, tmp_path, monkeypatch, **changes):
        monkeypatch.setattr("optional_review.time.monotonic", lambda: self.now)
        self.now = 100.
        gallery = SimpleNamespace(people=lambda:[dict(id="person",name="Participant",registered=[dict(id="r",score=90)])])
        return ReviewService(tmp_path,gallery,settings(**changes))

    def send(self, service, value, unknown=True, **patch):
        self.now += 1
        data = dict(item(value,captured=self.now,unknown=unknown),review_epoch=service.epoch)
        data.update(patch)
        service.process(data)

    def test_defaults_off_and_optin_required(self,tmp_path,monkeypatch):
        assert not DEFAULTS["unknown_capture_enabled"] and not DEFAULTS["flagged_review_enabled"]
        with pytest.raises(ValueError): validate_review_settings(dict(DEFAULTS, unknown_capture_enabled=True))
        service=self.service(tmp_path,monkeypatch)
        for i in range(5): self.send(service,i)
        assert service.store.counts()["unknowns"] == 0

    def test_repeated_unknown_not_one_frame(self,tmp_path,monkeypatch):
        service=self.service(tmp_path,monkeypatch,unknown_capture_enabled=True)
        self.send(service,0); self.send(service,1)
        assert service.store.counts()["unknowns"] == 0
        self.send(service,2)
        assert service.store.counts()["unknowns"] == 1

    def test_strict_storage_and_rejected_stale_results(self,tmp_path,monkeypatch):
        service=self.service(tmp_path,monkeypatch,unknown_capture_enabled=True)
        for i in range(5): self.send(service,i,quality=dict(accepted=True,score=10,width=20))
        self.send(service,10,captured=0)
        service.current=lambda row:False
        self.send(service,11)
        assert service.store.counts()["unknowns"] == 0
        assert service.metrics["quality_rejected"] == 5 and service.metrics["stale_dropped"] == 2

    def test_disable_invalidates_queue_without_deleting_existing(self,tmp_path,monkeypatch):
        service=self.service(tmp_path,monkeypatch,unknown_capture_enabled=True)
        for i in range(3): self.send(service,i)
        old_epoch=service.epoch
        service.configure(dict(DEFAULTS))
        self.send(service,4,review_epoch=old_epoch)
        assert service.store.counts()["unknowns"] == 1
        assert len(service.store.list("unknowns")["items"][0]["photos"]) == 1

    def test_unknown_never_becomes_identity(self,tmp_path,monkeypatch):
        service=self.service(tmp_path,monkeypatch,unknown_capture_enabled=True)
        for i in range(3): self.send(service,i)
        for i in range(3,7): self.send(service,i,unknown=False,person_id=None,strict_match=False)
        assert service.store.counts()["notices"] == 0

    def test_single_reference_requires_five_fresh_strict_matches(self,tmp_path,monkeypatch):
        service=self.service(tmp_path,monkeypatch,flagged_review_enabled=True)
        service.store.set_flag("person","Consented test",24)
        for i in range(4): self.send(service,i,unknown=False)
        assert service.store.counts()["notices"] == 0
        self.send(service,4,unknown=False)
        assert service.store.counts()["notices"] == 1

    def test_nonblocking_submit(self,tmp_path,monkeypatch):
        service=self.service(tmp_path,monkeypatch,unknown_capture_enabled=True)
        entered=threading.Event(); release=threading.Event()
        def busy():
            with service.lock: entered.set(); release.wait(2)
        thread=threading.Thread(target=busy); thread.start(); assert entered.wait(1)
        service.submit(item())
        release.set(); thread.join()
        assert service.metrics["queue_dropped"] == 1 and service.queue.empty()

    def test_shorter_retention_applies_to_existing_records(self,tmp_path,monkeypatch):
        service=self.service(tmp_path,monkeypatch,unknown_capture_enabled=True)
        wall=[1000.]
        service.store.clock=lambda:wall[0]
        for i in range(3): self.send(service,i)
        wall[0] += 7200
        service.configure(dict(service.settings,unknown_retention_hours=1))
        assert service.store.counts()["unknowns"] == 0
        assert not list(service.store.images.glob('*.jpg'))


def test_manager_optional_settings_are_validated(tmp_path):
    manager=Manager(tmp_path,tmp_path,"CPUExecutionProvider",SimpleNamespace())
    for changes in ({"unknown_capture_enabled":True}, {"review_consent":"true"}, {"unknown_max_records":1.5}, {"review_match_threshold":.1}):
        with pytest.raises(ValueError): manager.update_settings(changes)
    manager.update_settings(dict(review_consent=True,unknown_capture_enabled=True))
    assert manager.settings["unknown_capture_enabled"]
    manager.update_settings(dict(DEFAULTS))
    assert not manager.settings["unknown_capture_enabled"]


def test_invalid_persisted_consent_does_not_enable_capture(tmp_path):
    (tmp_path/'config').mkdir()
    (tmp_path/'config/settings.json').write_text(json.dumps(dict(review_consent='false',unknown_capture_enabled=True)))
    manager=Manager(tmp_path,tmp_path,'CPUExecutionProvider',SimpleNamespace())
    assert manager.settings['review_consent'] is False
    assert manager.settings['unknown_capture_enabled'] is False
