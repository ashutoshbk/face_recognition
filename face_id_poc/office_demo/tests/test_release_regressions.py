import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

OFFICE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(OFFICE))
from domain import HeadTrackIdentity, SingleReferenceReview, TrackEvidence


SETTINGS = dict(match_threshold=.45, match_margin=.06,
                admission_threshold=.7, admission_margin=.1)
QUALITY = dict(accepted=True, score=80, width=100)


def ranking(person="A", count=2, score=.9):
    return [dict(person_id=person, name=person, score=score,
                 registered_score=score, registered=[score]*count, live=[],
                 anchor_version=1, registered_revision=1)]


@pytest.mark.parametrize("rate", [.2, .3, .5, 1., 2., 5.])
def test_scheduled_distinct_votes_confirm_at_all_supported_rates(rate):
    identity = HeadTrackIdentity()
    settings = dict(SETTINGS, recognition_hz=rate)
    results = [identity.add(ranking(), 100+i*(1/rate+.1), i, settings)
               for i in range(12)]
    confirmed = [result for result in results if result["state"] == "Confirmed"]
    assert confirmed and any(result["admission"] for result in confirmed)
    assert all(result["votes"] >= 3 for result in confirmed)
    assert all(result["observations"] <= 5 for result in results)


def test_slow_votes_expire_and_duplicates_cannot_confirm():
    evidence = TrackEvidence()
    settings = dict(SETTINGS, recognition_hz=.2)
    evidence.add(ranking(), 100., 1, settings)
    duplicate = evidence.add(ranking(), 105., 1, settings)
    assert duplicate["votes"] == duplicate["observations"] == 1
    evidence.add(ranking(), 110., 2, settings)
    assert evidence.summary(136.)["state"] == "Unknown"
    fresh = evidence.add(ranking(), 136., 3, settings)
    assert fresh["votes"] == fresh["observations"] == 1
    assert not fresh["admission"]


def test_slow_cadence_still_requires_consecutive_strict_votes():
    identity = HeadTrackIdentity()
    settings = dict(SETTINGS, recognition_hz=.2)
    for i, score in enumerate([.9, .9, .5, .9, .9]):
        result = identity.add(ranking(score=score), 100+i*5.1, i, settings)
        assert not result["admission"]
    assert identity.add(ranking(), 125.5, 5, settings)["admission"]


@pytest.mark.parametrize("period", [1., 1.8, 3.4, 5., 5.2])
def test_review_learns_slow_cadence_without_reducing_independent_count(period):
    review = SingleReferenceReview()
    for i in range(6):
        result = review.observe(ranking(count=1), "A", QUALITY,
                                100+i*period, i, str(i))
        if result:
            assert len(review.observations) == result["count"] == 5
            assert len({item["frame"] for item in review.observations}) == 5
            assert len({item["digest"] for item in review.observations}) == 5
            assert result["span_seconds"] >= 4.
            break
        assert len(review.observations) < 5
    else:
        pytest.fail("Distinct scheduled observations never produced a review proposal")


def test_review_does_not_bridge_initial_gap_or_unbounded_gap():
    review = SingleReferenceReview()
    for i in range(4):
        assert review.observe(ranking(count=1), "A", QUALITY, 100+i, i, str(i)) is None
    assert review.observe(ranking(count=1), "A", QUALITY, 108., 4, "4") is None
    assert len(review.observations) == 1
    assert review.observe(ranking(count=1), "A", QUALITY, 115., 5, "5") is None
    assert len(review.observations) == 1 and review.interval == 1.


def test_review_slow_cadence_duplicates_and_failed_quality_do_not_count():
    review = SingleReferenceReview()
    for i in range(5):
        assert review.observe(ranking(count=1), "A", QUALITY, 100+i*5., i, str(i)) is None
    assert len(review.observations) == 4
    assert review.observe(ranking(count=1), "A", QUALITY, 125., 4, "duplicate-frame") is None
    assert review.observe(ranking(count=1), "A", QUALITY, 125., 5, "4") is None
    assert len(review.observations) == 4
    assert review.observe(ranking(count=1), "A", dict(QUALITY, accepted=False),
                          125., 5, "5") is None
    assert not review.observations and review.interval == 1.
    assert review.observe(ranking(count=1), "A", QUALITY, 130., 6, "6") is None
    assert len(review.observations) == 1


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js required for UI handler test")
@pytest.mark.parametrize("response,expected,error", [
    ({"accepted": True}, "Candidate approved.", False),
    ({"accepted": False, "reason": "Duplicate or near-identical reference"},
     "Candidate not saved: Duplicate or near-identical reference", True),
    ({"accepted": False, "reason": "Five better or equal references already stored"},
     "Candidate not saved: Five better or equal references already stored", True),
    ({"accepted": False}, "Candidate not saved: The reference was not accepted.", True),
])
def test_approval_ui_reports_server_acceptance(response, expected, error):
    script = r"""
const fs = require('node:fs');
const vm = require('node:vm');
const assert = require('node:assert/strict');
const source = fs.readFileSync(process.argv[1], 'utf8');
const start = source.indexOf("  $('candidate-review-form').addEventListener('submit', async (event) => {");
const end = source.indexOf('  const settingNames =', start);
assert(start >= 0 && end > start);
let submit;
let closed = 0;
const toasts = [];
const refreshed = [];
const elements = {
  'candidate-review-form': {addEventListener: (event, handler) => {submit = handler;}, reportValidity: () => true},
  'candidate-review-confirm': {disabled: false},
  'candidate-review-dialog': {close: () => {closed++;}},
  'candidate-identity-verified': {checked: true},
};
const context = {
  $: (id) => elements[id],
  syncCandidateReview: () => {},
  candidateReview: {candidate: {id: 'test'}, busy: false},
  app: {busy: new Set()},
  notice: () => {}, updateActions: () => {},
  api: async () => JSON.parse(process.argv[2]),
  idPath: encodeURIComponent, json: (method, body) => ({method, body}),
  describe: String, toast: (...args) => toasts.push(args),
  candidateFlight: null, stateFlight: null,
  refreshCandidates: async () => refreshed.push('candidates'),
  refreshState: async () => refreshed.push('state'),
};
vm.runInNewContext(source.slice(start, end), context);
(async () => {
  await submit({preventDefault() {}});
  assert.equal(closed, 1);
  assert.equal(toasts.length, 1);
  assert.equal(toasts[0][0], process.argv[3]);
  assert.equal(Boolean(toasts[0][1]), JSON.parse(process.argv[4]));
  assert.deepEqual(refreshed, ['candidates', 'state']);
  assert.equal(context.app.busy.size, 0);
  assert.equal(context.candidateReview.busy, false);
})().catch((error) => {console.error(error); process.exitCode = 1;});
"""
    subprocess.run([shutil.which("node"), "-e", script, str(OFFICE / "static/app.js"),
                    json.dumps(response), expected, json.dumps(error)],
                   check=True, capture_output=True, text=True, timeout=10)
