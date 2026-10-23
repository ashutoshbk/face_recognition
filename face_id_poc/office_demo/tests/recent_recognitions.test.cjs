const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const context = {};
const prelude = source.slice(source.indexOf('  const esc ='), source.indexOf('  const icons ='));
const history = source.slice(source.indexOf('  function recognitionName('), source.indexOf('  class CameraTile'));
const markup = source.slice(source.indexOf('  function recognitionTime('), source.indexOf('  function evidenceMarkup('));
vm.runInNewContext(prelude + history + markup + '\nglobalThis.exports = {RecognitionHistory, identity, recognitionMarkup};', context);
const {RecognitionHistory, identity, recognitionMarkup} = context.exports;
const people = [{id: 'a', name: 'Alice'}, {id: 'b', name: 'Bob'}];
const matched = (id = 1, extra = {}) => ({id, person_id: 'a', name: 'Alice', state: 'Tentative', score: 0.6, ...extra});
const camera = (tracks = [], generation = 1) => ({generation, tracks});

test('unknown tracks never enter recent recognitions', () => {
  const h = new RecognitionHistory();
  assert.equal(h.update(camera([{id: 1, name: null, state: 'Unknown', score: 0.2}]), people, 1000).length, 0);
  assert.equal(h.update(camera([matched(1, {state: 'Association ambiguous'})]), people, 1001).length, 0);
});

test('plain name is tentative, never implicitly confirmed', () => {
  assert.equal(identity(matched()).label, 'Tentative');
  assert.equal(identity(matched(1, {state: 'Confirmed'})).label, 'Confirmed');
  assert.equal(identity(matched(1, {state: 'Tracked'})).label, 'Head-tracked');
});

test('same track updates one stable row with latest score and timestamps', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched()]), people, 1000);
  const row = h.entries[0];
  h.update(camera([matched(1, {state: 'Confirmed', score: 0.53})]), people, 2000);
  assert.equal(h.entries.length, 1);
  assert.equal(h.entries[0], row);
  assert.equal(row.firstSeen, 1000);
  assert.equal(row.lastSeen, 2000);
  assert.equal(row.score, 0.53);
  assert.equal(row.decision.label, 'Confirmed');
});

test('head turn and departure retain the last row without claiming a new face match', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched(1, {state: 'Confirmed'})]), people, 1000);
  h.update(camera([matched(1, {state: 'Tracked', face_visible: false})]), people, 2000);
  assert.match(recognitionMarkup(h.entries[0]), /Head-tracked/);
  h.update(camera([]), people, 3000);
  assert.equal(h.entries.length, 1);
  assert.equal(h.entries[0].lastSeen, 2000);
  assert.match(recognitionMarkup(h.entries[0]), /Earlier match/);
  assert.match(recognitionMarkup(h.entries[0]), /Confirmed/);
  assert.doesNotMatch(recognitionMarkup(h.entries[0]), /Head-tracked|Current track/);
});

test('contradictory or unavailable current evidence is explicitly not revalidated', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched(1, {state: 'Confirmed'})]), people, 1000);
  h.update(camera([{id: 1, name: null, state: 'Unknown', score: 0.1}]), people, 2000);
  const row = h.entries[0];
  assert.equal(row.name, 'Alice');
  assert.equal(row.lastSeen, 1000);
  assert.equal(row.score, 0.6);
  assert.match(recognitionMarkup(row), /Not revalidated/);
  assert.match(recognitionMarkup(row), /No current match/);
  assert.doesNotMatch(recognitionMarkup(row), />Confirmed</);
});

test('a corrected identity replaces the row and cannot inherit confirmation', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched(1, {state: 'Confirmed'})]), people, 1000);
  h.update(camera([matched(1, {person_id: 'b', name: 'Bob', score: 0.7})]), people, 2000);
  assert.equal(h.entries.length, 1);
  const row = h.entries[0];
  assert.equal(row.name, 'Bob');
  assert.equal(row.firstSeen, 2000);
  assert.equal(row.decision.label, 'Tentative');
  assert.match(recognitionMarkup(row), /Updated from Alice/);
  assert.doesNotMatch(recognitionMarkup(row), />Confirmed</);
});

test('new arrivals go first; repeat matches do not reorder existing entries', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched(1)]), people, 1000);
  h.update(camera([matched(1), matched(2)]), people, 2000);
  h.update(camera([matched(1)]), people, 3000);
  assert.equal(h.entries.map(x => x.trackId).join(','), '2,1');
});

test('twenty-row limit cannot cycle evicted active tracks back into the list', () => {
  const h = new RecognitionHistory();
  const tracks = Array.from({length: 25}, (_, i) => matched(i));
  h.update(camera(tracks), people, 1000);
  const order = h.entries.map(x => x.trackId).join(',');
  for (let i = 0; i < 10; i++) h.update(camera(tracks), people, 2000 + i);
  assert.equal(h.entries.length, 20);
  assert.equal(h.entries.map(x => x.trackId).join(','), order);
  assert.equal(h.byTrack.size, 20);
  assert.equal(h.suppressed.size, 5);
});

test('camera restart cannot merge a reused track ID with an earlier identity', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched(1)], 1), people, 1000);
  h.update(camera([matched(1, {person_id: 'b', name: 'Bob'})], 2), people, 2000);
  assert.equal(h.entries.length, 2);
  assert.equal(h.entries[0].name, 'Bob');
  assert.equal(h.entries[1].name, 'Alice');
  assert.equal(h.entries[1].present, false);
});

test('deleted people disappear; renamed people use the current roster name', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched()]), people, 1000);
  h.update(camera([]), [{id: 'a', name: 'Alice Smith'}], 2000);
  assert.equal(h.entries[0].name, 'Alice Smith');
  h.update(camera([matched()]), [{id: 'b', name: 'Bob'}], 3000);
  assert.equal(h.entries.length, 0);
});

test('clearing for changed settings removes old threshold decisions', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched()]), people, 1000);
  h.clear();
  assert.equal(h.entries.length, 0);
  assert.equal(h.byTrack.size, 0);
  assert.equal(h.suppressed.size, 0);
});

test('history stores metadata only and markup escapes names', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched(1, {name: '<img onerror=bad>', person_id: null, samples: [{image: 'private'}]})]), people, 1000);
  assert.equal(h.entries[0].samples, undefined);
  const html = recognitionMarkup(h.entries[0]);
  assert.match(html, /&lt;img onerror=bad&gt;/);
  assert.doesNotMatch(html, /<img/);
  assert.match(html, /First seen/);
  assert.match(html, /Last seen/);
});

test('camera panel uses history while all raw tracks remain in evidence', () => {
  assert.match(source, /recognitionMarkup\(entry, this\.snapshots\.get\(entry\)\)/);
  assert.doesNotMatch(source, /tracks\.map\(trackMarkup\)/);
  assert.match(source, /evidenceMarkup\(tracks\)/);
  assert.match(source, /this\.history\.clear\(\)/);
});

test('snapshot candidate belongs to the named person and selects the best quality', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched(1, {samples: [
    {id: 'wrong-person', person_id: 'b', state: 'Confirmed', quality_score: 100},
    {id: 'unknown', person_id: 'a', state: 'Unknown', quality_score: 99},
    {id: 'small', person_id: 'a', state: 'Tentative', quality_score: 40},
    {id: 'best', person_id: 'a', state: 'Confirmed', quality_score: 80},
  ]})]), people, 1000);
  assert.equal(h.entries[0].snapshotCandidate.id, 'best');
  assert.equal(h.entries[0].snapshotCandidate.quality, 80);
  h.update(camera([matched(1, {person_id: 'b', name: 'Bob', samples: []})]), people, 2000);
  assert.equal(h.entries[0].snapshotCandidate, null);
});

test('observed snapshot is clickable and never falls back to an enrollment photo', () => {
  const h = new RecognitionHistory();
  h.update(camera([matched()]), people, 1000);
  const html = recognitionMarkup(h.entries[0], {url: 'blob:observed-crop'});
  assert.match(html, /data-action="recognition-photo"/);
  assert.match(html, /Observed camera face crop/);
  assert.match(html, /blob:observed-crop/);
  assert.doesNotMatch(html, /api\/photo\//);
  assert.match(recognitionMarkup(h.entries[0]), /No snapshot/);
});
