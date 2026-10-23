const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const context = {AbortController, setTimeout, clearTimeout};
vm.runInNewContext(
  source.slice(source.indexOf('  const esc ='), source.indexOf('  const icons =')) +
  source.slice(source.indexOf('  function recognitionName('), source.indexOf('  class CameraTile')) +
  '\nglobalThis.Cache = RecognitionSnapshots;', context,
);
const Cache = context.Cache;
const response = (type = 'image/jpeg', bytes = 'test crop') => new Response(new Blob([bytes], {type}));
const entry = (id = 'crop1', quality = 70, personId = 'a') => ({
  key: '1:track1', personId, name: personId, snapshotCandidate: {id, quality},
});

function setup(fetcher) {
  const created = [], revoked = [], changes = [];
  const cache = new Cache(() => changes.push(true), {
    fetcher,
    createUrl: () => {const url = 'blob:crop-' + created.length; created.push(url); return url;},
    revokeUrl: (url) => revoked.push(url),
  });
  return {cache, created, revoked, changes};
}

test('default browser fetch is not invoked as a cache object method', async () => {
  context.fetch = function () {
    'use strict';
    assert.equal(this, undefined);
    return Promise.resolve(response());
  };
  const cache = new Cache(() => {}, {createUrl: () => 'blob:default', revokeUrl: () => {}});
  const e = entry();
  cache.sync([e]);
  await cache.get(e).promise;
  assert.equal(cache.get(e).url, 'blob:default');
  cache.clear();
});

test('downloads once and retains a blob after the source crop disappears', async () => {
  let calls = 0;
  const {cache, created} = setup(async () => {calls++; return response();});
  const e = entry();
  cache.sync([e]);
  await cache.get(e).promise;
  assert.equal(cache.get(e).url, 'blob:crop-0');
  assert.equal(cache.get(e).blob.size, 9);
  cache.sync([e]);
  cache.sync([{...e, snapshotCandidate: null}]);
  assert.equal(calls, 1);
  assert.equal(created.length, 1);
  assert.equal(cache.get(e).url, 'blob:crop-0');
  cache.clear();
});

test('only a better quality crop replaces a retained snapshot', async () => {
  const {cache, created, revoked} = setup(async () => response());
  cache.sync([entry()]);
  await cache.get(entry()).promise;
  cache.sync([entry('worse', 60)]);
  assert.equal(created.length, 1);
  cache.sync([entry('better', 85)]);
  await cache.get(entry()).promise;
  assert.equal(cache.get(entry()).sampleId, 'better');
  assert.deepEqual(revoked, ['blob:crop-0']);
  cache.clear();
});

test('a failed replacement preserves the old crop without repeated fetches', async () => {
  let calls = 0;
  const {cache} = setup(async () => ++calls === 1 ? response() : new Response('', {status: 404}));
  cache.sync([entry()]);
  await cache.get(entry()).promise;
  cache.sync([entry('expired', 90)]);
  await cache.get(entry()).promise;
  cache.sync([entry('expired', 90)]);
  assert.equal(calls, 2);
  assert.equal(cache.get(entry()).url, 'blob:crop-0');
  cache.clear();
});

test('identity changes release the previous person snapshot', async () => {
  const {cache, revoked} = setup(async () => response());
  cache.sync([entry()]);
  await cache.get(entry()).promise;
  const replacement = entry('other', 80, 'b');
  cache.sync([replacement]);
  assert.equal(cache.get(entry()), undefined);
  await cache.get(replacement).promise;
  assert.deepEqual(revoked, ['blob:crop-0']);
  assert.equal(cache.get(replacement).url, 'blob:crop-1');
  cache.clear();
});

test('late responses for removed entries cannot repopulate the cache', async () => {
  let resolve;
  const {cache, created, changes} = setup(() => new Promise(r => {resolve = r;}));
  const e = entry();
  cache.sync([e]);
  const pending = cache.get(e).promise;
  cache.sync([]);
  resolve(response());
  await pending;
  assert.equal(cache.items.size, 0);
  assert.equal(created.length, 0);
  assert.equal(changes.length, 0);
});

test('rejects invalid or oversized content and uses only the local sample route', async () => {
  const paths = [];
  const {cache, created} = setup(async url => {paths.push(url); return response('text/html');});
  const e = entry('https://wrong.invalid/photo');
  cache.sync([e]);
  await cache.get(e).promise;
  assert.equal(paths[0], '/api/samples/https%3A%2F%2Fwrong.invalid%2Fphoto/photo');
  assert.equal(created.length, 0);
  cache.clear();
  const tooLarge = setup(async () => new Response(new Blob(['x'], {type: 'image/jpeg'}), {headers: {'content-length': '2000001'}}));
  tooLarge.cache.sync([entry()]);
  await tooLarge.cache.get(entry()).promise;
  assert.equal(tooLarge.created.length, 0);
  tooLarge.cache.clear();
});

test('eviction and clear revoke retained object URLs', async () => {
  const {cache, revoked} = setup(async () => response());
  const a = entry();
  const b = {...entry('second'), key: '1:track2'};
  cache.sync([a, b]);
  await Promise.all([cache.get(a).promise, cache.get(b).promise]);
  cache.sync([b]);
  assert.equal(cache.items.size, 1);
  assert.equal(revoked.length, 1);
  cache.clear();
  assert.equal(cache.items.size, 0);
  assert.equal(revoked.length, 2);
});
