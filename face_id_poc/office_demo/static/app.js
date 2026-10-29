(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const esc = (value) => String(value ?? '').replace(/[&<>"']/g, (c) => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c]));
  const list = (value) => Array.isArray(value) ? value : [];
  const idPath = (id) => encodeURIComponent(String(id));
  const numeric = (value, digits = 3) => value !== null && value !== '' && value !== undefined && Number.isFinite(Number(value)) ? Number(value).toFixed(digits) : 'Not reported';
  const icons = (root = document) => window.lucide?.createIcons({root, attrs: {'aria-hidden': 'true'}});
  const running = (camera) => ['running', 'starting', 'connecting', 'reconnecting', 'active'].includes(String(camera.state).toLowerCase());
  const app = {tab: 'cameras', data: null, connected: false, busy: new Set(), cameras: new Map(), peopleKey: '', candidatesKey: '', candidates: []};
  const candidateReview = {candidate: null, key: '', busy: false, referenceLoaded: false, cropLoaded: false};
  let stateTimer, stateFlight, candidateTimer, candidateFlight, toastTimer, recognitionPhotoUrl;
  let disposed = false;

  function redact(value) {
    return String(value ?? '').replace(/rtsps?:\/\/[^\s<>"']+/gi, '[RTSP URL redacted]').replace(/(https?:\/\/)[^\s/@]+:[^\s/@]+@/gi, '$1[redacted]@');
  }

  function describe(value) {
    if (value === undefined || value === null) return 'Not reported';
    if (typeof value === 'boolean') return value ? 'Yes' : 'No';
    return redact(typeof value === 'object' ? JSON.stringify(value, null, 2) : value);
  }

  function sourceLabel(value) {
    if (!value) return 'Address not reported';
    try {
      const url = new URL(String(value));
      if (['rtsp:', 'rtsps:'].includes(url.protocol)) return url.protocol + '//' + url.host + url.pathname + (url.search ? '?[redacted]' : '');
    } catch { /* The API may already return a masked address. */ }
    return redact(value);
  }

  function imageUrl(value, fallback = '') {
    try {
      const url = new URL(value || fallback, location.origin);
      return url.origin === location.origin && !url.username && !url.password && ['http:', 'https:'].includes(url.protocol) ? url.href : '';
    } catch { return ''; }
  }

  function notice(id, text) {
    $(id).textContent = text ? redact(text) : '';
    $(id).hidden = !text;
  }

  function toast(message, error = false) {
    clearTimeout(toastTimer);
    $('toast').textContent = redact(message);
    $('toast').classList.toggle('error', error);
    $('toast').hidden = false;
    toastTimer = setTimeout(() => { $('toast').hidden = true; }, error ? 7000 : 4200);
  }

  async function api(path, options = {}) {
    const {timeoutMs = 20000, ...requestOptions} = options;
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), timeoutMs);
    try {
      const response = await fetch(path, {cache: 'no-store', credentials: 'same-origin', ...requestOptions, signal: controller.signal});
      const body = response.status === 204 ? '' : await response.text();
      let data;
      try { data = body ? JSON.parse(body) : {}; } catch { data = null; }
      if (!response.ok) {
        const detail = data?.detail ?? data?.error ?? data?.message;
        throw new Error(detail ? describe(detail) : 'Request failed (' + response.status + ').');
      }
      if (data === null) throw new Error('The server returned an invalid JSON response.');
      return data;
    } catch (error) {
      if (error.name === 'AbortError') throw new Error('Request timed out. Server outcome is unknown; refresh before retrying.');
      throw error;
    } finally { clearTimeout(timeout); }
  }

  function json(method, body) {
    return {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
  }

  async function confirmAction(title, message, label = 'Delete', danger = true) {
    if ($('confirm-dialog').open) return false;
    $('confirm-title').textContent = title;
    $('confirm-message').textContent = message;
    $('confirm-action').textContent = label;
    $('confirm-action').className = danger ? 'danger-button' : 'primary';
    $('confirm-dialog').returnValue = 'cancel';
    return new Promise((resolve) => {
      $('confirm-dialog').addEventListener('close', () => resolve($('confirm-dialog').returnValue === 'confirm'), {once: true});
      $('confirm-dialog').showModal();
    });
  }

  async function mutate(key, path, options, message) {
    if (app.busy.has(key)) return false;
    app.busy.add(key);
    updateActions();
    try {
      await api(path, options);
      if (message) toast(message);
      // Wait out any older state request before reading the mutation's result.
      if (stateFlight) await stateFlight;
      await refreshState();
      return true;
    } catch (error) {
      toast(error.message || 'Request failed.', true);
      return false;
    } finally {
      app.busy.delete(key);
      updateActions();
    }
  }

  function recognitionName(track) {
    const name = track.person_name ?? track.identity?.name ?? track.name;
    return typeof name === 'string' && name.trim() ? name.trim() : null;
  }

  function identity(track) {
    const state = String(track.identity_status ?? track.match_status ?? track.identity?.status ?? track.state ?? track.status ?? '').toLowerCase();
    if (state === 'tracked') return {label: 'Head-tracked', className: 'tracked'};
    if (track.confirmed === true || track.identity_confirmed === true || state === 'confirmed') return {label: 'Confirmed', className: 'confirmed'};
    if (recognitionName(track)) return {label: 'Tentative', className: 'tentative'};
    return {label: 'Unconfirmed', className: 'neutral'};
  }

  class RecognitionHistory {
    constructor(limit = 20) {
      this.limit = limit;
      this.clear();
    }

    clear() {
      this.entries = [];
      this.byTrack = new Map();
      this.suppressed = new Set();
      this.generation = undefined;
      this.session = 0;
    }

    update(camera, people, now = Date.now()) {
      if (this.generation !== camera.generation) {
        this.generation = camera.generation;
        this.session++;
        this.byTrack.clear();
        this.suppressed.clear();
      }
      const roster = new Map(list(people).map((person) => [String(person.id), person.name]));
      this.entries = this.entries.filter((entry) => !entry.personId || roster.has(entry.personId));
      const retained = new Set(this.entries);
      for (const [key, entry] of this.byTrack) if (!retained.has(entry)) this.byTrack.delete(key);
      for (const entry of this.entries) {
        entry.present = false;
        entry.revalidated = false;
        if (entry.personId) entry.name = roster.get(entry.personId);
      }
      const tracks = list(camera.tracks);
      const keyFor = (track) => this.session + ':' + String(track.id ?? track.track_id);
      const present = new Set(tracks.map(keyFor));
      for (const key of this.suppressed) if (!present.has(key)) this.suppressed.delete(key);
      for (const track of tracks) {
        if (track.id == null && track.track_id == null) continue;
        const key = keyFor(track);
        let entry = this.byTrack.get(key);
        if (entry) entry.present = true;
        const name = recognitionName(track);
        const state = String(track.identity_status ?? track.match_status ?? track.identity?.status ?? track.state ?? track.status ?? '').toLowerCase();
        const rejected = ['unknown', 'no usable face', 'quality rejected', 'association ambiguous', 'recognition unavailable'].includes(state);
        const personId = track.person_id == null ? null : String(track.person_id);
        if (!name || rejected || (personId && !roster.has(personId)) || this.suppressed.has(key)) continue;
        if (!entry) {
          entry = {key, trackId: String(track.id ?? track.track_id), firstSeen: now};
          this.byTrack.set(key, entry);
          this.entries.unshift(entry);
        }
        const nextName = personId ? roster.get(personId) : name;
        if (entry.name && (entry.personId !== personId || entry.name !== nextName)) {
          entry.previousName = entry.name;
          entry.firstSeen = now;
          entry.snapshotCandidate = null;
        }
        entry.name = nextName;
        entry.personId = personId;
        entry.lastSeen = now;
        entry.present = true;
        entry.revalidated = true;
        entry.decision = identity(track);
        entry.lastDecision = entry.decision.className === 'tracked' ? {label: 'Confirmed', className: 'confirmed'} : entry.decision;
        const score = track.score ?? track.match_score ?? track.similarity;
        entry.score = score != null && Number.isFinite(Number(score)) ? Number(score) : null;
        const samples = list(track.samples).filter((sample) => sample.id && personId &&
          String(sample.person_id) === personId && ['tentative', 'confirmed'].includes(String(sample.state).toLowerCase()) &&
          sample.quality_score != null && Number.isFinite(Number(sample.quality_score)));
        samples.sort((a, b) => Number(b.quality_score) - Number(a.quality_score));
        if (samples.length) entry.snapshotCandidate = {id: String(samples[0].id), quality: Number(samples[0].quality_score)};
      }
      while (this.entries.length > this.limit) {
        const removed = this.entries.pop();
        this.byTrack.delete(removed.key);
        if (present.has(removed.key)) this.suppressed.add(removed.key);
      }
      return this.entries;
    }
  }

  function recognitionSnapshotKey(entry) {
    return JSON.stringify([entry.key, entry.personId ?? entry.name]);
  }

  class RecognitionSnapshots {
    constructor(onChange, options = {}) {
      this.onChange = onChange;
      this.fetcher = options.fetcher || ((...args) => fetch(...args));
      this.createUrl = options.createUrl || ((blob) => URL.createObjectURL(blob));
      this.revokeUrl = options.revokeUrl || ((url) => URL.revokeObjectURL(url));
      this.items = new Map();
    }

    get(entry) {
      return this.items.get(recognitionSnapshotKey(entry));
    }

    release(item) {
      item.controller?.abort();
      if (item.url) this.revokeUrl(item.url);
      item.blob = item.url = null;
    }

    clear() {
      for (const item of this.items.values()) this.release(item);
      this.items.clear();
    }

    sync(entries) {
      const keys = new Set(entries.map(recognitionSnapshotKey));
      for (const [key, item] of this.items) if (!keys.has(key)) {
        this.release(item);
        this.items.delete(key);
      }
      for (const entry of entries) {
        const sample = entry.snapshotCandidate;
        if (!sample) continue;
        const key = recognitionSnapshotKey(entry);
        let item = this.items.get(key);
        if (!item) {
          item = {url: null, blob: null, quality: -Infinity, sampleId: null, failedId: null, pending: false};
          this.items.set(key, item);
        }
        if (item.pending || sample.id === item.failedId || sample.id === item.sampleId ||
            (item.url && sample.quality <= item.quality)) continue;
        item.pending = true;
        item.controller = new AbortController();
        item.promise = this.load(key, item, sample);
      }
    }

    async load(key, item, sample) {
      const timeout = setTimeout(() => item.controller.abort(), 5000);
      try {
        const response = await this.fetcher('/api/samples/' + idPath(sample.id) + '/photo', {
          cache: 'no-store', credentials: 'same-origin', redirect: 'error', signal: item.controller.signal,
        });
        if (!response.ok || Number(response.headers.get('content-length')) > 2000000) throw new Error('Snapshot unavailable');
        const blob = await response.blob();
        if (!blob.size || blob.size > 2000000 || !['image/jpeg', 'image/png', 'image/webp'].includes(blob.type)) throw new Error('Invalid snapshot');
        if (this.items.get(key) !== item || item.controller.signal.aborted) return;
        const url = this.createUrl(blob);
        if (item.url) this.revokeUrl(item.url);
        item.url = url;
        item.blob = blob;
        item.quality = sample.quality;
        item.sampleId = sample.id;
        item.failedId = null;
      } catch {
        if (this.items.get(key) === item) item.failedId = sample.id;
      } finally {
        clearTimeout(timeout);
        item.pending = false;
        if (this.items.get(key) === item) this.onChange();
      }
    }
  }

  class CameraTile {
    constructor(camera) {
      this.id = String(camera.id);
      this.visible = false;
      this.active = false;
      this.epoch = 0;
      this.frameUrl = null;
      this.history = new RecognitionHistory();
      this.snapshots = new RecognitionSnapshots(() => this.renderRecognitions());
      this.element = document.createElement('article');
      this.element.className = 'camera-tile';
      this.element.dataset.id = this.id;
      this.element.innerHTML = '<div class="camera-header"><div class="camera-title"><h3></h3><div class="camera-source"></div></div><div class="camera-controls"><button class="icon-button" data-action="camera-toggle"><i data-lucide="play"></i></button><button class="icon-button danger" data-action="camera-delete" title="Delete camera" aria-label="Delete camera"><i data-lucide="trash-2"></i></button></div></div>' +
        '<div class="camera-frame"><img hidden><div class="frame-placeholder"><i data-lucide="video-off"></i><span>Camera stopped</span></div><span class="frame-state" hidden></span></div>' +
        '<div class="camera-info"><span class="camera-status badge neutral"></span><span class="camera-track-count"></span><span class="camera-fps"></span></div><div class="recognition-heading"><h4>Recent recognitions</h4><span class="recognition-count"></span></div><div class="track-list" role="list" aria-label="Recent recognitions"></div><details class="camera-evidence"><summary>Track evidence / Best-quality crops</summary><div class="evidence-body"></div></details><div class="camera-error notice error" hidden></div>';
      this.img = this.element.querySelector('img');
      this.frame = this.element.querySelector('.camera-frame');
      this.frame.dataset.id = this.id;
      this.placeholder = this.element.querySelector('.frame-placeholder');
      this.frameState = this.element.querySelector('.frame-state');
      $('camera-grid').append(this.element);
      observer.observe(this.frame);
      this.update(camera);
    }

    update(camera) {
      if (this.camera && this.camera.generation !== camera.generation) this.stop(true);
      this.camera = camera;
      this.element.querySelector('h3').textContent = camera.name || 'Unnamed camera';
      this.img.alt = (camera.name || 'Camera') + ' annotated camera frame';
      const source = this.element.querySelector('.camera-source');
      source.textContent = sourceLabel(camera.url_redacted);
      source.title = source.textContent;
      const state = String(camera.state || 'Unknown');
      const badge = this.element.querySelector('.camera-status');
      badge.textContent = state;
      badge.className = 'camera-status badge ' + (state.toLowerCase() === 'running' ? 'good' : state.toLowerCase() === 'error' ? 'bad' : 'neutral');
      this.element.querySelector('.camera-fps').textContent = 'Analysis ' + numeric(camera.metrics?.fps, 1) + (camera.metrics?.fps != null ? ' FPS' : '');
      const tracks = list(camera.tracks);
      const counts = camera.face_counts;
      this.element.querySelector('.camera-track-count').textContent = tracks.length + (tracks.length === 1 ? ' track' : ' tracks') + (counts ? ' / ' + counts.detected + ' faces / ' + counts.usable + ' usable' : '');
      const settingsKey = JSON.stringify(app.data?.settings || {});
      const uptime = app.data?.metrics?.uptime_seconds;
      if ((this.settingsKey !== undefined && this.settingsKey !== settingsKey) ||
          (uptime != null && this.uptime != null && uptime < this.uptime)) {
        this.history.clear();
        this.snapshots.clear();
      }
      this.settingsKey = settingsKey;
      this.uptime = uptime;
      const recent = this.history.update(camera, app.data?.people);
      this.snapshots.sync(recent);
      this.renderRecognitions();
      const evidence = this.element.querySelector('.camera-evidence');
      evidence.querySelector('summary').textContent = 'Track evidence / Best-quality crops / ' + tracks.length + (tracks.length === 1 ? ' track' : ' tracks');
      const evidenceKey = JSON.stringify(tracks.map((track) => [track.id ?? track.track_id, track.name, track.person_name, track.votes, track.observations, track.samples]));
      if (evidenceKey !== this.evidenceKey) {
        this.evidenceKey = evidenceKey;
        evidence.querySelector('.evidence-body').innerHTML = evidenceMarkup(tracks);
        hydrateEvidence(evidence);
      }
      const error = this.element.querySelector('.camera-error');
      error.textContent = camera.error ? redact(camera.error) : '';
      error.hidden = !camera.error;
      this.sync();
    }

    renderRecognitions() {
      const recent = this.history.entries;
      this.element.querySelector('.recognition-count').textContent = recent.length + ' / ' + this.history.limit;
      const trackHtml = recent.length ? recent.map((entry) => recognitionMarkup(entry, this.snapshots.get(entry))).join('') : '<div class="inline-empty">No recognitions yet</div>';
      const trackList = this.element.querySelector('.track-list');
      if (trackList.innerHTML !== trackHtml) trackList.innerHTML = trackHtml;
    }

    sync() {
      const enabled = !disposed && app.connected && app.tab === 'cameras' && !document.hidden && this.visible && running(this.camera);
      if (enabled && !this.active) {
        this.active = true;
        this.poll(this.epoch);
      } else if (!enabled) {
        this.stop(!running(this.camera));
      }
      if (!running(this.camera)) this.placeholder.querySelector('span').textContent = this.camera.error ? 'Camera unavailable' : 'Camera stopped';
      else if (!this.frameUrl) this.placeholder.querySelector('span').textContent = this.visible ? 'Loading camera preview...' : 'Preview loads when in view';
    }

    stop(clear = false) {
      if (this.active || this.controller || this.timer) {
        this.active = false;
        this.epoch++;
        clearTimeout(this.timer);
        this.timer = null;
        this.controller?.abort();
        this.controller = null;
      }
      if (clear && this.frameUrl) {
        this.img.hidden = true;
        this.img.removeAttribute('src');
        URL.revokeObjectURL(this.frameUrl);
        this.frameUrl = null;
      }
      this.placeholder.hidden = !!this.frameUrl;
      this.frameState.hidden = !this.frameUrl;
      if (this.frameUrl) this.frameState.textContent = app.connected ? 'Preview paused' : 'Disconnected - last frame';
    }

    async poll(epoch) {
      if (!this.active || epoch !== this.epoch) return;
      const started = performance.now();
      const controller = new AbortController();
      this.controller = controller;
      const timeout = setTimeout(() => controller.abort(), 5000);
      let nextUrl;
      let failed = false;
      try {
        const response = await fetch('/api/cameras/' + idPath(this.id) + '/frame.jpg?t=' + Date.now(), {cache: 'no-store', credentials: 'same-origin', signal: controller.signal});
        if (!response.ok) throw new Error('Frame unavailable (' + response.status + ')');
        const blob = await response.blob();
        if (!blob.size || (blob.type && !blob.type.startsWith('image/'))) throw new Error('Invalid camera image');
        nextUrl = URL.createObjectURL(blob);
        const decoded = new Image();
        decoded.src = nextUrl;
        await decoded.decode();
        if (!this.active || epoch !== this.epoch) return;
        const previous = this.frameUrl;
        this.img.src = nextUrl;
        this.frameUrl = nextUrl;
        nextUrl = null;
        this.img.hidden = false;
        this.placeholder.hidden = true;
        this.frameState.hidden = true;
        if (previous) URL.revokeObjectURL(previous);
      } catch {
        failed = true;
        if (this.active && epoch === this.epoch) {
          if (!this.frameUrl) this.placeholder.querySelector('span').textContent = 'Frame unavailable. Retrying...';
          else {
            this.frameState.hidden = false;
            this.frameState.textContent = 'Frame unavailable - last frame';
          }
        }
      } finally {
        clearTimeout(timeout);
        if (nextUrl) URL.revokeObjectURL(nextUrl);
        if (this.controller === controller) this.controller = null;
        // Completion-paced requests keep only one frame in flight per visible tile.
        if (this.active && epoch === this.epoch) this.timer = setTimeout(() => this.poll(epoch), failed ? 1500 : Math.max(0, 125 - (performance.now() - started)));
      }
    }

    destroy() {
      observer.unobserve(this.frame);
      this.snapshots.clear();
      this.stop(true);
      this.element.remove();
    }
  }

  const observer = new IntersectionObserver((entries) => {
    for (const entry of entries) {
      const tile = app.cameras.get(entry.target.dataset.id);
      if (tile) { tile.visible = entry.isIntersecting; tile.sync(); }
    }
  }, {threshold: 0.01});

  function admission(track) {
    let value = track.admission_status ?? track.registered_admission ?? track.admission ?? track.admitted;
    if (value && typeof value === 'object') value = value.status ?? value.passed ?? value.accepted;
    if (value === true || ['passed', 'admitted', 'accepted', 'confirmed'].includes(String(value).toLowerCase())) return 'Passed';
    if (value === false || ['failed', 'denied', 'rejected'].includes(String(value).toLowerCase())) return 'Not passed';
    if (['pending', 'tentative'].includes(String(value).toLowerCase())) return 'Pending';
    return 'Not reported';
  }

  function recognitionTime(value) {
    const date = new Date(value);
    return '<time datetime="' + date.toISOString() + '" title="' + esc(date.toLocaleString()) + '">' +
      esc(date.toLocaleTimeString([], {hour: '2-digit', minute: '2-digit', second: '2-digit'})) + '</time>';
  }

  function recognitionMarkup(entry, snapshot) {
    const decision = entry.present && !entry.revalidated ? {label: 'Not revalidated', className: 'neutral'} : entry.present ? entry.decision : entry.lastDecision;
    const presence = entry.present ? entry.revalidated ? 'Current track' : 'No current match' : 'Earlier match';
    const correction = entry.previousName ? '<div class="recognition-correction">Updated from ' + esc(entry.previousName) + '</div>' : '';
    const photo = snapshot?.url ? '<button class="recognition-photo" data-action="recognition-photo" data-entry="' + esc(entry.key) + '" title="Enlarge observed face crop" aria-label="Enlarge observed face crop for ' + esc(entry.name) + '"><img src="' + esc(snapshot.url) + '" alt="Observed camera face crop"><span>Camera crop</span></button>' : '<div class="recognition-photo recognition-photo-empty" role="img" aria-label="Observed snapshot unavailable"><span>' + (snapshot?.pending ? 'Loading' : 'No snapshot') + '</span></div>';
    return '<div class="track-row" role="listitem">' + photo + '<div class="recognition-summary"><div class="track-name">' + esc(entry.name) + '<span class="badge ' + decision.className + '">' + decision.label + '</span><span class="track-id">Track ' + esc(entry.trackId) + '</span></div><span class="track-score">Match ' + numeric(entry.score) + '</span><div class="recognition-times">First seen ' + recognitionTime(entry.firstSeen) + '<span>Last seen ' + recognitionTime(entry.lastSeen) + '</span></div><div class="recognition-presence">' + presence + '</div>' + correction + '</div></div>';
  }

  function evidenceMarkup(tracks) {
    if (!tracks.length) return '<div class="inline-empty">No track evidence</div>';
    return tracks.map((track) => {
      const samples = list(track.samples).slice(0, 5);
      const votes = track.votes == null ? 'Current votes not reported' : 'Current votes ' + track.votes + ' / ' + (track.observations ?? '-');
      const matchAge = track.identity_basis === 'head_track' && track.last_face_match_age_seconds != null ? ' / Last face match ' + numeric(track.last_face_match_age_seconds, 1) + ' s ago' : '';
      return '<section class="evidence-track"><div class="evidence-heading"><strong>' + esc(track.person_name ?? track.name ?? 'Track ' + (track.id ?? track.track_id ?? '-')) + '</strong><span>' + esc(votes) + '</span></div><div class="evidence-status">' + esc(track.state || 'Unknown') + ' / Registered-only admission: ' + admission(track) + matchAge + '</div>' + (samples.length ? '<div class="sample-strip">' + samples.map((sample) => '<figure><button class="sample-open" data-action="sample-open" data-sample="' + esc(sample.id) + '" title="Inspect track sample" aria-label="Inspect track sample"><img data-src="' + esc(imageUrl(sample.url, '/api/samples/' + idPath(sample.id) + '/photo')) + '" alt="Best-quality track crop"></button><figcaption>Quality ' + numeric(sample.quality_score ?? sample.quality?.score) + '</figcaption></figure>').join('') + '</div>' : '<div class="inline-empty">No sample photos</div>') + '</section>';
    }).join('');
  }

  function hydrateEvidence(details) {
    if (!details.open) return;
    for (const image of details.querySelectorAll('img[data-src]')) {
      image.src = image.dataset.src;
      delete image.dataset.src;
    }
  }

  $('camera-grid').addEventListener('toggle', (event) => {
    if (event.target.matches('.camera-evidence')) hydrateEvidence(event.target);
  }, true);

  function renderCameras() {
    const cameras = list(app.data?.cameras);
    const ids = new Set(cameras.map((camera) => String(camera.id)));
    for (const [id, tile] of app.cameras) if (!ids.has(id)) { tile.destroy(); app.cameras.delete(id); }
    for (const camera of cameras) {
      const id = String(camera.id);
      if (app.cameras.has(id)) app.cameras.get(id).update(camera);
      else app.cameras.set(id, new CameraTile(camera));
    }
    $('camera-count').textContent = cameras.length;
    $('camera-summary').textContent = cameras.filter(running).length + ' active / ' + cameras.length + ' configured';
    $('camera-empty').hidden = cameras.length > 0;
    if (!cameras.length) $('camera-empty').innerHTML = '<i data-lucide="camera"></i><h3>No cameras configured</h3>';
    icons($('view-cameras'));
  }

  function photoMarkup(photo, bank) {
    const url = imageUrl(photo.url, '/api/photo/' + idPath(photo.id));
    const score = photo.score ?? photo.quality?.score;
    return '<figure class="photo-item"><div class="photo-visual"><button class="photo-open" data-action="photo-open" data-photo="' + esc(photo.id) + '" title="Inspect ' + esc(bank) + ' photo" aria-label="Inspect ' + esc(bank) + ' photo"><img src="' + esc(url) + '" alt="' + esc(bank) + ' enrollment photo" loading="lazy" decoding="async"></button><button class="icon-button photo-delete danger" data-action="photo-delete" data-photo="' + esc(photo.id) + '" title="Delete photo" aria-label="Delete photo"><i data-lucide="trash-2"></i></button></div><figcaption class="photo-caption"><strong>Score ' + numeric(score) + '</strong>' + esc(describe(photo.source ?? bank)) + '</figcaption></figure>';
  }

  function bankMarkup(photos, label) {
    return '<section><div class="bank-title"><h4>' + label + '</h4><span class="count">' + photos.length + '</span></div>' + (photos.length ? '<div class="photo-bank">' + photos.slice(0, 5).map((photo) => photoMarkup(photo, label)).join('') + '</div>' + (photos.length > 5 ? '<p class="muted">' + (photos.length - 5) + ' additional photos</p>' : '') : '<div class="bank-empty">No ' + label.toLowerCase() + ' photos</div>') + '</section>';
  }

  function renderPeople(force = false) {
    const people = list(app.data?.people);
    const search = $('people-search').value.trim().toLocaleLowerCase();
    const flags = list(app.data?.review?.flags);
    const key = JSON.stringify([people, flags, search, [...app.busy]]);
    if (!force && key === app.peopleKey) return;
    app.peopleKey = key;
    $('people-count').textContent = people.length;
    $('people-summary').textContent = people.length + (people.length === 1 ? ' person' : ' people') + ' / separate registered and live banks';
    const visible = people.filter((person) => String(person.name || '').toLocaleLowerCase().includes(search));
    $('people-list').innerHTML = visible.map((person) => '<article class="person-section" data-person="' + esc(person.id) + '"><div class="person-heading"><div><h3>' + esc(person.name || 'Unnamed person') + '</h3></div><div class="actions"><button data-action="person-enroll" title="Add registered photos"><i data-lucide="image-plus"></i>Add photos</button><button class="icon-button danger" data-action="person-delete" title="Delete person" aria-label="Delete ' + esc(person.name || 'person') + '"><i data-lucide="trash-2"></i></button></div></div><div class="person-banks">' + bankMarkup(list(person.registered), 'Registered') + bankMarkup(list(person.live), 'Live') + '</div></article>').join('');
    $('people-empty').hidden = visible.length > 0;
    if (!visible.length) $('people-empty').innerHTML = '<i data-lucide="users"></i><h3>' + (search ? 'No matching people' : 'No enrolled people') + '</h3>';
    icons($('view-people'));
    for (const section of $('people-list').querySelectorAll('[data-person]')) {
      const flag = flags.find((f) => f.person_id === section.dataset.person);
      const person = people.find((p) => p.id === section.dataset.person);
      const button = document.createElement('button');
      button.dataset.reviewAction = flag ? 'unflag-person' : 'flag-person';
      button.dataset.person = section.dataset.person;
      button.innerHTML = '<i data-lucide="flag"></i>' + (flag ? 'Remove flag' : 'Flag for review');
      button.disabled = !list(person?.registered).length;
      section.querySelector('.person-heading .actions').prepend(button);
      if (flag) {
        const badge = document.createElement('span');
        badge.className = 'badge review-flag';
        badge.textContent = 'Flagged for review';
        section.querySelector('.person-heading h3').append(' ', badge);
      }
    }
    icons($('view-people'));
    updateActions();
  }

  function readout(id, values) {
    const pairs = values && typeof values === 'object' && !Array.isArray(values) ? Object.entries(values) : [['State', values]];
    const html = (pairs.length ? pairs : [['State', 'Not reported']]).map(([key, value]) => '<dt>' + esc(key.replace(/_/g, ' ')) + '</dt><dd>' + esc(describe(value)) + '</dd>').join('');
    if ($(id).innerHTML !== html) $(id).innerHTML = html;
  }

  function renderDiagnostics() {
    const engine = app.data?.engine || {};
    const providers = engine.providers;
    const providerNames = providers && typeof providers === 'object' ? [...new Set(Object.values(providers).flat().map((provider) => describe(provider)))] : providers ? [describe(providers)] : [];
    $('provider-summary').textContent = (engine.face_model ? engine.face_model + ' / ' : '') + (providerNames.length ? providerNames.join(' / ') : 'Providers not reported');
    $('engine-ready').textContent = engine.ready === true ? 'Ready' : engine.ready === false ? 'Not ready' : 'Not reported';
    $('engine-ready').className = 'badge ' + (engine.ready === true ? 'good' : engine.ready === false ? 'warning' : 'neutral');
    readout('engine-details', engine);
    const hardware = app.data?.metrics?.hardware || {};
    const hardwareValue = (value, suffix, digits = 1) => value == null ? 'Not reported' : numeric(value, digits) === 'Not reported' ? describe(value) : numeric(value, digits) + suffix;
    $('hardware-cpu').textContent = hardwareValue(hardware.cpu_percent, '%');
    $('hardware-ram-used').textContent = hardwareValue(hardware.ram_used_gb, ' GB');
    $('hardware-ram-available').textContent = hardwareValue(hardware.ram_available_gb, ' GB');
    $('hardware-process').textContent = hardwareValue(hardware.process_rss_mb, ' MB');
    $('hardware-gpu-util').textContent = hardwareValue(hardware.gpu_percent, '%');
    $('hardware-gpu-memory').textContent = hardwareValue(hardware.gpu_memory_mib, ' MiB', 0);
    $('hardware-gpu-temperature').textContent = hardwareValue(hardware.gpu_temperature_c, ' C');
    $('hardware-gpu').textContent = hardware.gpu ? 'GPU: ' + describe(hardware.gpu) : 'GPU not reported';
    $('hardware-scope').textContent = hardware.scope ? describe(hardware.scope) : 'Scope not reported';
    const {hardware: unusedHardware, ...sessionMetrics} = app.data?.metrics || {};
    readout('metrics-details', sessionMetrics);
    readout('index-details', app.data?.index);
    syncSettings();
  }

  function updateActions() {
    const cameras = list(app.data?.cameras);
    const allBusy = app.busy.has('all-cameras');
    const anyCameraBusy = [...app.busy].some((key) => key.startsWith('camera:'));
    $('start-all').disabled = !app.connected || !cameras.some((camera) => !running(camera)) || allBusy || anyCameraBusy;
    $('stop-all').disabled = !app.connected || !cameras.some(running) || allBusy || anyCameraBusy;
    for (const [id, tile] of app.cameras) {
      const toggle = tile.element.querySelector('[data-action="camera-toggle"]');
      const active = running(tile.camera);
      const action = active ? 'Stop' : 'Start';
      toggle.title = action + ' ' + (tile.camera.name || 'camera');
      toggle.setAttribute('aria-label', toggle.title);
      const iconName = active ? 'square' : 'play';
      if (toggle.dataset.icon !== iconName) { toggle.innerHTML = '<i data-lucide="' + iconName + '"></i>'; toggle.dataset.icon = iconName; }
      for (const button of tile.element.querySelectorAll('button')) button.disabled = !app.connected || allBusy || app.busy.has('camera:' + id);
    }
    for (const section of $('people-list').querySelectorAll('[data-person]')) {
      const id = section.dataset.person;
      for (const button of section.querySelectorAll('button')) button.disabled = !app.connected || app.busy.has('person:' + id) || app.busy.has('photo:' + button.dataset.photo);
    }
    for (const button of $('candidate-list').querySelectorAll('button')) button.disabled = !app.connected || app.busy.has('candidate:' + button.dataset.id);
    syncCandidateReview();
    icons($('camera-grid'));
  }

  async function refreshState() {
    clearTimeout(stateTimer);
    if (stateFlight) return stateFlight;
    stateFlight = (async () => {
      try {
        const data = await api('/api/state');
        if (!data || !Array.isArray(data.cameras) || !Array.isArray(data.people)) throw new Error('Incomplete state response.');
        app.data = data;
        app.connected = true;
        $('connection-state').className = 'status online';
        $('connection-state').innerHTML = '<span class="status-dot"></span>Connected';
        notice('connection-error', '');
        $('last-update').textContent = 'State updated ' + new Date().toLocaleTimeString();
        renderCameras();
        renderPeople();
        renderDiagnostics();
        document.dispatchEvent(new CustomEvent('office:state', {detail: data}));
      } catch (error) {
        app.connected = false;
        $('connection-state').className = 'status offline';
        $('connection-state').innerHTML = '<span class="status-dot"></span>Disconnected';
        notice('connection-error', (app.data ? 'Connection lost. Displayed state may be stale. ' : 'Unable to load evaluation state. ') + (error.message || 'Server unavailable.'));
        if (!app.data) {
          $('camera-summary').textContent = 'State unavailable';
          $('people-summary').textContent = 'State unavailable';
          $('camera-empty').innerHTML = '<i data-lucide="wifi-off"></i><h3>Camera state unavailable</h3>';
          $('people-empty').innerHTML = '<i data-lucide="wifi-off"></i><h3>People state unavailable</h3>';
        }
        for (const tile of app.cameras.values()) tile.sync();
        icons();
      } finally {
        stateFlight = null;
        syncSettings();
        updateActions();
        if (!disposed && !document.hidden) stateTimer = setTimeout(refreshState, app.connected ? 1200 : 3000);
      }
    })();
    return stateFlight;
  }

  async function refreshCandidates() {
    clearTimeout(candidateTimer);
    if (candidateFlight || disposed || app.tab !== 'diagnostics' || document.hidden) return candidateFlight;
    candidateFlight = (async () => {
      try {
        const data = await api('/api/candidates');
        if (!Array.isArray(data.candidates)) throw new Error('Invalid candidate response.');
        app.candidates = data.candidates;
        notice('candidate-error', '');
        renderCandidates();
      } catch (error) {
        notice('candidate-error', (app.candidates.length ? 'Displayed candidates may be stale. ' : '') + error.message);
        $('candidate-empty').textContent = 'Candidates unavailable';
        $('candidate-count').textContent = 'State unavailable';
      } finally {
        candidateFlight = null;
        if (!disposed && app.tab === 'diagnostics' && !document.hidden) candidateTimer = setTimeout(refreshCandidates, 2500);
      }
    })();
    return candidateFlight;
  }

  function renderCandidates() {
    const key = JSON.stringify(app.candidates);
    $('candidate-count').textContent = app.candidates.length + ' pending';
    $('candidate-empty').textContent = 'No candidates awaiting review';
    $('candidate-empty').hidden = app.candidates.length > 0;
    if (key !== app.candidatesKey) {
      app.candidatesKey = key;
      $('candidate-list').innerHTML = app.candidates.map((candidate) => '<article class="candidate"><img src="' + esc(imageUrl(candidate.url)) + '" alt="Candidate evaluation image" loading="lazy"><div><h4>' + esc(candidate.person_name || 'Unassigned candidate') + '</h4><p>' + esc(candidatePolicy(candidate)) + '<br>Quality ' + numeric(candidate.score, 1) + (candidate.evidence?.count != null ? '<br>' + esc(candidate.evidence.count) + ' crops / ' + numeric(candidate.evidence.span_seconds, 1) + ' s' : '') + (candidate.reason ? '<br>' + esc(describe(candidate.reason)) : '') + '</p>' + (candidate.policy === 'single_reference_review' ? '<p class="policy-warning">One reference: human approval required. Similarity is not proof of identity.</p>' : '') + '<div class="actions"><button data-action="candidate-approve" data-id="' + esc(candidate.id) + '"><i data-lucide="check"></i>Approve</button><button data-action="candidate-reject" data-id="' + esc(candidate.id) + '"><i data-lucide="x"></i>Reject</button></div></div></article>').join('');
      icons($('candidate-list'));
    }
    updateActions();
  }

  function candidatePolicy(candidate) {
    if (candidate.policy === 'single_reference_review') return 'One-reference review';
    if (candidate.policy === 'registered_consensus') return 'Registered consensus';
    return 'Manual review';
  }

  function candidateReference(candidate) {
    if (candidate.anchor_reference_url || candidate.anchor_reference_id) return imageUrl(candidate.anchor_reference_url, '/api/photo/' + idPath(candidate.anchor_reference_id));
    if (candidate.review_required || candidate.policy === 'single_reference_review') return '';
    const person = list(app.data?.people).find((item) => candidate.person_id != null && String(item.id) === String(candidate.person_id));
    const reference = list(person?.registered)[0];
    return reference ? imageUrl(reference.url, '/api/photo/' + idPath(reference.id)) : '';
  }

  function syncCandidateReview() {
    const current = app.candidates.find((candidate) => String(candidate.id) === String(candidateReview.candidate?.id));
    const changed = !!candidateReview.candidate && (!current || JSON.stringify(current) !== candidateReview.key);
    const unavailable = !candidateReview.candidate || changed || !app.connected || !candidateReview.referenceLoaded || !candidateReview.cropLoaded;
    const checkbox = $('candidate-identity-verified');
    if (unavailable) checkbox.checked = false;
    checkbox.disabled = candidateReview.busy || unavailable;
    $('candidate-review-confirm').disabled = candidateReview.busy || unavailable || !checkbox.checked;
    $('candidate-review-form').setAttribute('aria-busy', String(candidateReview.busy));
    for (const button of $('candidate-review-dialog').querySelectorAll('[data-close]')) button.disabled = candidateReview.busy;
    if (changed && !candidateReview.busy) notice('candidate-review-error', 'This candidate changed or is no longer available. Close this review and refresh the candidates.');
  }

  function openCandidateReview(candidate) {
    if ($('candidate-review-dialog').open || candidateReview.busy) return;
    candidateReview.candidate = candidate;
    candidateReview.key = JSON.stringify(candidate);
    candidateReview.referenceLoaded = false;
    candidateReview.cropLoaded = false;
    $('candidate-review-form').reset();
    const name = candidate.person_name || 'Unassigned candidate';
    $('candidate-review-name').textContent = name;
    $('candidate-review-policy').textContent = candidatePolicy(candidate);
    $('candidate-review-warning').hidden = candidate.policy !== 'single_reference_review';
    $('candidate-review-quality').textContent = 'Candidate quality ' + numeric(candidate.score, 1);
    const evidence = candidate.evidence || {};
    const values = [
      ['Crops', evidence.count == null ? 'Not reported' : describe(evidence.count)],
      ['Span', evidence.span_seconds == null ? 'Not reported' : numeric(evidence.span_seconds, 1) + ' s'],
      ['Reference similarity', numeric(evidence.reference_similarity)],
      ['Threshold', numeric(evidence.threshold)],
      ['Margin', numeric(evidence.margin)],
      ['Min. quality', numeric(evidence.min_quality, 0)],
      ['Min. width', evidence.min_width == null ? 'Not reported' : numeric(evidence.min_width, 0) + ' px'],
    ];
    $('candidate-review-evidence').innerHTML = values.map(([label, value]) => '<div><dt>' + esc(label) + '</dt><dd>' + esc(value) + '</dd></div>').join('');
    $('candidate-identity-label').textContent = 'I verified this is ' + name + ' and this photo may be added to their live references.';
    notice('candidate-review-error', '');
    for (const [kind, url] of [['reference', candidateReference(candidate)], ['crop', imageUrl(candidate.url)]]) {
      const image = $('candidate-review-' + kind);
      image.hidden = true;
      image.removeAttribute('src');
      image.alt = name + (kind === 'reference' ? ' / registered reference' : ' / candidate crop');
      const status = $('candidate-review-' + kind + '-status');
      status.hidden = false;
      status.textContent = url ? 'Loading ' + (kind === 'reference' ? 'reference...' : 'candidate...') : 'Image unavailable. Approval is blocked.';
      if (url) image.src = url;
    }
    syncCandidateReview();
    $('candidate-review-dialog').showModal();
  }

  for (const kind of ['reference', 'crop']) {
    const image = $('candidate-review-' + kind);
    image.addEventListener('load', () => {
      if (!candidateReview.candidate || !$('candidate-review-dialog').open) return;
      candidateReview[kind + 'Loaded'] = true;
      image.hidden = false;
      $('candidate-review-' + kind + '-status').hidden = true;
      syncCandidateReview();
    });
    image.addEventListener('error', () => {
      candidateReview[kind + 'Loaded'] = false;
      image.hidden = true;
      const status = $('candidate-review-' + kind + '-status');
      status.hidden = false;
      status.textContent = 'Image unavailable. Approval is blocked.';
      syncCandidateReview();
    });
  }
  $('candidate-identity-verified').addEventListener('change', syncCandidateReview);
  $('candidate-review-dialog').addEventListener('cancel', (event) => { if (candidateReview.busy) event.preventDefault(); });
  $('candidate-review-dialog').addEventListener('close', () => {
    candidateReview.candidate = null;
    candidateReview.key = '';
    candidateReview.referenceLoaded = false;
    candidateReview.cropLoaded = false;
    $('candidate-review-form').reset();
    for (const kind of ['reference', 'crop']) $('candidate-review-' + kind).removeAttribute('src');
    syncCandidateReview();
  });
  $('candidate-review-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    syncCandidateReview();
    if ($('candidate-review-confirm').disabled || !$('candidate-review-form').reportValidity()) return;
    const id = candidateReview.candidate.id;
    const key = 'candidate:' + id;
    if (app.busy.has(key)) return;
    candidateReview.busy = true;
    app.busy.add(key);
    notice('candidate-review-error', '');
    updateActions();
    try {
      const result = await api('/api/candidates/' + idPath(id) + '/approve', json('POST', {identity_verified: true}));
      $('candidate-review-dialog').close();
      if (result.accepted === true) toast('Candidate approved.');
      else toast('Candidate not saved: ' + describe(result.reason || 'The reference was not accepted.'), true);
      if (candidateFlight) await candidateFlight;
      await refreshCandidates();
      if (stateFlight) await stateFlight;
      await refreshState();
    } catch (error) {
      $('candidate-identity-verified').checked = false;
      notice('candidate-review-error', error.message || 'Approval failed.');
    } finally {
      candidateReview.busy = false;
      app.busy.delete(key);
      updateActions();
    }
  });

  const settingNames = ['match_threshold', 'match_margin', 'admission_threshold', 'admission_margin', 'adaptation_mode', 'single_reference_mode', 'face_detection_mode', 'analysis_fps', 'recognition_hz', 'face_min_width', 'quality_strictness', 'unknown_capture_enabled', 'flagged_review_enabled', 'review_consent', 'unknown_retention_hours', 'unknown_max_records', 'review_notice_cooldown', 'review_match_threshold', 'review_match_margin'];
  const dirtySettings = new Set();
  let savingSettings = false;
  const settingInput = (name) => $('settings-form').elements.namedItem(name);
  const settingValue = (name) => app.data?.settings?.[name] ?? (name === 'single_reference_mode' ? 'off' : name === 'face_detection_mode' ? 'full_frame' : undefined);

  function syncSettings() {
    const autoLocked = app.data?.metrics?.auto_admission_enabled === false;
    settingInput('adaptation_mode').querySelector('[value="auto"]').disabled = autoLocked;
    $('auto-lock-reason').hidden = !autoLocked;
    $('auto-lock-reason').textContent = autoLocked ? (app.data?.metrics?.auto_admission_reason || 'Auto is locked pending explicit threshold validation.') : '';
    for (const id of ['settings-fields', 'admission-fields', 'capture-fields', 'review-fields']) $(id).disabled = savingSettings || !app.connected;
    for (const name of settingNames) {
      const input = settingInput(name);
      const value = settingValue(name);
      input.disabled = savingSettings || !app.connected || value == null;
      if (!dirtySettings.has(name) && value != null) {
        if (input.type === 'checkbox') input.checked = Boolean(value);
        else input.value = String(value);
      }
    }
    $('single-reference-warning').hidden = settingInput('single_reference_mode').value !== 'review';
    $('quality-strictness').value = settingInput('quality_strictness').value;
    $('quality-strictness').disabled = settingInput('quality_strictness').disabled;
    $('settings-defaults').disabled = savingSettings || !app.connected;
    $('save-settings').disabled = savingSettings || !app.connected || !dirtySettings.size;
    $('settings-status').textContent = savingSettings ? 'Saving...' : dirtySettings.size ? 'Unsaved changes' : app.data ? 'Current settings' : 'Waiting for settings';
  }

  $('settings-form').addEventListener('input', (event) => {
    if (!settingNames.includes(event.target.name)) return;
    const original = settingValue(event.target.name);
    const current = event.target.type === 'checkbox' ? event.target.checked : event.target.value;
    if (String(original) === String(current)) dirtySettings.delete(event.target.name);
    else dirtySettings.add(event.target.name);
    syncSettings();
  });

  $('quality-strictness').addEventListener('input', (event) => {
    const input = settingInput('quality_strictness');
    input.value = event.target.value;
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });

  $('settings-defaults').addEventListener('click', async () => {
    if (savingSettings || !app.connected) return;
    savingSettings = true;
    syncSettings();
    notice('settings-error', '');
    try {
      const preset = await api('/api/settings/defaults');
      if (!preset?.settings || settingNames.some((name) => !Object.hasOwn(preset.settings, name)) ||
          Object.keys(preset.settings).length !== settingNames.length) throw new Error('Incomplete default settings; nothing was reset.');
      const summary = settingNames.map((name) => name.replace(/_/g, ' ') + ': ' + describe(preset.settings[name])).join('\n');
      if (!(await confirmAction('Reset all settings?', 'Restore the Office Demo evaluation preset? These are not production-calibrated thresholds. Cameras and enrolled people will not be removed.\n\n' + summary, 'Reset all', false))) return;
      await api('/api/settings', json('POST', preset.settings));
      dirtySettings.clear();
      if (stateFlight) await stateFlight;
      await refreshState();
      toast('All settings restored to the Office Demo preset.');
    } catch (error) { notice('settings-error', error.message); }
    finally { savingSettings = false; syncSettings(); }
  });

  $('settings-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (savingSettings || !dirtySettings.size || !$('settings-form').reportValidity()) return;
    const patch = {};
    for (const name of dirtySettings) {
      const input = settingInput(name);
      patch[name] = input.type === 'checkbox' ? input.checked : input.tagName === 'SELECT' ? input.value : Number(input.value);
    }
    if (patch.adaptation_mode === 'auto' && !(await confirmAction('Enable automatic adaptation?', 'Change the live photo adaptation mode to Auto for this evaluation?', 'Enable auto', false))) return;
    savingSettings = true;
    syncSettings();
    notice('settings-error', '');
    try {
      await api('/api/settings', json('POST', patch));
      dirtySettings.clear();
      if (stateFlight) await stateFlight;
      await refreshState();
      toast('Settings saved.');
    } catch (error) { notice('settings-error', error.message); }
    finally { savingSettings = false; syncSettings(); }
  });

  function selectTab(name, focus = false) {
    app.tab = name;
    for (const tab of document.querySelectorAll('[data-tab]')) {
      const selected = tab.dataset.tab === name;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      $('view-' + tab.dataset.tab).hidden = !selected;
      if (selected && focus) tab.focus();
    }
    for (const tile of app.cameras.values()) tile.sync();
    clearTimeout(candidateTimer);
    if (name === 'diagnostics') refreshCandidates();
    document.dispatchEvent(new CustomEvent('office:tab', {detail: name}));
  }

  document.querySelector('.view-tabs').addEventListener('click', (event) => {
    const button = event.target.closest('[data-tab]');
    if (button) selectTab(button.dataset.tab);
  });
  document.querySelector('.view-tabs').addEventListener('keydown', (event) => {
    const tabs = [...document.querySelectorAll('[data-tab]')];
    const index = tabs.indexOf(event.target);
    if (index < 0 || !['ArrowLeft', 'ArrowRight', 'Home', 'End'].includes(event.key)) return;
    event.preventDefault();
    const next = event.key === 'Home' ? 0 : event.key === 'End' ? tabs.length - 1 : (index + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length;
    selectTab(tabs[next].dataset.tab, true);
  });

  $('start-all').addEventListener('click', () => mutate('all-cameras', '/api/cameras/start-all', {method: 'POST'}, 'Start requested for all cameras.'));
  $('stop-all').addEventListener('click', () => mutate('all-cameras', '/api/cameras/stop-all', {method: 'POST'}, 'Stop requested for all cameras.'));
  $('add-camera').addEventListener('click', () => { notice('camera-error', ''); $('camera-dialog').showModal(); });
  $('camera-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if (app.busy.has('add-camera') || !$('camera-form').reportValidity()) return;
    const name = $('camera-name').value.trim();
    const url = $('camera-url').value.trim();
    if (!name || !/^rtsps?:\/\//i.test(url)) { notice('camera-error', 'A name and valid rtsp:// or rtsps:// URL are required.'); return; }
    app.busy.add('add-camera');
    $('camera-submit').disabled = true;
    for (const button of $('camera-dialog').querySelectorAll('[data-close]')) button.disabled = true;
    notice('camera-error', '');
    try {
      await api('/api/cameras', json('POST', {name, url, authorized: $('camera-authorized').checked}));
      $('camera-dialog').close();
      toast('Camera added.');
      if (stateFlight) await stateFlight;
      await refreshState();
    } catch (error) { notice('camera-error', error.message); }
    finally {
      app.busy.delete('add-camera');
      $('camera-submit').disabled = false;
      for (const button of $('camera-dialog').querySelectorAll('[data-close]')) button.disabled = false;
    }
  });

  $('camera-grid').addEventListener('click', async (event) => {
    const button = event.target.closest('[data-action]');
    const tile = button?.closest('.camera-tile');
    if (!button || button.disabled || !tile) return;
    const id = tile.dataset.id;
    const camera = app.cameras.get(id)?.camera;
    if (!camera) return;
    if (button.dataset.action === 'recognition-photo') {
      const cameraTile = app.cameras.get(id);
      const entry = cameraTile.history.entries.find((item) => item.key === button.dataset.entry);
      const snapshot = entry && cameraTile.snapshots.get(entry);
      if (!snapshot?.blob) return;
      if (recognitionPhotoUrl) URL.revokeObjectURL(recognitionPhotoUrl);
      recognitionPhotoUrl = URL.createObjectURL(snapshot.blob);
      $('photo-title').textContent = entry.name + ' / observed camera crop';
      $('photo-large').src = recognitionPhotoUrl;
      $('photo-details').textContent = (camera.name || 'Camera') + ' / Track ' + entry.trackId + ' / Crop quality ' + numeric(snapshot.quality, 1) + ' / ' + (entry.present && !entry.revalidated ? 'Not revalidated' : entry.lastDecision.label);
      $('photo-dialog').showModal();
    } else if (button.dataset.action === 'sample-open') {
      const sample = list(camera.tracks).flatMap((track) => list(track.samples)).find((item) => String(item.id) === button.dataset.sample);
      if (!sample) return;
      $('photo-title').textContent = (camera.name || 'Camera') + ' / track evidence';
      $('photo-large').src = imageUrl(sample.url, '/api/samples/' + idPath(sample.id) + '/photo');
      $('photo-details').textContent = 'Best-quality track crop / quality ' + numeric(sample.quality_score ?? sample.quality?.score);
      $('photo-dialog').showModal();
    } else if (button.dataset.action === 'camera-toggle') {
      const action = running(camera) ? 'stop' : 'start';
      await mutate('camera:' + id, '/api/cameras/' + idPath(id) + '/' + action, {method: 'POST'}, action === 'start' ? 'Camera start requested.' : 'Camera stop requested.');
    } else if (await confirmAction('Delete camera?', 'Remove ' + (camera.name || 'this camera') + ' from this evaluation?')) {
      await mutate('camera:' + id, '/api/cameras/' + idPath(id), {method: 'DELETE'}, 'Camera deleted.');
    }
  });

  function findPhoto(id) {
    for (const person of list(app.data?.people)) for (const bank of ['registered', 'live']) {
      const photo = list(person[bank]).find((item) => String(item.id) === id);
      if (photo) return {person, photo, bank};
    }
    return null;
  }

  $('people-search').addEventListener('input', () => renderPeople());
  $('people-list').addEventListener('click', async (event) => {
    const button = event.target.closest('[data-action]');
    if (!button || button.disabled) return;
    const id = button.closest('[data-person]').dataset.person;
    const person = list(app.data?.people).find((item) => String(item.id) === id);
    if (!person) return;
    const action = button.dataset.action;
    if (action === 'person-enroll') openEnrollment(id);
    if (action === 'person-delete' && await confirmAction('Delete person?', 'Delete ' + person.name + ' and all their registered and live photos? This cannot be undone.')) await mutate('person:' + id, '/api/people/' + idPath(id), {method: 'DELETE'}, 'Person deleted.');
    if (action === 'photo-delete') {
      const match = findPhoto(button.dataset.photo);
      if (match && await confirmAction('Delete photo?', 'Delete this ' + match.bank + ' photo for ' + person.name + '? This cannot be undone.')) await mutate('photo:' + button.dataset.photo, '/api/photos/' + idPath(button.dataset.photo), {method: 'DELETE'}, 'Photo deleted.');
    }
    if (action === 'photo-open') {
      const match = findPhoto(button.dataset.photo);
      if (!match) return;
      $('photo-title').textContent = person.name + ' / ' + match.bank;
      $('photo-large').src = imageUrl(match.photo.url, '/api/photo/' + idPath(match.photo.id));
      $('photo-details').textContent = 'Score ' + numeric(match.photo.score ?? match.photo.quality?.score) + ' | Source: ' + describe(match.photo.source ?? match.bank) + (match.photo.width && match.photo.height ? ' | ' + match.photo.width + ' x ' + match.photo.height + ' px' : '');
      $('photo-dialog').showModal();
    }
  });

  $('candidate-list').addEventListener('click', async (event) => {
    const button = event.target.closest('[data-action]');
    if (!button || button.disabled) return;
    const candidate = app.candidates.find((item) => String(item.id) === button.dataset.id);
    if (!candidate) return;
    if (button.dataset.action === 'candidate-approve') { openCandidateReview(candidate); return; }
    if (button.dataset.action !== 'candidate-reject') return;
    const success = await mutate('candidate:' + button.dataset.id, '/api/candidates/' + idPath(button.dataset.id) + '/reject', {method: 'POST'}, 'Candidate rejected.');
    if (success) { if (candidateFlight) await candidateFlight; await refreshCandidates(); }
  });

  const enrollment = {files: [], skipped: 0, objectUrls: [], token: null, accepted: 0, busy: false, revision: 0};
  const mode = () => $('enroll-form').elements.namedItem('enroll_mode').value;

  function clearThumbnails() {
    for (const url of enrollment.objectUrls) URL.revokeObjectURL(url);
    enrollment.objectUrls = [];
    $('selection-thumbnails').replaceChildren();
  }

  function invalidatePreview() {
    enrollment.revision++;
    enrollment.token = null;
    enrollment.accepted = 0;
    $('preview-section').hidden = true;
    $('preview-groups').replaceChildren();
    notice('enroll-error', '');
    notice('enroll-progress', '');
    updateEnrollment();
  }

  function updateEnrollment() {
    const consent = $('enroll-consent').checked;
    const named = mode() !== 'person' || $('enroll-person').value || $('enroll-name').value.trim();
    $('preview-enrollment').disabled = enrollment.busy || !enrollment.files.length || !consent || !named;
    $('preview-enrollment').hidden = !!enrollment.token;
    $('commit-enrollment').hidden = !enrollment.token;
    $('commit-enrollment').disabled = enrollment.busy || !enrollment.token || !enrollment.accepted || !consent;
    $('revise-enrollment').hidden = !enrollment.token;
    $('revise-enrollment').disabled = enrollment.busy;
    for (const input of $('enroll-form').querySelectorAll('input, select')) input.disabled = enrollment.busy;
    $('enroll-name').disabled = enrollment.busy || !!$('enroll-person').value;
    $('clear-selection').disabled = enrollment.busy;
    $('open-webcam').disabled = enrollment.busy;
    for (const button of $('enroll-dialog').querySelectorAll('[data-close]')) button.disabled = enrollment.busy;
  }

  function resetEnrollment() {
    clearThumbnails();
    $('enroll-form').reset();
    enrollment.files = [];
    enrollment.skipped = 0;
    enrollment.busy = false;
    $('selection-summary').textContent = 'No images selected';
    $('clear-selection').hidden = true;
    $('person-inputs').hidden = false;
    $('folder-inputs').hidden = true;
    $('zip-inputs').hidden = true;
    invalidatePreview();
  }

  function openEnrollment(personId = '') {
    resetEnrollment();
    $('enroll-person').innerHTML = '<option value="">New person</option>' + list(app.data?.people).map((person) => '<option value="' + esc(person.id) + '">' + esc(person.name) + '</option>').join('');
    $('enroll-person').value = personId;
    const person = list(app.data?.people).find((item) => String(item.id) === personId);
    $('enroll-name').value = person?.name || '';
    updateEnrollment();
    $('enroll-dialog').showModal();
  }

  function selectFiles(files) {
    clearThumbnails();
    invalidatePreview();
    const all = [...files];
    enrollment.files = all.filter((file) => mode() === 'zip' ? /\.zip$/i.test(file.name) : /\.(jpe?g|png)$/i.test(file.name) && (!file.type || ['image/jpeg', 'image/png'].includes(file.type)));
    enrollment.skipped = all.length - enrollment.files.length;
    $('selection-summary').textContent = enrollment.files.length + (mode() === 'zip' ? ' archive selected' : ' images selected') + (enrollment.skipped ? ' / ' + enrollment.skipped + ' unsupported files excluded' : '');
    $('clear-selection').hidden = !all.length;
    for (const file of enrollment.files.slice(0, 12)) {
      if (mode() === 'zip') continue;
      const url = URL.createObjectURL(file);
      enrollment.objectUrls.push(url);
      const figure = document.createElement('figure');
      const image = document.createElement('img');
      image.src = url;
      image.alt = file.name;
      const caption = document.createElement('figcaption');
      caption.textContent = file.webkitRelativePath || file.name;
      figure.append(image, caption);
      $('selection-thumbnails').append(figure);
    }
    updateEnrollment();
  }

  $('add-person').addEventListener('click', () => openEnrollment());
  document.addEventListener('office:enroll-files', (event) => {
    if (!app.connected || enrollment.busy) return;
    openEnrollment();
    selectFiles(event.detail);
  });
  document.addEventListener('office:refresh', refreshState);
  $('enroll-files').addEventListener('change', (event) => selectFiles(event.target.files));
  $('enroll-folder').addEventListener('change', (event) => selectFiles(event.target.files));
  $('enroll-zip').addEventListener('change', (event) => selectFiles(event.target.files));
  $('clear-selection').addEventListener('click', () => { $('enroll-files').value = ''; $('enroll-folder').value = ''; $('enroll-zip').value = ''; selectFiles([]); });
  $('revise-enrollment').addEventListener('click', invalidatePreview);
  for (const input of document.querySelectorAll('[name="enroll_mode"]')) input.addEventListener('change', () => {
    $('person-inputs').hidden = mode() !== 'person';
    $('folder-inputs').hidden = mode() !== 'folder';
    $('zip-inputs').hidden = mode() !== 'zip';
    $('enroll-files').value = '';
    $('enroll-folder').value = '';
    $('enroll-zip').value = '';
    $('enroll-consent').checked = false;
    selectFiles([]);
  });
  $('enroll-person').addEventListener('change', () => {
    const person = list(app.data?.people).find((item) => String(item.id) === $('enroll-person').value);
    $('enroll-name').value = person?.name || '';
    invalidatePreview();
  });
  $('enroll-name').addEventListener('input', invalidatePreview);
  $('enroll-consent').addEventListener('change', updateEnrollment);

  const webcam = {stream: null, files: [], urls: [], session: 0, pending: false, snapping: false};

  function stopWebcam() {
    webcam.session++;
    for (const track of webcam.stream?.getTracks() || []) track.stop();
    webcam.stream = null;
    webcam.pending = false;
    webcam.snapping = false;
    $('webcam-video').srcObject = null;
    $('webcam-placeholder').hidden = false;
    $('webcam-placeholder').textContent = 'Camera stopped';
    updateWebcam();
  }

  function updateWebcam() {
    $('webcam-start').disabled = webcam.pending;
    $('webcam-start').innerHTML = '<i data-lucide="' + (webcam.stream ? 'square' : 'video') + '"></i>' + (webcam.pending ? 'Starting...' : webcam.stream ? 'Stop camera' : 'Start camera');
    $('webcam-snap').disabled = !webcam.stream || $('webcam-video').readyState < 2 || webcam.files.length >= 5 || webcam.snapping;
    $('webcam-use').disabled = !webcam.files.length || webcam.snapping;
    $('webcam-count').textContent = webcam.files.length + ' / 5 photos';
    icons($('webcam-dialog'));
  }

  function renderWebcamPhotos() {
    for (const url of webcam.urls) URL.revokeObjectURL(url);
    webcam.urls = webcam.files.map((file) => URL.createObjectURL(file));
    $('webcam-photos').innerHTML = webcam.urls.map((url, index) => '<div class="webcam-photo"><img src="' + url + '" alt="Captured photo ' + (index + 1) + '"><button class="icon-button danger" data-remove-capture="' + index + '" title="Remove captured photo" aria-label="Remove captured photo ' + (index + 1) + '"><i data-lucide="x"></i></button></div>').join('');
    updateWebcam();
  }

  $('open-webcam').addEventListener('click', () => {
    webcam.files = [];
    renderWebcamPhotos();
    notice('webcam-error', '');
    $('webcam-dialog').showModal();
  });
  $('webcam-start').addEventListener('click', async () => {
    if (webcam.stream) { stopWebcam(); return; }
    if (webcam.pending) return;
    if (!navigator.mediaDevices?.getUserMedia) { notice('webcam-error', 'Camera access is unavailable. Use localhost or a secure connection.'); return; }
    const session = ++webcam.session;
    webcam.pending = true;
    updateWebcam();
    notice('webcam-error', '');
    try {
      const stream = await navigator.mediaDevices.getUserMedia({video: {width: {ideal: 1280}, height: {ideal: 720}}, audio: false});
      if (!$('webcam-dialog').open || session !== webcam.session) { stream.getTracks().forEach((track) => track.stop()); return; }
      webcam.stream = stream;
      for (const track of stream.getTracks()) track.addEventListener('ended', () => { if (webcam.stream === stream) stopWebcam(); }, {once: true});
      $('webcam-video').srcObject = stream;
      await $('webcam-video').play();
      $('webcam-placeholder').hidden = true;
    } catch (error) {
      if (session === webcam.session) {
        stopWebcam();
        notice('webcam-error', error.name === 'NotAllowedError' ? 'Camera permission was not granted.' : 'Camera unavailable: ' + error.message);
      }
    } finally {
      if (session === webcam.session) { webcam.pending = false; updateWebcam(); }
    }
  });
  $('webcam-video').addEventListener('loadeddata', updateWebcam);
  $('webcam-snap').addEventListener('click', async () => {
    if ($('webcam-snap').disabled) return;
    const session = webcam.session;
    const video = $('webcam-video');
    const canvas = document.createElement('canvas');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    webcam.snapping = true;
    updateWebcam();
    try {
      canvas.getContext('2d').drawImage(video, 0, 0);
      const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/jpeg', .92));
      if (session !== webcam.session || !$('webcam-dialog').open) return;
      if (!blob) throw new Error('No image was captured.');
      webcam.files.push(new File([blob], 'capture-' + Date.now() + '-' + webcam.files.length + '.jpg', {type: 'image/jpeg'}));
      renderWebcamPhotos();
    } catch (error) { notice('webcam-error', error.message); }
    finally { webcam.snapping = false; updateWebcam(); }
  });
  $('webcam-photos').addEventListener('click', (event) => {
    const button = event.target.closest('[data-remove-capture]');
    if (!button || webcam.snapping) return;
    webcam.files.splice(Number(button.dataset.removeCapture), 1);
    renderWebcamPhotos();
  });
  $('webcam-use').addEventListener('click', () => {
    if ($('webcam-use').disabled) return;
    selectFiles([...enrollment.files, ...webcam.files]);
    $('webcam-dialog').close();
  });
  $('webcam-dialog').addEventListener('close', () => {
    stopWebcam();
    webcam.files = [];
    renderWebcamPhotos();
  });

  function renderPreview(data) {
    const groups = list(data.groups);
    enrollment.accepted = groups.reduce((total, group) => total + list(group.items).filter((item) => item.accepted === true).length, 0);
    const total = groups.reduce((sum, group) => sum + list(group.items).length, 0);
    $('preview-summary').textContent = enrollment.accepted + ' accepted / ' + total + ' reviewed';
    $('preview-groups').innerHTML = groups.map((group) => '<section class="preview-group"><h4>' + esc(group.name || 'Unnamed group') + (group.person_id ? ' / existing person' : '') + '</h4><table class="preview-table"><thead><tr><th scope="col">File</th><th scope="col">Decision</th><th scope="col">Quality</th><th scope="col">Reason</th></tr></thead><tbody>' + list(group.items).map((item) => '<tr><td>' + esc(item.filename) + '</td><td><span class="badge ' + (item.accepted === true ? 'good' : 'neutral') + '">' + (item.accepted === true ? 'Accepted' : 'Not accepted') + '</span></td><td>' + numeric(item.quality?.score ?? item.score) + '</td><td>' + esc([item.reason ? describe(item.reason) : '', item.duplicate ? 'Duplicate: ' + describe(item.duplicate) : ''].filter(Boolean).join(' / ') || '-') + '</td></tr>').join('') + '</tbody></table></section>').join('');
    $('preview-section').hidden = false;
  }

  $('enroll-form').addEventListener('submit', async (event) => {
    event.preventDefault();
    if ($('preview-enrollment').disabled || enrollment.busy || enrollment.token) return;
    const files = enrollment.files;
    const revision = enrollment.revision;
    const form = new FormData();
    for (const file of files) form.append('files', file, file.name);
    form.append('paths', JSON.stringify(files.map((file) => file.webkitRelativePath || file.name)));
    form.append('consent', 'true');
    if (mode() === 'person') {
      if ($('enroll-person').value) form.append('person_id', $('enroll-person').value);
      if ($('enroll-name').value.trim()) form.append('name', $('enroll-name').value.trim());
    }
    enrollment.busy = true;
    updateEnrollment();
    notice('enroll-error', '');
    notice('enroll-progress', 'Uploading images for preview...');
    try {
      const data = await api('/api/enroll/preview', {method: 'POST', body: form, timeoutMs: 300000});
      if (revision !== enrollment.revision) return;
      if (!data.token || !Array.isArray(data.groups)) throw new Error('Incomplete preview response. Nothing has been committed.');
      enrollment.token = data.token;
      renderPreview(data);
      notice('enroll-progress', enrollment.accepted ? 'Preview ready. Enrollment has not been committed.' : 'No images accepted. Enrollment has not been committed.');
    } catch (error) { notice('enroll-error', error.message); notice('enroll-progress', ''); }
    finally { enrollment.busy = false; updateEnrollment(); }
  });

  $('commit-enrollment').addEventListener('click', async () => {
    if ($('commit-enrollment').disabled || enrollment.busy || !$('enroll-consent').checked) return;
    enrollment.busy = true;
    updateEnrollment();
    notice('enroll-error', '');
    notice('enroll-progress', 'Committing enrollment...');
    try {
      const result = await api('/api/enroll/commit', {...json('POST', {token: enrollment.token}), timeoutMs: 120000});
      enrollment.busy = false;
      $('enroll-dialog').close();
      toast(typeof result.summary === 'string' ? result.summary : result.accepted != null ? 'Enrollment committed: ' + result.accepted + ' accepted, ' + (result.rejected ?? 0) + ' not added.' : 'Enrollment committed.');
      if (stateFlight) await stateFlight;
      await refreshState();
    } catch (error) { notice('enroll-error', error.message); notice('enroll-progress', ''); }
    finally { enrollment.busy = false; updateEnrollment(); }
  });

  document.addEventListener('click', (event) => {
    const button = event.target.closest('[data-close]');
    if (button && !button.disabled) $(button.dataset.close).close();
  });
  $('enroll-dialog').addEventListener('cancel', (event) => { if (enrollment.busy) event.preventDefault(); });
  $('enroll-dialog').addEventListener('close', () => { if ($('webcam-dialog').open) $('webcam-dialog').close(); stopWebcam(); resetEnrollment(); });
  $('camera-dialog').addEventListener('cancel', (event) => { if (app.busy.has('add-camera')) event.preventDefault(); });
  $('camera-dialog').addEventListener('close', () => { $('camera-form').reset(); notice('camera-error', ''); });
  $('photo-dialog').addEventListener('close', () => {
    $('photo-large').removeAttribute('src');
    if (recognitionPhotoUrl) URL.revokeObjectURL(recognitionPhotoUrl);
    recognitionPhotoUrl = null;
  });
  document.addEventListener('error', (event) => {
    if (event.target === $('photo-large')) {
      event.target.hidden = true;
      $('photo-details').textContent = 'Image unavailable. The sample may have expired or been removed.';
      return;
    }
    if (!(event.target instanceof HTMLImageElement) || !event.target.closest('.photo-open, .sample-open, .candidate')) return;
    const image = event.target;
    const missing = document.createElement('span');
    missing.className = 'image-unavailable';
    missing.textContent = 'Image unavailable';
    image.replaceWith(missing);
  }, true);
  $('photo-large').addEventListener('load', () => { $('photo-large').hidden = false; });

  $('refresh-state').addEventListener('click', async () => {
    $('refresh-state').disabled = true;
    try { await refreshState(); await refreshCandidates(); }
    finally { $('refresh-state').disabled = false; }
  });
  $('refresh-candidates').addEventListener('click', refreshCandidates);
  $('download-report').addEventListener('click', async () => {
    $('download-report').disabled = true;
    try {
      const data = await api('/api/report');
      const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], {type: 'application/json'}));
      const anchor = document.createElement('a');
      anchor.href = url;
      anchor.download = 'office-evaluation-' + new Date().toISOString().replace(/[:.]/g, '-') + '.json';
      document.body.append(anchor);
      anchor.click();
      anchor.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (error) { toast(error.message, true); }
    finally { $('download-report').disabled = false; }
  });

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopWebcam();
    for (const tile of app.cameras.values()) tile.sync();
    clearTimeout(stateTimer);
    clearTimeout(candidateTimer);
    if (!document.hidden) { refreshState(); if (app.tab === 'diagnostics') refreshCandidates(); }
  });
  window.addEventListener('pagehide', () => {
    stopWebcam();
    disposed = true;
    clearTimeout(stateTimer);
    clearTimeout(candidateTimer);
    clearThumbnails();
    for (const tile of app.cameras.values()) { tile.stop(true); tile.snapshots.clear(); }
    if (recognitionPhotoUrl) URL.revokeObjectURL(recognitionPhotoUrl);
    recognitionPhotoUrl = null;
  });
  window.addEventListener('pageshow', (event) => {
    if (!event.persisted) return;
    disposed = false;
    refreshState();
    if (app.tab === 'diagnostics') refreshCandidates();
  });

  icons();
  refreshState();
})();
