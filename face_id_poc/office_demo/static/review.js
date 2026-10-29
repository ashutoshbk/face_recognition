(() => {
  'use strict';
  const $ = (id) => document.getElementById(id);
  const esc = (v) => String(v ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const date = (v) => v ? new Date(v*1000).toLocaleString() : 'No expiry';
  const icon = () => window.lucide?.createIcons();
  const state = {tab:'cameras', data:null, unknowns:{offset:0,items:[],total:0}, notices:{offset:0,items:[],total:0}, busy:false, flight:false};
  const prefix = (kind) => kind === 'unknowns' ? 'unknown' : 'notices';
  let timer;
  async function request(path, method='GET', body) {
    const response = await fetch(path, {method, cache:'no-store', headers:body ? {'Content-Type':'application/json'} : {}, body:body ? JSON.stringify(body) : undefined});
    const value = await response.json();
    if (!response.ok) throw new Error(value.detail || 'Review request failed');
    return value;
  }
  function error(kind, text) {
    const node = $(prefix(kind)+'-error'); node.textContent = text || ''; node.hidden = !text;
  }
  const photos = (row) => row.photos.map(p => '<button class="review-photo" data-review-action="photo" data-photo="'+esc(p.id)+'" title="Inspect crop"><img src="'+esc(p.url)+'" alt="Review crop" loading="lazy"><span>Quality '+Number(p.score).toFixed(1)+' / '+Math.round(p.width)+' px</span></button>').join('');
  function render(kind) {
    const collection = state[kind], key = prefix(kind);
    $(key+'-list').innerHTML = collection.items.map(row => kind === 'unknowns' ?
      '<article class="review-record"><div class="section-heading compact"><div><h3>Unknown track '+esc(row.track_id)+'</h3><p class="muted">'+esc(row.camera_name)+' / '+date(row.first_seen)+'<br>Expires '+date(row.expires)+'</p></div><div class="actions"><button data-review-action="enroll" data-id="'+row.id+'"><i data-lucide="user-plus"></i>Enroll photos</button><button class="icon-button danger" data-review-action="delete-unknown" data-id="'+row.id+'" title="Delete unknown record" aria-label="Delete unknown record"><i data-lucide="trash-2"></i></button></div></div><div class="review-photo-row">'+photos(row)+'</div></article>' :
      '<article class="review-record"><div class="section-heading compact"><div><h3>'+esc(row.person_name)+' <span class="badge review-flag">'+esc(row.status === 'pending' ? 'Needs human review' : 'Reviewed')+'</span></h3><p>'+esc(row.reason)+'</p><p class="muted">'+esc(row.camera_name)+' / '+date(row.created)+'</p></div><div class="actions">'+(row.status === 'pending' ? '<button data-review-action="acknowledge" data-id="'+row.id+'"><i data-lucide="check"></i>Mark reviewed</button>' : '')+'<button class="icon-button danger" data-review-action="delete-notice" data-id="'+row.id+'" title="Delete notice" aria-label="Delete notice"><i data-lucide="trash-2"></i></button></div></div><div class="notice-evidence"><figure><figcaption>Registered reference</figcaption><img src="'+esc(row.reference_url)+'" alt="Registered reference"></figure><figure><figcaption>Fresh observation</figcaption><div class="review-photo-row">'+photos(row)+'</div></figure></div><p class="muted">Similarity '+Number(row.score).toFixed(3)+' / margin '+Number(row.margin).toFixed(3)+' / '+row.votes+' fresh observations</p></article>'
    ).join('') || '<div class="inline-empty">No '+(kind==='unknowns' ? 'unknown faces' : 'review notices')+'</div>';
    $(key+'-page').textContent = collection.total ? (collection.offset+1)+'-'+Math.min(collection.offset+50,collection.total)+' of '+collection.total : '0 records';
    $(key+'-prev').disabled = collection.offset === 0;
    $(key+'-next').disabled = collection.offset+50 >= collection.total;
    icon();
  }
  async function refresh(kind) {
    if (state.flight) return;
    state.flight = true;
    try { state[kind] = await request('/api/review/'+kind+'?offset='+state[kind].offset+'&limit=50'); render(kind); error(kind,''); }
    catch (e) { error(kind,e.message); }
    finally { state.flight=false; }
  }
  function schedule() {
    clearTimeout(timer);
    if (document.hidden || !['unknown','notices'].includes(state.tab)) return;
    timer = setTimeout(async () => { await refresh(state.tab === 'unknown' ? 'unknowns' : 'notices'); schedule(); },2500);
  }
  document.addEventListener('office:state', (event) => {
    state.data = event.detail;
    const r = state.data.review || {};
    $('unknown-count').textContent = r.unknowns || 0;
    $('notice-count').textContent = r.pending_notices || 0;
    $('unknown-summary').textContent = (state.data.settings.unknown_capture_enabled ? 'Collection on' : 'Collection off')+' / '+(r.unknowns||0)+' records / retention '+state.data.settings.unknown_retention_hours+' h';
    $('notices-summary').textContent = (state.data.settings.flagged_review_enabled ? 'Flagged-person review on' : 'Flagged-person review off')+' / '+(r.pending_notices||0)+' awaiting review';
    $('review-storage').textContent = ((r.storage_bytes||0)/1e6).toFixed(1)+' / 250 MB used; '+(r.queue_size||0)+' queued'+(r.error ? ' / '+r.error : '');
  });
  document.addEventListener('office:tab', async (event) => {
    state.tab = event.detail;
    if (['unknown','notices'].includes(state.tab)) await refresh(state.tab==='unknown' ? 'unknowns':'notices');
    schedule();
  });
  for (const kind of ['unknowns','notices']) {
    const key = prefix(kind);
    $(key+'-prev').addEventListener('click', () => { state[kind].offset=Math.max(0,state[kind].offset-50); refresh(kind); });
    $(key+'-next').addEventListener('click', () => { state[kind].offset+=50; refresh(kind); });
    document.querySelector('[data-review-refresh="'+kind+'"]').addEventListener('click', () => refresh(kind));
  }
  document.addEventListener('click', async (event) => {
    const button = event.target.closest('[data-review-action]');
    if (!button || state.busy) return;
    const action = button.dataset.reviewAction, id = button.dataset.id;
    if (action === 'flag-person') {
      const person = state.data.people.find(p=>p.id===button.dataset.person);
      $('flag-form').reset(); $('flag-form').dataset.person=person.id; $('flag-person-name').textContent=person.name;
      $('flag-error').hidden=true; $('flag-dialog').showModal(); return;
    }
    state.busy=true; button.disabled=true;
    try {
      if (action === 'unflag-person') await request('/api/people/'+encodeURIComponent(button.dataset.person)+'/flag','DELETE');
      if (action === 'delete-unknown' || action === 'delete-notice') {
        const kind=action==='delete-unknown'?'unknowns':'notices';
        if (!window.confirm('Delete this record and its saved crops?')) return;
        await request('/api/review/'+kind+'/'+id,'DELETE'); await refresh(kind);
      }
      if (action === 'acknowledge') { await request('/api/review/notices/'+id+'/acknowledge','POST'); await refresh('notices'); }
      if (action === 'enroll') {
        const row=state.unknowns.items.find(r=>r.id===id), files=[];
        for (const [i,photo] of row.photos.entries()) {
          const response=await fetch(photo.url,{cache:'no-store'});
          if (!response.ok) throw new Error('Crop expired; refresh the record');
          files.push(new File([await response.blob()],'review-'+i+'.jpg',{type:'image/jpeg'}));
        }
        document.dispatchEvent(new CustomEvent('office:enroll-files',{detail:files}));
      }
      if (action === 'photo') {
        $('photo-title').textContent='Private review crop'; $('photo-large').src='/api/review/photos/'+button.dataset.photo;
        $('photo-details').textContent='Not an enrolled identity. Subject to retention.'; $('photo-dialog').showModal();
      }
      document.dispatchEvent(new Event('office:refresh'));
    } catch(e) { error(state.tab==='notices'?'notices':'unknowns',e.message); }
    finally { state.busy=false; button.disabled=false; }
  });
  $('flag-form').addEventListener('submit', async (event) => {
    event.preventDefault(); if (state.busy || !$('flag-form').reportValidity()) return;
    state.busy=true; $('flag-save').disabled=true;
    try {
      const hours=$('flag-expiry').value;
      await request('/api/people/'+encodeURIComponent($('flag-form').dataset.person)+'/flag','POST',{
        reason:$('flag-reason').value, expires_hours:hours ? Number(hours):null, participant_consented:$('flag-consent').checked});
      $('flag-dialog').close(); document.dispatchEvent(new Event('office:refresh'));
    } catch(e) { $('flag-error').textContent=e.message; $('flag-error').hidden=false; }
    finally { state.busy=false; $('flag-save').disabled=false; }
  });
  document.addEventListener('visibilitychange',schedule);
  window.addEventListener('pagehide',()=>clearTimeout(timer));
})();
