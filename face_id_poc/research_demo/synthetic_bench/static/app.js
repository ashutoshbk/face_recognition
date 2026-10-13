const $ = (id) => document.getElementById(id);
let selected = '', loaded = '', busy = false;
const number = (v) => Number(v || 0).toLocaleString();
async function api(path, options) {
  const response = await fetch(path, options);
  if (!response.ok) { const body = await response.json(); throw Error(body.detail || response.statusText); }
  return response.json();
}
function metrics(result) {
  $('count').textContent = number(result.gallery_vectors);
  for (const id of ['p50','p95']) $(''+id).textContent = result.search_latency_ms[id].toFixed(2) + ' ms';
  $('recall').textContent = (100*result.ann_recall_at10).toFixed(1)+'%';
  $('top1').textContent = (100*result.synthetic_closed_set_top1).toFixed(1)+'%';
  $('embedding').textContent = `Median ${result.embedding_ms.p50.toFixed(1)} ms; p95 ${result.embedding_ms.p95.toFixed(1)} ms, parallel CPU workers`;
  $('queries').textContent = number(result.queries);
  $('gallery-foot').textContent = result.indexed_vectors ? `${number(result.indexed_vectors)} HNSW indexed` : 'Small pilot · exact scan';
  const end = result.image_to_results_latency_ms;
  $('end-to-end').textContent = `Median ${end.p50.toFixed(1)} ms; p95 ${end.p95.toFixed(1)} ms, ${end.count} freshly encoded image queries`;
}
async function loadExamples(name) {
  const result = await api(`/api/runs/${name}/examples`);
  if (selected !== name) return;
  loaded = name;
  metrics(result);
  $('probe').replaceChildren(...result.examples.map((e) => new Option(`Synthetic ${e.identity}${e.correct_top1?'':' · top-1 mismatch'}`,e.identity)));
  $('probe').disabled = $('search').disabled = false;
  $('export').href = `/api/runs/${name}/result`; $('export').hidden = false;
  await query();
}
function resetResults() {
  loaded = ''; $('export').hidden = true;
  for (const id of ['p50','p95','recall','top1','embedding','queries','end-to-end']) $(id).textContent = '—';
  $('probe').replaceChildren(); $('probe').disabled = $('search').disabled = true;
  $('query-image').hidden = true; $('query-label').textContent = 'Awaiting completed gallery';
  $('query-time').textContent = ''; $('matches').replaceChildren();
}
async function query() {
  if (busy || !loaded) return;
  busy=true; $('search').disabled=true;
  const name=selected, identity=$('probe').value;
  try {
    const result = await api(`/api/runs/${name}/probe/${identity}`,{method:'POST'});
    if (name!==selected) return;
    $('query-image').src=`/api/runs/${name}/image/probe/${identity}`; $('query-image').hidden=false;
    $('query-label').textContent=`Synthetic ${identity}`;
    $('query-time').textContent=`${result.search_ms.toFixed(2)} ms search · cached embedding`;
    $('matches').replaceChildren(...result.matches.map((m,i)=>{
      const node=document.createElement('article');node.className='match'+(m.identity===result.identity?' correct':'');
      const img=document.createElement('img');img.src=`/api/runs/${name}/image/gallery/${m.identity}`;img.alt=`Synthetic gallery identity ${m.identity}`;
      const title=document.createElement('strong');title.textContent=`${i+1}. Synthetic ${m.identity}`;
      const score=document.createElement('span');score.textContent=`Cosine ${m.cosine.toFixed(3)}${m.identity===result.identity?' · same synthetic ID':''}`;
      node.append(img,title,score);return node;
    }));
  } catch(error) {$('error').textContent=error.message;}
  finally{busy=false;$('search').disabled=!loaded;}
}
async function refresh() {
  try {
    const value=await api('/api/status');
    document.querySelector('header a').href=value.webcam_url+'/?mode=self';
    const names=value.runs.map(r=>r.name).sort();
    if (!selected || !names.includes(selected)) selected=names.includes('all_identities')?'all_identities':names[0]||'';
    if (JSON.stringify([...$('run').options].map(o=>o.value))!==JSON.stringify(names)) $('run').replaceChildren(...names.map(n=>new Option(n.replaceAll('_',' '),n)));
    $('run').value=selected;
    const active=value.runs.some(r=>['preparing','embedding','loading','indexing','querying'].includes(r.status));
    $('resume').disabled=active;$('pause').disabled=!active;
    const state=value.runs.find(r=>r.name===selected);
    if(!state){$('status').textContent='Ready';return;}
    $('status').textContent=state.status;
    $('count').textContent=number(state.gallery_vectors);
    $('gallery-label').textContent=state.status==='complete'?'Qdrant gallery':'Prepared vectors';
    if(state.status!=='complete') $('gallery-foot').textContent=state.status==='embedding'?'Qdrant upload pending':'Synthetic identities';
    $('progress').value=state.status==='complete'?100:state.status==='embedding'?80*(state.completed||0)/state.total:state.status==='indexing'?90:state.status==='querying'?95:0;
    $('build-detail').textContent=`${number(state.completed)} / ${number(state.total)} samples processed · ${number(state.gallery_vectors)} gallery vectors · ${number(state.rejected)} rejected${state.eta_seconds>0?' · approximately '+Math.ceil(state.eta_seconds/60)+' minutes remaining for encoding':''}`;
    $('error').textContent=state.error||'';
    if(state.result_available && loaded!==selected) await loadExamples(selected);
  }catch(error){$('error').textContent=error.message;}
}
$('run').onchange=()=>{selected=$('run').value;resetResults();refresh();};
$('probe').onchange=query; $('search').onclick=query;
for (const [id,path] of [['resume','start'],['pause','pause']]) $(id).onclick=async()=>{try{await api(`/api/build/${path}`,{method:'POST'});await refresh();}catch(error){$('error').textContent=error.message;}};
refresh(); setInterval(refresh,3000);
