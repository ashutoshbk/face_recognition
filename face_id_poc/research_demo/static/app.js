const $ = (id) => document.getElementById(id);
const state = {
  session: null,
  stream: null,
  running: false,
  enrolling: false,
  generation: 0,
  last: null,
  testers: [],
  events: [],
  frameTimes: [],
  recognitionTimes: [],
  lastEvent: "",
  benchStamp: null,
  frameSerial: 0,
  lastRecognitionSequence: null,
  recognitionImages: new Map(),
  previewRequest: null,
  previewToken: 0,
  previewRetryAfter: 0,
};
const icons = () => window.lucide?.createIcons();
const selfMode = () => $("live-mode").value === "self";
const modeQuery = () => `mode=${selfMode() ? "self" : "verify"}`;
async function refreshSelfStatus() {
  if (!selfMode()) return;
  const info = await api('/api/self-test');
  $('self-gallery-count').textContent = info.ready ? `${info.synthetic_count.toLocaleString()} synthetic + ${info.self_count} self` : 'Qdrant unavailable';
  if (!info.ready) $('self-search-note').textContent = info.error;
}
function updateMode() {
  $('self-search-section').hidden = !selfMode();
  $('tester-heading').textContent = selfMode() ? 'My enrolled profile' : 'Selected tester';
  $('comparison-label').textContent = selfMode() ? 'Latest Qdrant lookup' : 'Selected-template comparison';
  $('mode-scope').textContent = selfMode() ? 'Local self-test · synthetic distractors' : 'Consenting testers · 1:1 verification';
  $('consent-label').textContent = selfMode() ? 'I am enrolling myself and consent to this local research test.' : 'This tester consents to local face enrollment.';
  $('tester').disabled = selfMode();
  $('analysis-fps-label').textContent=selfMode()?'detection / tracking FPS':'analysis FPS';
  $('threshold-assessment').hidden = !selfMode();
  $('recognition-summary').hidden = !selfMode();
  $('enroll-photo').hidden = !selfMode();
}
async function assessThreshold() {
  if(!selfMode() || !state.testers.length) return;
  $('assess-threshold').disabled=true;$('apply-threshold').disabled=true;
  $('threshold-advice').textContent='Comparing enrolled template with synthetic negatives...';
  const id=state.testers[0].id;
  try {
    const result=await api('/api/self-test/threshold-analysis',{method:'POST'});
    if(!selfMode() || state.testers[0]?.id!==id) return;
    state.thresholdSuggestion=result.threshold;
    $('threshold-advice').textContent=`Provisional ${result.threshold.toFixed(2)} · ${result.validation_exceedances} / ${result.validation_count.toLocaleString()} held-out synthetic scores exceeded it. Real genuine-match testing remains pending.`;
    $('apply-threshold').disabled=false;
  }catch(e){$('threshold-advice').textContent=e.message;}
  finally{$('assess-threshold').disabled=!state.testers.length;}
}
function toast(message, error = false) {
  const el = $("toast");
  el.textContent = message;
  el.classList.toggle("error", error);
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 6000);
}
async function api(path, options = {}) {
  const response = await fetch(path, {
    signal: AbortSignal.timeout(15000),
    ...options,
  });
  if (!response.ok) {
    let error;
    try {
      error = (await response.json()).detail;
    } catch {
      error = response.statusText;
    }
    throw new Error(typeof error === "string" ? error : JSON.stringify(error));
  }
  return response.json();
}
const json = (method, value) => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(value),
});
function settings() {
  return {
    selected: $("tester").value || null,
    minimum_width: +$("min-width").value,
    minimum_sharpness: +$("min-sharpness").value,
    threshold: +$("threshold").value,
  };
}
async function saveSettings() {
  if (!state.session) return;
  const generation = state.running ? ++state.generation : state.generation;
  if (state.running) clearView();
  try {
    await api(
      `/api/sessions/${state.session}/settings`,
      json("PUT", settings()),
    );
  } finally {
    if (state.running && generation === state.generation) loop(generation);
  }
}
function emptySamples() {
  state.sampleKey=null;
  const container = $("samples");
  container.replaceChildren();
  for (let i = 0; i < 5; i++) {
    const el = document.createElement("div");
    el.className = "sample";
    el.innerHTML =
      '<div class="sample-placeholder"><i data-lucide="scan-face"></i></div><div class="sample-meta"><span>Sample ' +
      (i + 1) +
      "</span><span>--</span></div>";
    container.append(el);
  }
  icons();
}
function renderSamples(samples) {
  if (!samples.length) return emptySamples();
  const key=samples.map(s=>s.image).join('|');
  if(state.sampleKey===key) return;
  state.sampleKey=key;
  const container = $("samples");
  container.replaceChildren();
  for (let i = 0; i < 5; i++) {
    const el = document.createElement("div");
    el.className = "sample" + (i === 0 ? " best" : "");
    if (samples[i]) {
      const sample = samples[i];
      const img = document.createElement("img");
      img.src = sample.image;
      img.alt = `Accepted sample ${i + 1}`;
      el.append(img);
      const meta = document.createElement("div");
      meta.className = "sample-meta";
      meta.textContent = `${i === 0 ? "BEST" : "#" + (i + 1)} · Q ${sample.quality}`;
      el.append(meta);
    } else {
      el.innerHTML =
        '<div class="sample-placeholder"><i data-lucide="scan-face"></i></div><div class="sample-meta">Awaiting sample</div>';
    }
    container.append(el);
  }
  icons();
}
function clearView() {
  state.last = null;
  state.frameTimes = [];
  state.recognitionTimes=[];
  state.sampleKey=null;state.lastRecognitionSequence=null;
  state.previewToken++;state.previewRequest=null;state.previewRetryAfter=0;
  if(!state.running)state.recognitionImages.clear();
  $("fps").textContent = "0.0";
  $("processing").textContent = "--";
  $("track-id").textContent = "No active track";
  $("gate-state").textContent = "Awaiting camera";
  $("sample-count").textContent = "0 / 5 accepted";
  $('capture-status').textContent='Waiting for capture';
  $('capture-reset').textContent='No samples collected';
  $("track-count").textContent = "0 heads / 0 faces";
  $("head-status").textContent = "No head observed";
  $("head-age").textContent = "0.0 s";
  $("turnaway-status").textContent = "Not started";
  $("turnaway-start").disabled = true;
  $("turnaway-export").disabled = true;
  [
    "quality-score",
    "face-width",
    "sharpness",
    "brightness",
    "alignment",
    "similarity",
    "detection-time",
    "embedding-time",
    "comparison-time",
    "roundtrip-time",
    "face-sample-age",
    "recognition-rate",
  ].forEach((id) => ($(id).textContent = "--"));
  $("quality-bar").value = 0;
  $("verdict").className = "verdict";
  $("verdict-label").textContent = $("tester").value
    ? "Awaiting camera"
    : "No tester selected";
  $("vote-count").textContent = "0 / 5";
  renderVotes([]);
  emptySamples();
  $('self-candidates').replaceChildren();
  state.selfCandidateKey=null;
  for (const id of ['self-score','self-lead','self-search-time']) $(id).textContent='--';
  for (const id of ['recognition-encode-ms','recognition-lookup-ms','recognition-total-ms','recognition-candidates']) $(id).textContent='--';
  $('recognition-status').textContent='Waiting';
  $('self-search-note').textContent='No query yet';
  updateEnroll();
}
function renderVotes(scores, passes) {
  $("votes").replaceChildren();
  scores.forEach((score, index) => {
    const el = document.createElement("span");
    el.className = "vote " + ((passes ? passes[index] : score >= +$("threshold").value) ? "pass" : "fail");
    el.title = score.toFixed(3);
    $("votes").append(el);
  });
  for (let i = scores.length; i < 5; i++) {
    const el = document.createElement("span");
    el.className = "vote";
    $("votes").append(el);
  }
}
function renderResult(result, roundtrip) {
  state.last = result;
  const now = performance.now();
  state.frameTimes.push(now);
  state.frameTimes = state.frameTimes.filter((t) => now - t < 3000);
  const span = now - state.frameTimes[0];
  if(result.recognition){
    const sequence=result.recognition.completed;
    if(result.self_search && sequence && sequence!==state.lastRecognitionSequence){state.recognitionTimes.push(now);state.lastRecognitionSequence=sequence;}
  } else if(result.timing.embedding_ms>0)state.recognitionTimes.push(now);
  state.recognitionTimes=state.recognitionTimes.filter(t=>now-t<3000);
  $('recognition-rate').textContent=`${span>500?(state.recognitionTimes.length*1000/span).toFixed(1):'--'} / s`;
  $("fps").textContent = (
    span > 0 ? ((state.frameTimes.length - 1) * 1000) / span : 0
  ).toFixed(1);
  $('preview-fps').textContent=$('fps').textContent;
  $("processing").textContent = result.timing.total_ms.toFixed(1);
  $("track-count").textContent =
    `${result.head_count || 0} heads / ${result.face_count} faces`;
  $("head-status").textContent = result.head_status || "No head observed";
  $("head-age").textContent = `${(result.head_age_seconds || 0).toFixed(1)} s`;
  const check = result.turnaway || { status: "idle", message: "Not started" };
  $("turnaway-status").textContent =
    `${check.message}${check.face_unavailable_seconds ? " · " + check.face_unavailable_seconds + " s without usable face" : ""}`;
  $("turnaway-status").dataset.status = check.status;
  $("turnaway-start").disabled =
    !result.active_id ||
    !result.quality?.accepted ||
    ["waiting_for_face_absence", "waiting_for_face_return"].includes(
      check.status,
    );
  $("turnaway-export").disabled = ![
    "passed",
    "failed",
    "incomplete",
    "interrupted",
  ].includes(check.status);
  $("track-id").textContent = result.active_id || "No active track";
  $("resolution").textContent = `${result.width} × ${result.height}`;
  $('face-sample-age').textContent=result.self_search ? `${result.self_search.sample_age_ms} ms · recognition crop` : '--';
  const q = result.quality;
  $("gate-state").textContent = q
    ? q.accepted
      ? "Quality accepted"
      : q.reasons.join(" · ")
    : result.verification.state;
  $("quality-score").textContent = q ? q.score.toFixed(0) : "--";
  $("quality-bar").value = q?.score || 0;
  $("face-width").textContent = q ? `${q.width} px` : "--";
  $("sharpness").textContent = q ? q.sharpness : "--";
  $("brightness").textContent = q ? q.brightness : "--";
  $("alignment").textContent = q ? q.frontal.toFixed(2) : "--";
  $("sample-count").textContent = `${result.samples.length} / 5 accepted`;
  if(result.capture){
    const c=result.capture;
    $('capture-status').textContent=`${c.state} · ${c.samples} samples · ${c.required} required`;
    $('capture-reset').textContent=c.pause_reason || (c.reset_count?`Last reset: ${c.last_reset}`:`${c.reset_count} capture resets`);
  }
  renderSamples(result.samples);
  const v = result.verification;
  $("verdict-label").textContent = v.state === 'Match' && result.self_search ? `Match · ${result.self_search.self_label}` : v.state;
  $("verdict").className =
    "verdict" +
    (v.state === "Match"
      ? " match"
      : v.state === "No match"
        ? " no-match"
        : "");
  $("vote-count").textContent = `${v.votes} / ${v.count}`;
  renderVotes(v.scores, v.vote_passes);
  if (selfMode()) renderSelfSearch(result);
  $("similarity").textContent = v.scores.length
    ? v.scores.at(-1).toFixed(3)
    : "--";
  $("detection-time").textContent =
    `${result.timing.detection_ms.toFixed(1)} ms`;
  const encoding=result.recognition?.latest_encoding_ms ?? result.timing.embedding_ms;
  $("embedding-time").textContent = encoding ? `${encoding.toFixed(1)} ms${result.recognition?' · background':''}` : "Not sampled";
  if(result.recognition){
    const r=result.recognition;
    $('recognition-status').textContent=r.state;
    for(const [id,value] of [['recognition-encode-ms',r.latest_encoding_ms],['recognition-lookup-ms',r.latest_search_ms],['recognition-total-ms',r.latest_total_ms]])$(id).textContent=value==null?'--':`${value.toFixed(1)} ms`;
    $('recognition-gallery-size').textContent=r.gallery_size==null?'--':r.gallery_size.toLocaleString();
    $('recognition-candidates').textContent=r.candidates_returned||'--';
  }
  $("comparison-time").textContent = result.timing.comparison_ms
    ? `${result.timing.comparison_ms.toFixed(3)} ms`
    : "--";
  $("roundtrip-time").textContent = `${roundtrip.toFixed(1)} ms`;
  if (v.state !== state.lastEvent) {
    state.lastEvent = v.state;
    state.events.unshift({
      time: new Date().toLocaleTimeString(),
      id: result.active_id || "--",
      status: v.state,
    });
    state.events = state.events.slice(0, 40);
    $("events").replaceChildren();
    for (const event of state.events) {
      const row = document.createElement("div");
      row.className = "event";
      for (const [tag, text] of [
        ["time", event.time],
        ["span", event.id],
        ["span", event.status],
      ]) {
        const child = document.createElement(tag);
        child.textContent = text;
        row.append(child);
      }
      $("events").append(row);
    }
  }
  updateEnroll();
}
function renderSelfSearch(result) {
  const search=result.self_search;
  if (!search) {
    $('self-candidates').replaceChildren();state.selfCandidateKey=null;
    for (const id of ['self-score','self-lead','self-search-time']) $(id).textContent='--';
    $('self-search-note').textContent=result.search_error || result.verification.state;
    return;
  }
  $('self-score').textContent=search.self_score == null ? 'Not enrolled' : search.self_score.toFixed(3);
  $('self-lead').textContent=search.lead == null ? '--' : search.lead.toFixed(3);
  $('self-search-time').textContent=`${search.search_ms.toFixed(2)} ms`;
  $('self-gallery-count').textContent=`Gallery size ${search.total.toLocaleString()}`;
  $('self-search-note').textContent=search.self_score == null ? 'No self profile enrolled' : search.self_is_first ? (search.self_score >= +$('threshold').value ? 'Self profile ranks first and passes the threshold' : 'Self profile ranks first but is below the threshold') : 'A synthetic candidate ranks higher · no self match';
  const key=search.candidates.map(c=>c.id+':'+c.kind).join('|');
  if (state.selfCandidateKey===key) {
    [...$('self-candidates').children].forEach((item,i)=>{const c=search.candidates[i];item.querySelector('span').textContent=`${c.kind==='self'?'Self':'Synthetic'} · cosine ${c.score.toFixed(3)}`;});
    applyCandidateImages();requestRecognitionPreviews(result);
    return;
  }
  $('self-candidates').replaceChildren();state.selfCandidateKey=key;
  for (const [index,c] of search.candidates.entries()) {
    const item=document.createElement('article');item.className='self-candidate'+(c.kind==='self'?' own':'');
    item.dataset.previewKey=`${c.kind}:${c.id}`;
    const media=document.createElement('div');media.className='candidate-image';
    const img=document.createElement('img');img.decoding='async';img.style.visibility='hidden';img.alt=c.kind==='self'?'My enrolled reference':c.label;
    img.onload=()=>{img.style.visibility='visible';};media.append(img);
    const label=document.createElement('strong');label.textContent=`${index+1}. ${c.label}`;
    const score=document.createElement('span');score.textContent=`${c.kind==='self'?'Self':'Synthetic'} · cosine ${c.score.toFixed(3)}`;
    item.append(media,label,score);$('self-candidates').append(item);
  }
  applyCandidateImages();requestRecognitionPreviews(result);
}
function applyCandidateImages(){
  for(const item of $('self-candidates').children){
    const key=item.dataset.previewKey,data=state.recognitionImages.get(key),img=item.querySelector('img');
    if(data && img.dataset.loadedKey!==key){img.dataset.loadedKey=key;img.src=data;}
  }
}
async function requestRecognitionPreviews(result){
  if(!state.running || state.previewRequest || performance.now()<state.previewRetryAfter || !result.self_search)return;
  if(result.self_search.candidates.every(c=>state.recognitionImages.has(`${c.kind}:${c.id}`)))return;
  const session=state.session,epoch=result.recognition.epoch,sequence=result.recognition.completed;
  const token=++state.previewToken;state.previewRequest=token;
  try{
    const response=await api(`/api/sessions/${session}/recognition-previews?epoch=${epoch}&sequence=${sequence}`);
    if(!state.running || state.session!==session || token!==state.previewToken || state.last?.recognition?.epoch!==epoch || state.last?.recognition?.completed!==sequence)return;
    for(const [key,value] of Object.entries(response.images)){
      state.recognitionImages.set(key,value);
      while(state.recognitionImages.size>256)state.recognitionImages.delete(state.recognitionImages.keys().next().value);
    }
    applyCandidateImages();
  }catch(error){/* A changed recognition result is retried without interrupting video. */}
  finally{if(state.previewRequest===token)state.previewRequest=null;state.previewRetryAfter=performance.now()+300;}
}
function updateEnroll() {
  $("enroll").disabled =
    state.enrolling ||
    !state.running ||
    !state.last?.enrollment_ready ||
    !$("consent").checked ||
    !$("tester-name").value.trim() || (selfMode() && state.testers.length>0);
  $('enroll-photo').disabled=state.enrolling || state.running || !selfMode() || state.testers.length>0 || !$('consent').checked || !$('tester-name').value.trim();
  $('enroll-photo').title=state.running?'Stop the camera before photo enrollment':'Enroll your own JPEG or PNG photo (up to 4 MB)';
}
async function cameras() {
  const devices = await navigator.mediaDevices.enumerateDevices();
  const previous = $("camera").value;
  $("camera").replaceChildren(new Option("Default camera", ""));
  devices
    .filter((d) => d.kind === "videoinput")
    .forEach((device, i) =>
      $("camera").append(
        new Option(device.label || `Camera ${i + 1}`, device.deviceId),
      ),
    );
  if ([...$("camera").options].some((o) => o.value === previous))
    $("camera").value = previous;
}
async function startCamera() {
  $("camera-toggle").disabled = true;
  try {
    if($('input-source').value==='rtsp'){
      if(!$('rtsp-consent').checked)throw Error('Confirm authorization for this controlled test stream');
      state.session=(await api(`/api/sessions?${modeQuery()}`,{method:'POST'})).id;
      await saveSettings();
      await api(`/api/sessions/${state.session}/rtsp`,json('POST',{url:$('rtsp-url').value,authorized:true}));
      state.running=true;state.source='rtsp';state.rtspSequence=null;state.generation++;clearView();
      $('empty').hidden=true;$('stream-state').textContent='SYNC · RTSP';
      $('camera-toggle').innerHTML='<i data-lucide="video-off"></i><span>Stop camera</span>';
      $('input-source').disabled=true;icons();loop(state.generation);return;
    }
    const id = $("camera").value;
    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        ...(id ? { deviceId: { exact: id } } : {}),
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30, max: 30 },
      },
      audio: false,
    });
    state.stream = stream;
    state.source='webcam';$('input-source').disabled=true;
    for (const track of stream.getVideoTracks()) {
      const interrupted = () => {
        if (state.running) {
          stopCamera();
          toast("Camera stream interrupted. Temporary samples cleared.", true);
        }
      };
      track.addEventListener("ended", interrupted);
      track.addEventListener("mute", interrupted);
    }
    $("capture").srcObject = stream;
    await $("capture").play();
    state.session = (await api(`/api/sessions?${modeQuery()}`, { method: "POST" })).id;
    await saveSettings();
    state.running = true;
    state.generation++;
    clearView();
    $("empty").hidden = true;
    $("stream-state").textContent = "SYNC · LOCAL";
    $("camera-toggle").innerHTML =
      '<i data-lucide="video-off"></i><span>Stop camera</span>';
    $("camera").disabled = true;
    await cameras();
    icons();
    loop(state.generation);
  } catch (error) {
    await stopCamera();
    toast(
      error.name === "NotAllowedError"
        ? "Camera permission was denied. Allow camera access in the browser, then try again."
        : error.message,
      true,
    );
  } finally {
    $("camera-toggle").disabled = false;
  }
}
async function stopCamera() {
  $("camera-toggle").disabled = true;
  state.running = false;
  state.generation++;
  $('preview-fps').textContent='0.0';
  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;
  $('input-source').disabled=false;
  $("capture").srcObject = null;
  const old = state.session;
  state.session = null;
  $("camera-toggle").innerHTML =
    '<i data-lucide="video"></i><span>Start camera</span>';
  $("camera").disabled = false;
  $("empty").hidden = false;
  $("stream-state").textContent = "OFFLINE";
  $("preview")
    .getContext("2d")
    .clearRect(0, 0, $("preview").width, $("preview").height);
  clearView();
  icons();
  if (old)
    await api(`/api/sessions/${old}`, { method: "DELETE" }).catch(() => {});
  $("camera-toggle").disabled = false;
}
function renderSynchronizedFrame(captureCanvas,result,frameId) {
  const canvas=$('preview'),width=result.width,height=result.height;
  if(canvas.width!==width||canvas.height!==height){canvas.width=width;canvas.height=height;}
  const ctx=canvas.getContext('2d');ctx.drawImage(captureCanvas,0,0,width,height);
  canvas.dataset.renderMode='synchronized';
  canvas.dataset.frameId=String(frameId);canvas.dataset.overlayFrameId=String(result.frame_id);
    const ratio=result.width/960;ctx.lineWidth=2.5*ratio;ctx.font=`${Math.max(13,16*ratio)}px -apple-system,sans-serif`;
    for(const track of result.tracks){
      const [x,y,w,h]=track.box;ctx.strokeStyle='#57dbc5';ctx.strokeRect(x,y,w,h);
      const label=track.id+(result.self_search&&result.verification.state==='Match'?` · ${result.self_search.self_label}`:'');
      const tw=Math.min(result.width-x,ctx.measureText(label).width+14),ty=Math.max(0,Math.min(result.height-24,y>26?y-24:y+4));
      ctx.fillStyle='#152c23e8';ctx.fillRect(x,ty,tw,22);ctx.fillStyle='#fff';ctx.fillText(label,x+6,ty+16,Math.max(1,tw-12));
      if(track.face_box){ctx.save();ctx.strokeStyle='#ffd17a';ctx.lineWidth=1.8*ratio;ctx.setLineDash([5*ratio,3*ratio]);ctx.strokeRect(...track.face_box);ctx.restore();}
    }
}
async function loop(generation) {
  if (!state.running || generation !== state.generation) return;
  const session = state.session;
  const start = performance.now();
  try {
    const captureCanvas=document.createElement('canvas');
    const frameId=++state.frameSerial;
    let blob;
    if(state.source==='rtsp'){
      const response=await fetch(`/api/sessions/${session}/rtsp-frame`,{signal:AbortSignal.timeout(8000)});
      if(response.status===503){clearView();$('preview').getContext('2d').clearRect(0,0,$('preview').width,$('preview').height);$('gate-state').textContent='Waiting for RTSP frame';setTimeout(()=>loop(generation),350);return;}
      if(!response.ok)throw Error('Test stream unavailable');
      const sequence=response.headers.get('X-Source-Frame');
      if(sequence===state.rtspSequence){setTimeout(()=>loop(generation),50);return;}
      state.rtspSequence=sequence;
      blob=await response.blob();const bitmap=await createImageBitmap(blob);
      captureCanvas.width=bitmap.width;captureCanvas.height=bitmap.height;
      captureCanvas.getContext('2d').drawImage(bitmap,0,0);bitmap.close();
    }else{
      const video = $("capture");
      if (!video.videoWidth) throw new Error("Camera has not produced a frame");
      const scale = Math.min(1, (selfMode()?640:960) / video.videoWidth);
      captureCanvas.width = Math.round(video.videoWidth * scale);captureCanvas.height = Math.round(video.videoHeight * scale);
      captureCanvas.getContext("2d").drawImage(video,0,0,captureCanvas.width,captureCanvas.height);
      blob=await new Promise(resolve=>captureCanvas.toBlob(resolve,'image/jpeg',.85));
    }
    if (!blob) throw new Error("Could not capture camera frame");
    if (
      !state.running ||
      generation !== state.generation ||
      session !== state.session
    )
      return;
    const result = await api(`/api/sessions/${session}/frame`, {
      method: "POST",
      headers: { "Content-Type": "image/jpeg", "X-Frame-Id":String(frameId) },
      body: blob,
    });
    if (!state.running || generation !== state.generation) return;
    if (result.selected !== ($("tester").value || null)) {
      setTimeout(() => loop(generation), 100);
      return;
    }
    if(result.frame_id!==frameId)throw new Error('Frame/result mismatch; restart the camera');
    renderSynchronizedFrame(captureCanvas,result,frameId);
    renderResult(result, performance.now() - start);
  } catch (error) {
    if (state.running && generation === state.generation) {
      toast(error.message, true);
      await stopCamera();
    }
    return;
  }
  setTimeout(
    () => loop(generation),
    Math.max(0, 100 - (performance.now() - start)),
  );
}
async function loadTesters(selectId) {
  state.testers = await api(`/api/testers?${modeQuery()}`);
  const current = selectId ?? (selfMode() ? state.testers[0]?.id : $("tester").value);
  $("tester").replaceChildren(new Option("No tester selected", ""));
  state.testers.forEach((t) => $("tester").append(new Option(t.name, t.id)));
  $("tester").value = state.testers.some((t) => t.id === current)
    ? current
    : "";
  $("tester-total").textContent = selfMode() ? `${state.testers.length} / 1 self profile` : `${state.testers.length} / 10 testers`;
  reference();
  await refreshSelfStatus();
  $('assess-threshold').disabled=!state.testers.length;
  if(!state.testers.length){$('apply-threshold').disabled=true;$('threshold-advice').textContent='Current ArcFace enrollment required';state.thresholdSuggestion=null;}
  if(selfMode() && state.testers.length && state.thresholdRequestedFor!==state.testers[0].id){
    state.thresholdRequestedFor=state.testers[0].id;assessThreshold();
  }
}
function reference() {
  const tester = state.testers.find((t) => t.id === $("tester").value);
  $("reference-photo").hidden = !tester;
  $("reference-empty").hidden = !!tester;
  $("reference-name").textContent = tester?.name || "Not selected";
  $("reference-details").textContent = tester
    ? tester.samples===1?'1 uploaded reference photo':`${tester.samples} quality-selected samples`
    : "Local enrollment";
  $("delete-tester").disabled = !tester;
  if (tester) $("reference-photo").src = `/api/testers/${tester.id}/photo?${modeQuery()}`;
  else $("reference-photo").removeAttribute("src");
}
function renderChart(values) {
  const canvas = $("latency-chart"),
    ctx = canvas.getContext("2d");
  const w = canvas.width,
    h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "#fff";
  ctx.fillRect(0, 0, w, h);
  if (!values?.length) return;
  const max = Math.max(...values) * 1.15,
    left = 52,
    bottom = 30,
    top = 20;
  ctx.font = "13px -apple-system,sans-serif";
  for (let i = 0; i < 5; i++) {
    const y = top + ((h - top - bottom) * i) / 4;
    ctx.strokeStyle = "#e4ebe5";
    ctx.beginPath();
    ctx.moveTo(left, y);
    ctx.lineTo(w - 15, y);
    ctx.stroke();
    ctx.fillStyle = "#68766f";
    ctx.fillText((max * (1 - i / 4)).toFixed(1), 5, y + 4);
  }
  ctx.beginPath();
  ctx.strokeStyle = "#087c72";
  ctx.lineWidth = 2;
  values.forEach((v, i) => {
    const x = left + (i * (w - left - 15)) / Math.max(1, values.length - 1),
      y = top + (h - top - bottom) * (1 - v / max);
    if (!i) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = "#68766f";
  ctx.fillText("1", left, h - 8);
  ctx.fillText(`Query ${values.length}`, w - 90, h - 8);
}
async function pollBenchmark() {
  try {
    const run = await api("/api/benchmark");
    const busy = ["loading", "indexing", "querying"].includes(run.status);
    $("run-benchmark").disabled = busy;
    $("cancel-benchmark").disabled = !busy;
    ["vector-count", "dimensions", "queries", "ef"].forEach(
      (id) => ($(id).disabled = busy),
    );
    if (busy) {
      $("vector-count").value = run.count;
      $("dimensions").value = run.dimensions;
      if (run.queries) $("queries").value = run.queries;
      if (run.ef) $("ef").value = run.ef;
    }
    $("bench-state").textContent = run.message;
    $("bench-percent").textContent = `${Math.round(run.progress)}%`;
    $("bench-progress").value = run.progress;
    const r = run.result;
    if (r && r.timestamp !== state.benchStamp) {
      state.benchStamp = r.timestamp;
      $("result-label").textContent =
        `Last completed run: ${r.vectors.toLocaleString()} vectors · ${r.dimensions}D · ef ${r.hnsw_ef}`;
      $("p50").textContent = r.p50_ms.toFixed(2);
      $("p95").textContent = r.p95_ms.toFixed(2);
      $("recall").textContent = `${(r.recall_at_10 * 100).toFixed(1)}%`;
      $("query-rate").textContent = r.serial_query_rate;
      $("chart-caption").textContent =
        `${r.queries} queries · ${r.vectors.toLocaleString()} vectors · ${r.dimensions}D`;
      renderChart(r.latencies_ms);
      $("run-info").replaceChildren();
      const fields = [
        ["Engine", `${r.engine} ${r.version}`],
        ["Vectors", r.vectors.toLocaleString()],
        ["Dimensions", r.dimensions],
        ["Indexed vectors", r.indexed_vectors.toLocaleString()],
        ["HNSW ef / m", `${r.hnsw_ef} / ${r.m}`],
        ["Queries / warmup", `${r.queries} / ${r.warmup_queries}`],
        ["Transport", r.transport],
        ["Exact search p50", `${r.exact_p50_ms} ms`],
        ["Completed", new Date(r.timestamp * 1000).toLocaleString()],
      ];
      for (const [key, value] of fields) {
        const dt = document.createElement("dt"),
          dd = document.createElement("dd");
        dt.textContent = key;
        dd.textContent = value;
        $("run-info").append(dt, dd);
      }
      $("query-examples").replaceChildren();
      for (const example of r.examples) {
        const tr = document.createElement("tr");
        for (const value of [
          example.query,
          `${example.milliseconds.toFixed(2)} ms`,
          `${(example.recall_at_10 * 100).toFixed(0)}%`,
          example.top_ids.join(", "),
        ]) {
          const td = document.createElement("td");
          td.textContent = value;
          tr.append(td);
        }
        $("query-examples").append(tr);
      }
    }
    if (run.status === "error" && pollBenchmark.error !== run.message) {
      toast(run.message, true);
      pollBenchmark.error = run.message;
    }
  } catch (error) {
    $("bench-state").textContent = "Backend unavailable";
  }
}
document.querySelectorAll("[data-tab]").forEach((button) =>
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-tab]")
      .forEach((b) => b.classList.toggle("active", b === button));
    $("live-view").hidden = button.dataset.tab !== "live";
    $("benchmark-view").hidden = button.dataset.tab !== "benchmark";
    if (button.dataset.tab === "benchmark") pollBenchmark();
  }),
);
$("camera-toggle").onclick = () =>
  state.running ? stopCamera() : startCamera();
$('input-source').onchange=()=>{$('rtsp-controls').hidden=$('input-source').value!=='rtsp';$('camera').disabled=$('input-source').value==='rtsp';};
$('live-mode').onchange=async()=>{
  try{await stopCamera();updateMode();await loadTesters();clearView();}
  catch(e){toast(e.message,true);}
};
$("turnaway-start").onclick = async () => {
  try {
    await api(`/api/sessions/${state.session}/turnaway`, { method: "POST" });
    toast("Continuity check started");
  } catch (error) {
    toast(error.message, true);
  }
};
$("turnaway-export").onclick = () => {
  if (state.session)
    window.location.href = `/api/sessions/${state.session}/turnaway/export`;
};
$("reset").onclick = async () => {
  const generation = state.running ? ++state.generation : state.generation;
  clearView();
  try {
    if (state.session)
      await api(`/api/sessions/${state.session}/reset`, { method: "POST" });
    clearView();
    toast("Temporary samples and tracking reset");
  } catch (e) {
    toast(e.message, true);
  } finally {
    if (state.running && generation === state.generation) loop(generation);
  }
};
$("bench-export").onclick = (event) => {
  if (!state.benchStamp) {
    event.preventDefault();
    toast("No completed benchmark to export");
  }
};
$("export").onclick = () => {
  if (state.session)
    window.location.href = `/api/sessions/${state.session}/export`;
  else toast("Start a session before exporting metrics");
};
$("tester").onchange = async () => {
  reference();
  try {
    await saveSettings();
    clearView();
  } catch (e) {
    toast(e.message, true);
  }
};
["tester-name", "consent"].forEach((id) =>
  $(id).addEventListener("input", updateEnroll),
);
$("enroll").onclick = async () => {
  const button = $("enroll");
  state.enrolling = true;
  button.disabled = true;
  try {
    const result = await api(
      `/api/sessions/${state.session}/enroll`,
      json("POST", {
        name: $("tester-name").value.trim(),
        consent: $("consent").checked,
        track_id: state.last.active_id,
      }),
    );
    await loadTesters(result.id);
    $("tester-name").value = "";
    $("consent").checked = false;
    clearView();
    toast(selfMode() ? "Your profile is registered in Qdrant for the local self-test" : "Tester registered locally");
  } catch (e) {
    toast(e.message, true);
  } finally {
    state.enrolling = false;
    updateEnroll();
  }
};
$('enroll-photo').onclick=()=>{$('enrollment-photo').click();};
$('enrollment-photo').onchange=async()=>{
  const file=$('enrollment-photo').files[0];
  $('enrollment-photo').value='';
  if(!file)return;
  if(state.running || !selfMode() || state.enrolling)return;
  if(!['image/jpeg','image/png'].includes(file.type) || file.size>4_000_000){toast('Choose a JPEG or PNG photo smaller than 4 MB',true);return;}
  state.enrolling=true;updateEnroll();
  $('camera-toggle').disabled=true;$('live-mode').disabled=true;
  try{
    const encoded=await new Promise((resolve,reject)=>{
      const reader=new FileReader();reader.onload=()=>resolve(reader.result.split(',')[1]);
      reader.onerror=()=>reject(new Error('Could not read the selected photo'));reader.readAsDataURL(file);
    });
    const result=await api('/api/self-test/enroll-photo',json('POST',{name:$('tester-name').value.trim(),consent:$('consent').checked,image_base64:encoded}));
    await loadTesters(result.id);$('tester-name').value='';$('consent').checked=false;
    clearView();toast('Your reference photo is registered in the local self-test gallery');
  }catch(error){toast(error.message,true);}
  finally{state.enrolling=false;$('camera-toggle').disabled=false;$('live-mode').disabled=false;updateEnroll();}
};
$("delete-tester").onclick = async () => {
  const id = $("tester").value;
  if (
    !id ||
    !confirm("Delete this tester’s saved face template and reference photo?")
  )
    return;
  try {
    await api(`/api/testers/${id}?${modeQuery()}`, { method: "DELETE" });
    await loadTesters("");
    await saveSettings();
    clearView();
    toast("Tester deleted from the local gallery");
  } catch (e) {
    toast(e.message, true);
  }
};
for (const id of ["min-width", "min-sharpness", "threshold"]) {
  $(id).oninput = () => {
    $(`${id}-out`).textContent =
      id === "min-width"
        ? `${$(id).value} px`
        : id === "threshold"
          ? (+$(id).value).toFixed(2)
          : $(id).value;
  };
  $(id).onchange = async () => {
    try {
      await saveSettings();
      clearView();
    } catch (e) {
      toast(e.message, true);
    }
  };
}
$("run-benchmark").onclick = async () => {
  try {
    await api(
      "/api/benchmark",
      json("POST", {
        count: +$("vector-count").value,
        dimensions: +$("dimensions").value,
        queries: +$("queries").value,
        ef: +$("ef").value,
      }),
    );
    await pollBenchmark();
  } catch (e) {
    toast(e.message, true);
  }
};
$("cancel-benchmark").onclick = async () => {
  await api("/api/benchmark/cancel", { method: "POST" });
  pollBenchmark();
};
$('assess-threshold').onclick=assessThreshold;
$('apply-threshold').onclick=()=>{
  if(state.thresholdSuggestion==null) return;
  $('threshold').value=state.thresholdSuggestion.toFixed(2);
  $('threshold').dispatchEvent(new Event('input'));
  $('threshold').dispatchEvent(new Event('change'));
};
window.addEventListener("beforeunload", () => {
  state.stream?.getTracks().forEach((t) => t.stop());
  if (state.session)
    fetch(`/api/sessions/${state.session}`, {
      method: "DELETE",
      keepalive: true,
    });
});
(async () => {
  if(new URLSearchParams(location.search).get('mode')==='self') $('live-mode').value='self';
  updateMode();
  icons();
  emptySamples();
  renderVotes([]);
  try {
    const health = await api("/api/health");
    $("health").textContent = health.model;
    if (health.legacy_sface_testers) {
      $("model-migration").hidden = false;
      $("model-migration").textContent =
        `${health.legacy_sface_testers} earlier SFace registration(s) preserved separately. ArcFace requires a new enrollment.`;
    }
    await loadTesters();
    if (navigator.mediaDevices) await cameras();
    else toast("Open this demo on localhost to enable camera access", true);
    const previous = (await api("/api/benchmark")).result;
    if (previous) {
      $("vector-count").value = previous.vectors;
      $("dimensions").value = previous.dimensions;
      $("queries").value = previous.queries;
      $("ef").value = previous.hnsw_ef;
    }
  } catch (e) {
    toast(e.message, true);
    $("health").textContent = "Backend unavailable";
  }
  await pollBenchmark();
  setInterval(pollBenchmark, 1500);
})();
