const {chromium}=require('playwright');
const fs=require('fs'),path=require('path'),crypto=require('crypto');
const {spawn}=require('child_process');
const root=__dirname, stamp=String(Date.now()), out=path.join(root,'qa/self-test');
fs.mkdirSync(out,{recursive:true});
const home=process.env.CAMEX_LAB_HOME||path.join(root,'.runtime');
const galleryRoot=process.env.CAMEX_GALLERY_ROOT||path.join(process.env.LOCAL_LAB_DATA||path.join(home,'data'),'digiface_research');
const sourceFile=path.join(galleryRoot,process.env.CAMEX_GALLERY_RUN||'all_identities','result.json');
const qdrant=`http://127.0.0.1:${process.env.CAMEX_QDRANT_PORT||16333}`;
const hash=()=>crypto.createHash('sha256').update(fs.readFileSync(sourceFile)).digest('hex');
const originalHash=hash(), source=JSON.parse(fs.readFileSync(sourceFile));
const collection='local_self_test_qa_'+stamp, port=process.env.CAMEX_QA_PORT||9162, base=`http://127.0.0.1:${port}`;
const log=fs.openSync(path.join(out,'server.log'),'a');
const server=spawn(path.join(root,'.venv/bin/python'),['-m','uvicorn','app:app','--host','127.0.0.1','--port',String(port)],
 {cwd:root,env:{...process.env,CAMEX_GALLERY_ROOT:galleryRoot,LOCAL_LAB_DATA:path.join(out,'data-'+stamp),LOCAL_SELF_TEST_COLLECTION:collection},stdio:['ignore',log,log]});
const sleep=ms=>new Promise(r=>setTimeout(r,ms));
let browser,debugPage,debugObservations=[];
async function request(url,options){const r=await fetch(url,options);if(!r.ok)throw Error(`${r.status}: ${await r.text()}`);return r.json();}
const body=x=>({method:'PUT',headers:{'Content-Type':'application/json'},body:JSON.stringify(x)});
(async()=>{
 for(let i=0;i<60;i++){try{await request(base+'/api/health');break;}catch(e){if(server.exitCode!=null)throw Error('Isolated QA server exited');if(i===59)throw e;await sleep(500);}}
 const originalCount=(await request(qdrant+'/collections/'+source.collection)).result.points_count;
 browser=await chromium.launch({executablePath:process.env.CAMEX_CHROME||undefined,headless:true,args:[
  '--use-fake-device-for-media-stream','--use-fake-ui-for-media-stream',`--use-file-for-fake-video-capture=${root}/qa/synchronized_camera.y4m`]});
 const page=await browser.newPage({viewport:{width:1440,height:1100},permissions:['camera']});const errors=[];
 debugPage=page;
 page.on('pageerror',e=>errors.push(String(e)));
 let delayFrames=false,phase='setup';const frames=new Map();const observations=[];const thumbnailRequests=[];
 debugObservations=observations;
 page.on('request',request=>{if(request.url().includes('/api/self-test/synthetic/'))thumbnailRequests.push(request.url());});
 await page.route('**/api/sessions/*/frame',async route=>{
   const request=route.request(),id=request.headers()['x-frame-id'];
   if(delayFrames)await sleep(350);
   const started=Date.now();const response=await route.fetch();
   if(response.ok()){
     const result=await response.json();
     frames.set(id,{bytes:request.postDataBuffer().toString('base64'),result});
     if(frames.size>12)frames.delete(frames.keys().next().value);
     observations.push({at:Date.now(),phase,frame_roundtrip_ms:Date.now()-started,tracking_ms:result.timing.total_ms,frame_id:result.frame_id,face_checked:result.face_checked,recognition:result.recognition});
   }
   await route.fulfill({response});
 });
 await page.goto(base+'/?mode=self');
 await page.waitForFunction(count=>document.querySelector('#self-gallery-count').textContent.includes(count),originalCount.toLocaleString());
 await page.getByRole('button',{name:'Start camera',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#sample-count').textContent==='5 / 5 accepted',null,{timeout:30000});
 await page.waitForFunction(()=>document.querySelector('#verdict-label').textContent==='No self profile enrolled');
 await page.fill('#tester-name','QA research self');await page.check('#consent');
 await page.getByRole('button',{name:'Register best 3 samples',exact:true}).click();
 await page.waitForFunction(()=>document.querySelector('#verdict-label').textContent==='Match · QA research self',null,{timeout:30000});
 await page.waitForFunction(count=>document.querySelector('#self-gallery-count').textContent.includes(count),(originalCount+1).toLocaleString());
 await page.waitForFunction(()=>document.querySelectorAll('.self-candidate').length===5&&[...document.querySelectorAll('.self-candidate img')].every(x=>x.complete&&x.naturalWidth>0));
 const profile=(await request(base+'/api/testers?mode=self'))[0];
 const stored=await request(qdrant+'/collections/'+collection+'/points/6290a7d8-6a36-4b56-b2ab-0fce48751585');
 if(stored.result.payload.tester_id!==profile.id)throw Error('Enrollment not actually stored in Qdrant');
 await page.waitForFunction(()=>document.querySelector('#threshold-advice').textContent.includes('Provisional'),null,{timeout:30000});
 phase='steady';await page.waitForTimeout(9000);phase='controls';
 const screenshotMetrics=await page.evaluate(()=>({fps:document.querySelector('#fps').textContent,preview_fps:document.querySelector('#preview-fps').textContent,score:document.querySelector('#self-score').textContent,
   search:document.querySelector('#self-search-time').textContent,lead:document.querySelector('#self-lead').textContent}));
 await page.waitForTimeout(700);
 await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
 delayFrames=true;
 await page.waitForTimeout(1800);
 let frameMatch;
 for(let attempt=0;attempt<8;attempt++){
   const id=await page.locator('#preview').getAttribute('data-frame-id');const sourceFrame=frames.get(id);
   if(!sourceFrame){await sleep(100);continue;}
   frameMatch=await page.locator('#preview').evaluate(async(canvas,input)=>{
     if(canvas.dataset.frameId!==input.id)return {retry:true};
     const expected=canvas.dataset.overlayFrameId;
     const pixel=canvas.getContext('2d').getImageData(20,canvas.height-15,1,1).data;
     const bytes=Uint8Array.from(atob(input.bytes),x=>x.charCodeAt(0));
     const bitmap=await createImageBitmap(new Blob([bytes],{type:'image/jpeg'}));
     const reference=document.createElement('canvas');reference.width=bitmap.width;reference.height=bitmap.height;
     const ctx=reference.getContext('2d');ctx.drawImage(bitmap,0,0);bitmap.close();
     const ref=ctx.getImageData(20,reference.height-15,1,1).data;
     return {frame_id:input.id,overlay_frame_id:expected,maximum_channel_error:Math.max(...[0,1,2].map(i=>Math.abs(pixel[i]-ref[i]))),mode:canvas.dataset.renderMode};
   },{id,bytes:sourceFrame.bytes});
   if(!frameMatch.retry)break;
 }
 if(!frameMatch || frameMatch.retry || frameMatch.frame_id!==frameMatch.overlay_frame_id || frameMatch.maximum_channel_error>8 || frameMatch.mode!=='synchronized')throw Error('Frame synchronization failed: '+JSON.stringify(frameMatch));
 delayFrames=false;
 await page.locator('#threshold').evaluate(el=>{el.value='0.90';el.dispatchEvent(new Event('input',{bubbles:true}));el.dispatchEvent(new Event('change',{bubbles:true}));});
 await page.waitForFunction(()=>document.querySelector('#threshold-out').textContent==='0.90');
 await page.waitForFunction(()=>document.querySelector('#verdict-label').textContent==='Match · QA research self',null,{timeout:30000});
 await page.setViewportSize({width:390,height:844});await page.waitForTimeout(500);
 await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
 if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow');
 await page.getByRole('button',{name:'Stop camera',exact:true}).click();
 const session=(await request(base+'/api/sessions?mode=self',{method:'POST'})).id;
 await request(`${base}/api/sessions/${session}/settings`,body({selected:profile.id,threshold:0.0,minimum_width:70,minimum_sharpness:45}));
 let negative;const negativeFrames=[];
 for(let i=0;i<60;i++){
  const tick=Date.now();
  negative=await request(`${base}/api/sessions/${session}/frame`,{method:'POST',headers:{'Content-Type':'image/jpeg'},body:fs.readFileSync(path.join(root,'qa/synthetic_2.jpg'))});
  negativeFrames.push({milliseconds:Date.now()-tick,track:negative.active_id,verification:negative.verification,recognition:negative.recognition});
  if(negative.verification.state==='Match')throw Error('Different synthetic identity falsely accepted');
  await sleep(100);
  if(negative.verification.state==='No match'&&negative.verification.count===5) break;
 }
 fs.writeFileSync(path.join(out,'negative-diagnostic.json'),JSON.stringify({verification:negative.verification,
   head_status:negative.head_status,quality:negative.quality,timing:negative.timing,self_search:negative.self_search,frames:negativeFrames},null,2));
 if(negative.verification.state!=='No match'||negative.self_search?.self_is_first)throw Error('Negative fixture check failed: '+JSON.stringify(negative.verification));
 await request(`${base}/api/testers/${profile.id}?mode=self`,{method:'DELETE'});
 if((await request(base+'/api/testers?mode=self')).length)throw Error('Local test profile was not deleted');
 const after=(await request(qdrant+'/collections/'+collection)).result.points_count;
 if(after!==0)throw Error('Qdrant test template was not deleted');
 if(hash()!==originalHash)throw Error('Benchmark report modified');
 if((await request(qdrant+'/collections/'+source.collection)).result.points_count!==originalCount)throw Error('Synthetic collection modified');
 if(errors.length)throw Error(errors.join('\n'));
 const steady=observations.filter(x=>x.phase==='steady');
 if(!steady.length || steady.some(x=>x.recognition.pipeline!=='process'||x.recognition.worker_pid===x.recognition.tracking_pid))throw Error('Recognition is not process-isolated');
 if(thumbnailRequests.length)throw Error('Individual synthetic thumbnails still use the tracking HTTP lane');
 const summarize=rows=>{const values=rows.map(x=>x.tracking_ms).sort((a,b)=>a-b);return values.length?{frames:values.length,p50_ms:values[Math.floor((values.length-1)*.5)],p95_ms:values[Math.floor((values.length-1)*.95)],maximum_ms:values.at(-1)}:null;};
 const isolation={tracking_pid:steady[0].recognition.tracking_pid,recognition_pid:steady[0].recognition.worker_pid,
   all:summarize(steady),recognition_running:summarize(steady.filter(x=>x.recognition.in_flight)),recognition_idle:summarize(steady.filter(x=>!x.recognition.in_flight)),individual_thumbnail_requests:thumbnailRequests.length};
 fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify({passed:true,synthetic_camera_fixture:true,physical_webcam_tested:false,
   baseline_gallery_unchanged:true,synthetic_gallery_size:originalCount,qa_tester_removed:true,metrics:screenshotMetrics,frame_synchronization:frameMatch,process_isolation:isolation,
   every_analyzed_frame_checks_faces:observations.every(x=>x.face_checked),recognition_samples:observations.slice(-30),
   negative_at_zero_threshold:{state:negative.verification.state,self_score:negative.self_search.self_score,nearest:negative.self_search.candidates[0].label},
   checks:['explicit enrollment writes Qdrant',`${originalCount} synthetic + 1 self query`,'self top1 + threshold + 3/5 votes',
    'threshold slider updates settings','different synthetic face rejected even at zero threshold',
    'five candidate images rendered','desktop/mobile layout','profile removal from SQLite and Qdrant','original benchmark untouched',
    'changing-frame pixels and overlay IDs match despite injected network delay','face detection every analyzed frame','background recognition timing and gallery size visible']},null,2));
 console.log('Self-test browser QA passed',screenshotMetrics);
})().catch(async e=>{
 console.error(e);
 const state=debugPage?await debugPage.evaluate(()=>Object.fromEntries(['sample-count','recognition-status','gate-state','fps'].map(id=>[id,document.getElementById(id)?.textContent]))).catch(()=>null):null;
 fs.writeFileSync(path.join(out,'failure-diagnostic.json'),JSON.stringify({error:String(e),state,observations:debugObservations.slice(-10)},null,2));
 console.error('QA state',state,debugObservations.slice(-1));process.exitCode=1;
}).finally(async()=>{
 if(browser)await browser.close();server.kill('SIGTERM');
 await fetch(qdrant+'/collections/'+collection,{method:'DELETE'}).catch(()=>{});
});
