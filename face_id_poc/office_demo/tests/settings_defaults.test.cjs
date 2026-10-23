const assert = require('node:assert/strict');
const fs = require('node:fs');
const path = require('node:path');
const test = require('node:test');
const vm = require('node:vm');

const source = fs.readFileSync(path.join(__dirname, '../static/app.js'), 'utf8');
const start = source.indexOf("  $('settings-defaults').addEventListener('click', async () => {");
const end = source.indexOf("  $('settings-form').addEventListener('submit'", start);
assert(start >= 0 && end > start);
const defaults = {match_threshold:.30, match_margin:.06, admission_threshold:.68,
  admission_margin:.10, adaptation_mode:'review', analysis_fps:10, recognition_hz:1,
  face_min_width:20, quality_strictness:49, single_reference_mode:'review', face_detection_mode:'full_frame'};

async function run({confirmed=true, preset=defaults, connected=true, reject=false} = {}) {
  let click;
  const calls=[], notices=[], toasts=[];
  const dirty = new Set(Object.keys(defaults));
  const context = {
    $: () => ({addEventListener: (name, handler) => {click=handler;}}),
    savingSettings:false, app:{connected}, settingNames:Object.keys(defaults), dirtySettings:dirty,
    syncSettings:()=>{}, notice:(...args)=>notices.push(args), toast:(...args)=>toasts.push(args),
    describe:String, stateFlight:null, refreshState:async()=>calls.push('refresh'),
    confirmAction:async(...args)=>{calls.push({confirm:args}); return confirmed;},
    json:(method,body)=>({method,body}),
    api:async(url,options)=>{
      calls.push({url,options});
      if (!options) return {profile:'office-demo-20261023',settings:preset};
      if(reject) throw Error('Save failed');
      return defaults;
    },
  };
  vm.runInNewContext(source.slice(start,end),context);
  await click();
  return {calls,notices,toasts,dirty,context};
}

test('Reset posts all eleven defaults fetched from the backend', async()=>{
  const result=await run();
  const posts=result.calls.filter(c=>c.options);
  assert.equal(posts.length,1);
  assert.equal(posts[0].url,'/api/settings');
  assert.deepEqual(posts[0].options.body,defaults);
  assert.equal(result.dirty.size,0);
  assert.equal(result.context.savingSettings,false);
  assert.equal(result.toasts[0][0],'All settings restored to the Office Demo preset.');
});

test('Cancel preserves unsaved fields and does not post', async()=>{
  const result=await run({confirmed:false});
  assert.equal(result.calls.filter(c=>c.options).length,0);
  assert.equal(result.dirty.size,11);
  assert.equal(result.context.savingSettings,false);
});

test('Incomplete preset or failed save cannot claim a successful reset', async()=>{
  for(const options of [{preset:{face_min_width:20}},{reject:true}]) {
    const result=await run(options);
    assert.equal(result.toasts.length,0);
    assert.equal(result.dirty.size,11);
    assert.ok(result.notices.some(c=>c[1]));
    assert.equal(result.context.savingSettings,false);
  }
});

test('Disconnected UI cannot issue a reset',async()=>{
  const result=await run({connected:false});
  assert.equal(result.calls.length,0);
});
