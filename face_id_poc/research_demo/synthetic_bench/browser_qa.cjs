const {chromium}=require('playwright');
const fs=require('fs'),path=require('path');
const root=path.resolve(__dirname,'..'),out=path.join(root,'qa/digiface');
const base=`http://127.0.0.1:${process.env.CAMEX_GALLERY_PORT||9161}`;
const run=process.env.CAMEX_GALLERY_RUN||'all_identities';
fs.mkdirSync(out,{recursive:true});
(async()=>{
 const browser=await chromium.launch({executablePath:process.env.CAMEX_CHROME||undefined,headless:true});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1100}}),errors=[];
  page.on('pageerror',e=>errors.push(String(e)));
  await page.goto(base);
  await page.selectOption('#run',run);
  await page.waitForSelector('.match img');
  await page.waitForFunction(()=>[...document.querySelectorAll('.match img')].every(i=>i.complete&&i.naturalWidth>0));
  await page.waitForFunction(()=>document.querySelector('#end-to-end').textContent.includes('Median'));
  await page.screenshot({path:path.join(out,'desktop.png'),fullPage:true});
  const options=await page.locator('#probe option').evaluateAll(els=>els.map(x=>x.value));
  await page.selectOption('#probe',options[Math.min(1,options.length-1)]);
  await page.click('#search');
  await page.waitForFunction(()=>!document.querySelector('#search').disabled);
  if(await page.locator('.match').count()!==5)throw Error('Expected five candidates');
  const download=await page.request.get(`${base}/api/runs/${run}/result`);
  if(!download.ok())throw Error('Result download failed');
  await page.setViewportSize({width:390,height:844});
  await page.screenshot({path:path.join(out,'mobile.png'),fullPage:true});
  if(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth))throw Error('Mobile overflow');
  if(errors.length)throw Error(errors.join('\n'));
  fs.writeFileSync(path.join(out,'browser-report.json'),JSON.stringify({passed:true,run,synthetic_only:true,
   checks:['held-out query','five candidate images','latency display','downloadable JSON','desktop/mobile layout']},null,2));
  console.log('Gallery browser QA passed');
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1});
