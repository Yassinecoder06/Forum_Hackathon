import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
import path from 'node:path';
const file=path.resolve(process.env.REVIEW_HTML||'../reports/private/lsfb-alignment-audit/video-review/index.html');
const browser=await chromium.launch({headless:true,executablePath:process.env.CHROME_PATH||'/home/jmal/.local/bin/google-chrome',args:['--no-sandbox']});
try {
 const page=await browser.newPage({viewport:{width:1280,height:900}}),errors=[];
 page.on('pageerror',e=>errors.push(e.message));
 await page.goto(pathToFileURL(file).href);
 const rows=page.locator('article');assert.equal(await rows.count(),55);
 assert.equal(await page.locator('video').count(),55);
 assert.equal(await page.locator('select.decision').evaluateAll(xs=>xs.filter(x=>x.value!=='').length),0);
 const first=rows.first(),video=first.locator('video');
 await video.evaluate(v=>{v.preload='auto';v.load();});
 await video.evaluate(v=>v.readyState>=1?null:new Promise((resolve,reject)=>{v.addEventListener('loadedmetadata',resolve,{once:true});v.addEventListener('error',()=>reject(Error('Video failed')),{once:true});}));
 const event=first.locator('.events button').first();await event.click();
 await page.waitForFunction(()=>document.querySelector('article .events button').getAttribute('aria-pressed')==='true');
 assert(await first.locator('.clock').innerText().then(t=>t.includes('Source')));
 const timing=await page.evaluate(()=>({expected:(data[0].hand_tiers[Object.keys(data[0].hand_tiers)[0]][0].start_ms-data[0].clip_start_ms)/1000,actual:document.querySelector('video').currentTime}));
 assert(Math.abs(timing.actual-Math.max(0,timing.expected))<.05);
 // Test rejection paths without persisting or exporting any review decision.
 await page.locator('#reviewer').fill('__automated_rejection_test__');
 await first.locator('.decision').selectOption('accept');await first.locator('.save').click();
 assert.match(await first.locator('.status').innerText(),/requires all four checks/);
 await first.locator('input[type=checkbox]').evaluateAll(xs=>xs.forEach(x=>x.checked=true));
 await page.evaluate(()=>{data[0].clip=null;});await first.locator('.save').click();
 assert.match(await first.locator('.status').innerText(),/requires an available source video/);
 assert.equal(await page.evaluate(()=>Object.keys(decisions).length),0);
 assert.deepEqual(errors,[]);
 console.log('55 videos/pending rows, actual video decode, timestamp seek/highlighting, and rejection gates passed; zero decisions created.');
} finally {await browser.close();}
