/** Real MediaPipe integration using Chromium's prerecorded fake camera, never hardware. */
import {chromium} from '../../e2e/node_modules/playwright/index.mjs';
import fs from 'node:fs/promises';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import assert from 'node:assert/strict';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'../..');
const artifacts=path.join(root,'.runtime/webcam-check');await fs.mkdir(artifacts,{recursive:true});
const fixture=process.env.WEBCAM_TEST_VIDEO;if(!fixture)throw Error('Set WEBCAM_TEST_VIDEO to a prerecorded .y4m file.');
const webcamUrl = new URL(process.env.WEBCAM_APP_URL || 'http://127.0.0.1:5180/');
const browser=await chromium.launch({executablePath:process.env.CHROME_PATH,headless:true,args:[
  '--no-sandbox','--use-gl=angle','--use-angle=swiftshader','--enable-unsafe-swiftshader',
  '--use-fake-ui-for-media-stream','--use-fake-device-for-media-stream','--use-file-for-fake-video-capture='+path.resolve(fixture)]});
const context=await browser.newContext({viewport:{width:1440,height:1080},acceptDownloads:true});
const page=await context.newPage(),errors=[],external=[];
page.on('pageerror',e=>errors.push(e.message));
page.on('request',r=>{if(new URL(r.url()).origin!==webcamUrl.origin&&!r.url().startsWith('blob:'))external.push(r.url());});
const report={physical_camera_used:false};
try{
  await page.goto(webcamUrl.href);
  assert.equal(await page.locator('#camera').evaluate(v=>v.srcObject),null);
  await page.locator('#acceleration').selectOption(process.env.WEBCAM_TEST_DELEGATE??'CPU');
  await page.locator('#start-empty').click();
  await page.waitForFunction(()=>!document.querySelector('#record').disabled,null,{timeout:120000});
  await page.waitForTimeout(1000);
  await page.screenshot({path:path.join(artifacts,'live.png'),fullPage:true});
  report.engine=await page.locator('#engine').innerText();
  await page.locator('#record').click();await page.waitForTimeout(4500);await page.locator('#record').click();
  report.take=await page.locator('#take-info').innerText();
  for(const [id,name] of [['raw','capture.json'],['motion','motion.json'],['animics','animics.json'],['video-export','source.webm']]){
    await page.waitForFunction(id=>!document.getElementById(id).disabled,id);
    const [download]=await Promise.all([page.waitForEvent('download',{timeout:30000}),page.locator('#'+id).click()]);
    await download.saveAs(path.join(artifacts,name));
    if(id==='video-export')await page.locator('#close-dialog').click();
  }
  const capture=JSON.parse(await fs.readFile(path.join(artifacts,'capture.json'),'utf8'));
  assert.ok(capture.frames.length>=2);assert.ok(capture.frames.some(f=>f.pose));
  assert.ok(capture.frames.some(f=>f.hands.left||f.hands.right));
  assert.ok(capture.frames.some(f=>f.face&&Object.keys(f.blendshapes).length>=50));
  report.frames=capture.frames.length;report.faceFrames=capture.frames.filter(f=>f.face).length;
  report.medianInferenceMs=capture.frames.map(f=>f.inferenceMs).sort((a,b)=>a-b)[Math.floor(capture.frames.length/2)];
  await page.locator('#play').click();await page.waitForTimeout(600);
  assert.equal(await page.locator('#replay-badge').isVisible(),true);
  await page.screenshot({path:path.join(artifacts,'replay.png'),fullPage:true});
  await page.locator('#timeline').fill('0');
  await page.locator('#return-live').click();
  const liveTrack=await page.locator('#camera').evaluateHandle(v=>v.srcObject.getVideoTracks()[0]);
  await page.locator('#stop-camera').click();
  assert.equal(await liveTrack.evaluate(t=>t.readyState),'ended');
  assert.equal(await page.locator('#camera').evaluate(v=>v.srcObject),null);
  assert.equal(await page.locator('#raw').isEnabled(),true);
  await page.setViewportSize({width:390,height:844});
  assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);
  await page.screenshot({path:path.join(artifacts,'mobile.png'),fullPage:true});
  await page.locator('#clear').click();assert.equal(await page.locator('#raw').isDisabled(),true);
  assert.deepEqual(errors,[]);assert.deepEqual(external,[]);
  report.pageErrors=errors;report.externalRequests=external;report.status='passed';
  console.log(JSON.stringify(report,null,2));
}catch(error){
  report.status='failed';report.error=error.stack;report.message=await page.locator('#message').innerText();
  await page.screenshot({path:path.join(artifacts,'error.png'),fullPage:true});throw error;
}finally{
  await fs.writeFile(path.join(artifacts,'report.json'),JSON.stringify(report,null,2));await browser.close();
}
