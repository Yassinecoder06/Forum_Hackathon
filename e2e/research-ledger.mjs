import { chromium } from 'playwright';
import assert from 'node:assert/strict';
import { mkdir } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
const base=process.env.APP_URL||'http://127.0.0.1:15177';
const output=process.env.E2E_OUTPUT||fileURLToPath(new URL('../reports/private/research-panel-browser/',import.meta.url));
await mkdir(output,{recursive:true});
const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{}),args:['--no-sandbox','--enable-unsafe-swiftshader']});
try {
 const page=await browser.newPage({viewport:{width:1440,height:1000}});const errors=[];page.on('pageerror',e=>errors.push(e.message));
 const response=await fetch(base+'/api/research/status');assert.equal(response.status,200);const data=await response.json();const entries=data.recent_experiments.entries;
 assert.equal(data.recent_experiments.available,true);assert.equal(entries.length,6);assert(entries.every(r=>r.evidence_status==='verified'&&!r.inference_ready));
 assert(!JSON.stringify(data.recent_experiments).includes('/home/'));assert(!JSON.stringify(data.recent_experiments).includes('checkpoint'));
 await page.goto(base);await page.getByRole('button',{name:'Recherche',exact:true}).click();const panel=page.getByRole('region',{name:'Bilan des expériences récentes'});
 await panel.locator('[data-run-id]').first().waitFor();assert.equal(await panel.locator('[data-run-id]').count(),6);assert.equal(await panel.getByRole('button').count(),0);
 const select=panel.getByLabel('Langue des expériences',{exact:true});await select.selectOption('fsl');assert.equal(await panel.locator('[data-run-id]').count(),2);assert(!await panel.innerText().then(t=>t.includes('geoquery-full-graph-v1')));
 await select.selectOption('sfb');assert.equal(await panel.locator('[data-run-id]').count(),4);assert(await panel.innerText().then(t=>t.includes('52.17 %')&&t.includes('F1 : 2.49 %')));
 await page.screenshot({path:output+'/research.png',fullPage:true});await page.setViewportSize({width:390,height:844});assert(await panel.isVisible());
 const width=await page.evaluate(()=>({body:document.documentElement.scrollWidth,viewport:innerWidth}));assert(width.body<=width.viewport+2,`Page overflows mobile viewport: ${JSON.stringify(width)}`);
 await page.screenshot({path:output+'/mobile.png',fullPage:true});assert.deepEqual(errors,[]);console.log('Actual catalog evidence, language filters, task-specific scores, no inference actions, privacy and mobile layout passed.');
} finally {await browser.close();}
