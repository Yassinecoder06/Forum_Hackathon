import { chromium } from 'playwright';
import assert from 'node:assert/strict';
const browser = await chromium.launch({headless:true,
  ...(process.env.CHROME_PATH ? {executablePath:process.env.CHROME_PATH} : {}),
  args:['--no-sandbox','--enable-unsafe-swiftshader']});
try {
  const page = await browser.newPage({acceptDownloads:true});
  await page.goto(process.env.APP_URL || 'http://127.0.0.1:5173');
  await page.locator('[data-avatar-ready="true"]').waitFor({timeout:60000});
  await page.getByLabel('Votre texte en français',{exact:true}).fill('chat dormir');
  const response = page.waitForResponse(r=>r.url().endsWith('/api/compose-animation') && r.request().method()==='POST');
  await page.getByRole('button',{name:'Créer l’animation',exact:true}).click();
  const plan = await (await response).json();
  await page.getByRole('button',{name:'Mettre en pause',exact:true}).click();
  const slider=page.getByRole('slider',{name:'Position de l’animation',exact:true});
  const canvas=page.locator('.avatar-stage canvas');
  await slider.fill('0'); await page.waitForTimeout(300);
  const zero=await canvas.evaluate(el=>el.toDataURL());
  await slider.fill(String(Math.round(plan.duration_ms*.8/10)*10)); await page.waitForTimeout(300);
  assert.notEqual(await canvas.evaluate(el=>el.toDataURL()),zero,'Distinct paused pose required for this regression');
  await page.evaluate(()=>{
    const original=CanvasRenderingContext2D.prototype.drawImage;
    CanvasRenderingContext2D.prototype.drawImage=function(...args){
      if(this.canvas.width===1280 && this.canvas.height===960 && !window.__exportFirstSource && args[0] instanceof HTMLCanvasElement)
        window.__exportFirstSource=args[0].toDataURL();
      return original.apply(this,args);
    };
  });
  const download=page.waitForEvent('download',{timeout:60000});
  await page.getByRole('button',{name:/Télécharger l’aperçu vidéo/}).click();
  await download;
  assert.equal(await page.evaluate(()=>window.__exportFirstSource),zero,'First encoded canvas must use frame zero, not old paused pose');
  console.log('Export reset passed: first captured source is exactly frame zero after seeking to a distinct pose.');
} finally {await browser.close();}
