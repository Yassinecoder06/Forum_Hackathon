import {createRequire} from 'node:module';
import {fileURLToPath} from 'node:url';
const require=createRequire(new URL('../e2e/package.json',import.meta.url));
const {chromium}=require('playwright');
const browser=await chromium.launch({headless:true,...(process.env.CHROME_PATH?{executablePath:process.env.CHROME_PATH}:{}),args:['--no-sandbox']});
try{
 const page=await browser.newPage();
 await page.goto(new URL('../docs/pipeline/pipeline.html',import.meta.url).href);
 await page.evaluate(()=>document.fonts.ready);
 const path=fileURLToPath(new URL('../SignaCulture_Full_Pipeline.pdf',import.meta.url));
 await page.pdf({path,preferCSSPageSize:true,printBackground:true,tagged:true});
 console.log(path);
}finally{await browser.close();}
