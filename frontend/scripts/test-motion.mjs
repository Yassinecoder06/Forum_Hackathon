import {build} from 'esbuild';
import {readdirSync} from 'node:fs';
// every tests/*.test.ts, in name order (motion.test.ts first)
const entries=readdirSync('tests').filter((f)=>f.endsWith('.test.ts')).sort((a,b)=>(a==='motion.test.ts'?-1:b==='motion.test.ts'?1:a.localeCompare(b))).map((f)=>'tests/'+f);
for(const entry of entries){
  const result=await build({entryPoints:[entry],bundle:true,platform:'node',format:'esm',write:false,external:['node:*']});
  await import('data:text/javascript;base64,'+Buffer.from(result.outputFiles[0].text).toString('base64'));
}