import { mkdir, cp, writeFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
const models={
  pose:'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task',
  hands:'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
  face:'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task',
};
await mkdir(path.join(root,'public/models'),{recursive:true});
await mkdir(path.join(root,'public/vendor/mediapipe'),{recursive:true});
await cp(path.join(root,'node_modules/@mediapipe/tasks-vision/wasm'),path.join(root,'public/vendor/mediapipe'),{recursive:true});
const manifest={sdk:'@mediapipe/tasks-vision@0.10.32',models:{}};
for(const [name,url] of Object.entries(models)){
  const target=path.join(root,'public/models',name+'.task');let bytes;
  try{bytes=await readFile(target);}catch{
    console.log('Downloading '+name+' model…');
    const response=await fetch(url);if(!response.ok)throw Error(`Model download failed: ${response.status}`);
    bytes=Buffer.from(await response.arrayBuffer());
    if(bytes.length<100000)throw Error('Unexpected model payload: '+name);
    await writeFile(target,bytes);
  }
  manifest.models[name]={url,bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')};
  console.log(`${name}: ${(bytes.length/1024/1024).toFixed(1)} MB`);
}
await writeFile(path.join(root,'public/models/manifest.json'),JSON.stringify(manifest,null,2)+'\n');
console.log('Local models and WASM ready. Camera frames stay in the browser.');
