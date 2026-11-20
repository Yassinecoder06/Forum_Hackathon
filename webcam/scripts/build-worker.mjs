import {build} from 'esbuild';
import {fileURLToPath} from 'node:url';
import path from 'node:path';
const root=path.resolve(path.dirname(fileURLToPath(import.meta.url)),'..');
// MediaPipe loads its WASM glue through importScripts: a classic worker is required.
await build({entryPoints:[path.join(root,'src/detector.js')],bundle:true,format:'iife',platform:'browser',target:'es2022',outfile:path.join(root,'public/vendor/detector-worker.js'),minify:true});
