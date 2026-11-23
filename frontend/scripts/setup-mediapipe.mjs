// Puts everything MediaPipe needs under public/mediapipe so the Studio never reads the internet at run time:
// the WebAssembly runtime (copied from node_modules) and the three landmark models (downloaded once, then kept).
import { copyFileSync, existsSync, mkdirSync, readdirSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const out = join(root, "public", "mediapipe");
const wasmFrom = join(root, "node_modules", "@mediapipe", "tasks-vision", "wasm");
const BASE = "https://storage.googleapis.com/mediapipe-models";
const models = {
  "pose_landmarker_lite.task": `${BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
  "hand_landmarker.task": `${BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
  "face_landmarker.task": `${BASE}/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
};

mkdirSync(join(out, "wasm"), { recursive: true });
mkdirSync(join(out, "models"), { recursive: true });
for (const f of readdirSync(wasmFrom)) copyFileSync(join(wasmFrom, f), join(out, "wasm", f));
console.log("wasm runtime copied");
for (const [name, url] of Object.entries(models)) {
  const dest = join(out, "models", name);
  if (existsSync(dest)) { console.log(`${name} already there`); continue; }
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${name}: HTTP ${r.status}`);
  writeFileSync(dest, Buffer.from(await r.arrayBuffer()));
  console.log(`${name} downloaded`);
}
