import { build } from "esbuild";
import path from "node:path";
const baseline = process.env.AVATAR_BASELINE_RETARGETER;
const result = await build({
  entryPoints: [process.env.AVATAR_MEASURE_ENTRY || "scripts/measure-avatar.ts"], bundle: true, platform: "node", format: "esm", write: false, external: ["node:*"],
  nodePaths: [path.resolve("node_modules")],
  plugins: baseline ? [{ name: "frozen-baseline", setup(build) {
    build.onResolve({ filter: /utils\/retargetHumanoid$/ }, () => ({ path: path.resolve(baseline) }));
  } }] : [],
});
await import("data:text/javascript;base64," + Buffer.from(result.outputFiles[0].text).toString("base64"));
