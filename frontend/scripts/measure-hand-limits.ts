/**
 * Anatomical joint-limit violations of the displayed hands (wrist + 15 finger joints per hand) on real motions.
 * Local engineering diagnostic: motions and the avatar are private files and are only read, never written.
 *
 *   AVATAR_MEASURE_ENTRY=scripts/measure-hand-limits.ts node scripts/measure-avatar.mjs <avatar.glb> <motions dir> [out.json]
 *   (HANDS_MODES=before + AVATAR_BASELINE_RETARGETER=<older retargetHumanoid.ts> measures an earlier runtime)
 *
 * Every motion is played at 60 fps through the retargeter and the performance layer (springs), once with the hand
 * constraints disabled (before) and once enabled (after). Violations are counted against the hard limits of
 * handConstraints.ts with 1 degree of slack, per joint type and kind.
 */
import fs from "node:fs";
import path from "node:path";
import * as THREE from "three";
import { HumanoidRetargeter } from "../src/utils/retargetHumanoid";
import { PerformanceLayer } from "../src/utils/performanceLayer";
import { HandConstraints, countViolations } from "../src/utils/handConstraints";
import { detectRigProfile } from "../src/utils/rigProfiles";
import type { MotionFile } from "../src/types/contract";

const [glbPath, motionDir, output] = process.argv.slice(2);
if (!glbPath || !motionDir) throw Error("usage: <avatar.glb> <motions dir> [out.json]");
const glb = fs.readFileSync(glbPath);
const model = JSON.parse(glb.subarray(20, 20 + glb.readUInt32LE(12)).toString());
function makeRig() {
  const joints = new Set<number>(model.skins.flatMap((skin: { joints: number[] }) => skin.joints));
  const nodes = model.nodes.map((n: { name?: string; translation?: number[]; rotation?: number[]; scale?: number[] }, i: number) => {
    const o = joints.has(i) ? new THREE.Bone() : new THREE.Object3D();
    o.name = n.name || "";
    if (n.translation) o.position.fromArray(n.translation);
    if (n.rotation) o.quaternion.fromArray(n.rotation);
    if (n.scale) o.scale.fromArray(n.scale);
    return o;
  });
  model.nodes.forEach((n: { children?: number[] }, i: number) => n.children?.forEach((j) => nodes[i].add(nodes[j])));
  const rig = new THREE.Group();
  model.scenes[model.scene || 0].nodes.forEach((i: number) => rig.add(nodes[i]));
  rig.updateMatrixWorld(true);
  return rig;
}

if (process.argv.includes("--bind")) {
  // Bind-pose geometry of the joint frames: anatomical flexion / abduction of the relaxed hand.
  const rig = makeRig(), hands = new HandConstraints(rig, detectRigProfile(rig));
  for (const j of hands.joints) console.log(j.name.padEnd(20), j.type.padEnd(9), "flex0", (j.flex0 / Math.PI * 180).toFixed(1).padStart(6), "abd0", (j.abd0 / Math.PI * 180).toFixed(1).padStart(6), "p.t", j.p.dot(j.t).toFixed(2));
  process.exit(0);
}
if (process.argv.includes("--worst")) {
  // Times (25 fps) of the largest total excess beyond the limits per motion, without constraints: where to look.
  for (const file of fs.readdirSync(motionDir).filter((f) => f.endsWith(".json")).sort()) {
    const motion: MotionFile = JSON.parse(fs.readFileSync(path.join(motionDir, file), "utf8"));
    const rig = makeRig(), profile = detectRigProfile(rig), meter = new HandConstraints(rig, profile);
    const retarget = new HumanoidRetargeter(rig, profile, { handConstraints: false });
    const rows: { t: number; excess: number; worst: string }[] = [];
    for (let t = 0; t <= motion.duration_ms; t += 40) {
      retarget.applyMotion(motion, t);
      let excess = 0, worst = "", top = 0;
      for (const j of meter.joints) {
        const a = meter.angles(j, j.bone.quaternion), L = j.limit, r = THREE.MathUtils.radToDeg;
        const e = Math.max(0, r(a.flex) - L.flex[1], L.flex[0] - r(a.flex)) + Math.max(0, Math.abs(r(a.abd)) - Math.max(-L.abd[0], L.abd[1])) + Math.max(0, Math.abs(r(a.twist)) - L.twist);
        excess += e;
        if (e > top) { top = e; worst = `${j.name} ${r(a.flex).toFixed(0)}/${r(a.abd).toFixed(0)}/${r(a.twist).toFixed(0)}`; }
      }
      rows.push({ t, excess, worst });
    }
    rows.sort((x, y) => y.excess - x.excess);
    console.log(file, rows.slice(0, 3).map((r) => `${r.t}ms ${r.excess.toFixed(0)}deg (${r.worst})`).join(" | "));
  }
  process.exit(0);
}
type Tally = Record<string, number>;
const files = fs.readdirSync(motionDir).filter((f) => f.endsWith(".json")).sort();
const groups: Record<string, Record<string, Tally>> = {};
const extremes: Record<string, Record<string, number>> = {};
for (const mode of (process.env.HANDS_MODES ?? "before,after").split(",") as ("before" | "after")[]) for (const stage of ["retarget", "display"] as const) {
  for (const file of files) {
    const motion: MotionFile = JSON.parse(fs.readFileSync(path.join(motionDir, file), "utf8"));
    const group = file.split("_")[0];
    const rig = makeRig(), profile = detectRigProfile(rig);
    const meter = new HandConstraints(rig, profile);
    const retarget = new HumanoidRetargeter(rig, profile, { handConstraints: mode === "after" });
    const performer = stage === "display" ? new PerformanceLayer(rig, profile, retarget.hands) : null;
    // Elbow frames at bind: forearm axis and the hinge axis (shoulder -> elbow -> wrist plane), forearm-local.
    const elbows: Record<string, { local: THREE.Quaternion; axis: THREE.Vector3; hinge: THREE.Vector3 }> = {};
    for (const side of ["Left", "Right"]) {
      const get = (n: string) => rig.getObjectByName(profile.bones?.[side + n] ?? side + n)!;
      const [s, e, w] = ["Arm", "ForeArm", "Hand"].map((n) => get(n).getWorldPosition(new THREE.Vector3()));
      const inv = get("ForeArm").getWorldQuaternion(new THREE.Quaternion()).invert();
      const u = e.clone().sub(s).normalize(), f = w.clone().sub(e).normalize();
      elbows[side] = { local: get("ForeArm").quaternion.clone(), axis: f.clone().applyQuaternion(inv).normalize(), hinge: u.clone().cross(f).normalize().applyQuaternion(inv) };
    }
    const key = `${mode}.${stage}`;
    const tally = ((groups[key] ??= {})[group] ??= {});
    const ext = (extremes[`${key}.${group}`] ??= {});
    for (let t = 0; t <= motion.duration_ms; t += 1000 / 60) {
      retarget.applyMotion(motion, t);
      performer?.update(t, true);
      countViolations(meter, tally);
      // Elbow (a hinge): forearm twist and sideways bend relative to the bind elbow axis, beyond 10 degrees.
      for (const side of ["Left", "Right"]) {
        const fore = rig.getObjectByName(profile.bones?.[side + "ForeArm"] ?? side + "ForeArm")!, bind = elbows[side];
        const delta = bind.local.clone().invert().multiply(fore.quaternion);
        const tw = delta.x * bind.axis.x + delta.y * bind.axis.y + delta.z * bind.axis.z;
        const twist = Math.abs(2 * Math.atan2(Math.abs(tw), Math.abs(delta.w)));
        const swing = delta.clone().multiply(new THREE.Quaternion(bind.axis.x * tw, bind.axis.y * tw, bind.axis.z * tw, delta.w).normalize().invert());
        const off = new THREE.Vector3(swing.x, swing.y, swing.z);
        const side_ = off.lengthSq() > 1e-12 ? 2 * Math.asin(Math.min(1, off.clone().sub(bind.hinge.clone().multiplyScalar(off.dot(bind.hinge))).length())) : 0;
        tally["elbow.samples"] = (tally["elbow.samples"] ?? 0) + 1;
        if (twist > 10 * Math.PI / 180) tally["elbow.twist"] = (tally["elbow.twist"] ?? 0) + 1;
        if (side_ > 10 * Math.PI / 180) tally["elbow.abduction"] = (tally["elbow.abduction"] ?? 0) + 1;
        ext["elbow.twist"] = Math.max(ext["elbow.twist"] ?? 0, THREE.MathUtils.radToDeg(twist));
        ext["elbow.abd"] = Math.max(ext["elbow.abd"] ?? 0, THREE.MathUtils.radToDeg(side_));
      }
      for (const j of meter.joints) {
        const a = meter.angles(j, j.bone.quaternion);
        for (const [name, v] of [["flexMax", a.flex], ["flexMin", -a.flex], ["abd", Math.abs(a.abd)], ["twist", Math.abs(a.twist)]] as const)
          ext[`${j.type}.${name}`] = Math.max(ext[`${j.type}.${name}`] ?? -Infinity, THREE.MathUtils.radToDeg(v));
      }
    }
  }
}
const types = ["elbow", "wrist", "mcp", "pip", "dip", "thumbCmc", "thumbMcp", "thumbIp"];
const kinds = ["flexion", "hyperextension", "abduction", "twist"];
for (const [key, byGroup] of Object.entries(groups)) {
  console.log(`\n== ${key} (joint-samples beyond the limits, % of samples of that joint type) ==`);
  for (const [group, tally] of Object.entries(byGroup)) {
    const cells = types.map((type) => {
      const n = tally[`${type}.samples`] || 1;
      return `${type}: ` + kinds.map((k) => `${k.slice(0, 5)} ${tally[`${type}.${k}`] ?? 0} (${(100 * (tally[`${type}.${k}`] ?? 0) / n).toFixed(1)}%)`).join(", ");
    });
    const total = types.reduce((s, type) => s + kinds.reduce((a, k) => a + (tally[`${type}.${k}`] ?? 0), 0), 0);
    console.log(`[${group}] total ${total}\n  ` + cells.join("\n  "));
  }
}
console.log("\n== extremes (degrees; flexMin = deepest hyperextension) ==");
for (const [key, ext] of Object.entries(extremes)) if (key.includes(".display.")) console.log(key, JSON.stringify(Object.fromEntries(Object.entries(ext).map(([k, v]) => [k, Math.round(v)]))));
if (output) fs.writeFileSync(output, JSON.stringify({ groups, extremes }, null, 1));
