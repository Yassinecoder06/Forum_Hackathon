/** Local real-source engineering diagnostic; no linguistic accuracy score. */
import fs from "node:fs";
import path from "node:path";
import { performance } from "node:perf_hooks";
import * as THREE from "three";
import { HumanoidRetargeter } from "../src/utils/retargetHumanoid";
import { interpolateJointPositions } from "../src/utils/motionInterpolator";
import { browWeights, faceCalibration } from "../src/utils/avatarPlayback";
import { handContact, type Capsule } from "../src/utils/handContact";
import type { MotionFile } from "../src/types/contract";

const root = path.resolve(process.cwd(), "..");
const output = process.argv[2] || path.join(root, "artifacts/avatar-review/measurements.json");
const baseline = process.argv.includes("--baseline");
const load = (p: string) => JSON.parse(fs.readFileSync(p, "utf8"));
// Read the real GLB's skeletal graph without textures, GPU, or synthetic rig fallback.
const glb = fs.readFileSync(path.join(root, ".runtime/avatar-candidates/Noa.glb"));
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
const library = load(path.join(root, ".runtime/library.json"));
const results = [];
for (const id of ["380", "490", "905", "535", "926", "988"]) {
  const row = library.find((r: { source_id: string }) => r.source_id === "DICTASIGN_PORTAL_" + id);
  if (!row) throw Error("Missing genuine source " + id);
  const motion: MotionFile = load(row.motion.path), rig = makeRig(), retarget = new HumanoidRetargeter(rig);
  const bones: THREE.Bone[] = [];
  rig.traverse((o) => { if (o instanceof THREE.Bone && /Hand(Thumb|Index|Middle|Ring|Pinky)[123]$/.test(o.name)) bones.push(o); });
  const rotations: THREE.Quaternion[][] = [], intervals: number[] = [];
  let noncontactSamples = 0, noncontactPenetrationSum = 0, noncontactMaxPenetration = 0, allMaxPenetration = 0;
  const render = (t: number) => baseline ? retarget.apply(interpolateJointPositions(motion, t)) : retarget.applyMotion(motion, t);
  const began = performance.now();
  for (let t = 0; t <= motion.duration_ms; t += 40) {
    const now = performance.now();
    render(t);
    intervals.push(performance.now() - now);
    rotations.push(bones.map((b) => b.quaternion.clone()));
    const source = interpolateJointPositions(motion, t);
    const capsules = (side: string): Capsule[] => ["Thumb", "Index", "Middle", "Ring", "Pinky"].flatMap((finger) => [1, 2, 3].flatMap((i) => {
      const a = rig.getObjectByName(`${side}Hand${finger}${i}`), b = rig.getObjectByName(`${side}Hand${finger}${i + 1}`);
      const sa = source[`${side.toLowerCase()}_${finger.toLowerCase()}_${i}`], sb = source[`${side.toLowerCase()}_${finger.toLowerCase()}_${i + 1}`];
      return a && b && sa && sb ? [{ a: a.getWorldPosition(new THREE.Vector3()), b: b.getWorldPosition(new THREE.Vector3()), sourceA: new THREE.Vector3(...sa), sourceB: new THREE.Vector3(...sb) }] : [];
    }));
    const left = capsules("Left"), right = capsules("Right");
    if (left.length === 15 && right.length === 15) {
      const contact = handContact(left, right);
      allMaxPenetration = Math.max(allMaxPenetration, contact.maxPenetration);
      if (!contact.contact) {
        noncontactSamples++;
        noncontactPenetrationSum += contact.maxPenetration;
        noncontactMaxPenetration = Math.max(noncontactMaxPenetration, contact.maxPenetration);
      }
    }
  }
  const changes: number[] = [];
  for (let i = 1; i < rotations.length; i++) rotations[i].forEach((q, j) => changes.push(q.angleTo(rotations[i - 1][j]) * 180 / Math.PI));
  const brow = motion.frames.flatMap((f) => f.face ? [f.face.brow_left, f.face.brow_right].filter(Number.isFinite) : []);
  const saturatedBefore = brow.filter((v) => (v - .055) * 3 >= 1).length;
  const neutral = faceCalibration(motion);
  const mapped = motion.frames.flatMap((f) => f.face ? ["brow_left", "brow_right"].map((k) => browWeights(f.face![k], neutral?.[k as "brow_left" | "brow_right"])) : []);
  const saturatedAfter = mapped.filter((v) => v.up >= 1).length;
  // Exercise the same pose after out-of-order seeking.
  render(333);
  const pose = bones.map((b) => b.quaternion.toArray());
  render(0); render(777); render(333);
  if (JSON.stringify(pose) !== JSON.stringify(bones.map((b) => b.quaternion.toArray()))) throw Error("Non-deterministic seeking");
  // Native data remain unchanged after rendering.
  if (JSON.stringify(motion) !== JSON.stringify(load(row.motion.path))) throw Error("Source mutation");
  changes.sort((a, b) => a - b); intervals.shift(); intervals.sort((a, b) => a - b);
  results.push({ source_id: row.source_id, source_video_sha256: row.provenance.source_sha256,
    frames: motion.frames.length, elapsed_ms: performance.now() - began,
    prepare: baseline ? null : retarget.diagnostics,
    steady_sample_p95_ms: intervals[Math.floor((intervals.length - 1) * .95)],
    finger_local_rotation_delta_40ms_degrees: { max: Math.max(...changes), p95: changes[Math.floor((changes.length - 1) * .95)] },
    hand_capsule_proxy_meters: { radius: .008, noncontact_samples: noncontactSamples, noncontact_mean_penetration: noncontactSamples ? noncontactPenetrationSum / noncontactSamples : null, noncontact_max_penetration: noncontactMaxPenetration, all_max_penetration: allMaxPenetration },
    brows: { samples: brow.length, saturated_before: saturatedBefore, saturated_after: saturatedAfter,
      engineering_rest_reference: neutral, max_up_after: Math.max(...mapped.map((v) => v.up)), max_down_after: Math.max(...mapped.map((v) => v.down)) },
    deterministic_seek: true, source_unchanged: true });
}
fs.mkdirSync(path.dirname(output), { recursive: true });
fs.writeFileSync(output, JSON.stringify({ kind: "Real Noa rig + existing source captures; engineering diagnostics, not LSF validation", baseline, results }, null, 2));
console.log(JSON.stringify(results, null, 2));

