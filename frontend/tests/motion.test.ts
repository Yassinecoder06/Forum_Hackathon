import assert from "node:assert/strict";
import * as THREE from "three";
import { stitchAnimationPlan, libraryPreview } from "../src/utils/planStitcher";
import { HumanoidRetargeter } from "../src/utils/retargetHumanoid";
import { detectRigProfile } from "../src/utils/rigProfiles";
import { MetaHumanFaceDriver } from "../src/utils/metahumanFace";
import { browWeights, faceCalibration, sampleFace, stabilizeRotations, hasHand } from "../src/utils/avatarPlayback";
import { handContact, segmentDistance } from "../src/utils/handContact";
import { interpolateJointPositions } from "../src/utils/motionInterpolator";
import type { AnimationPlan, MotionFile } from "../src/types/contract";
const makeClip = (x: number): MotionFile => ({
  rig_version: "signaculture-v1",
  fps: 30,
  duration_ms: 100,
  joints: ["hips"],
  synthetic: true,
  frames: [
    { time_ms: 0, joints: { hips: [x, 1, 0] } },
    { time_ms: 100, joints: { hips: [x + 0.1, 1, 0] } },
  ],
});
const handClip = makeClip(0);
handClip.frames[0].joints.right_wrist = [.4, 1.2, .3];
handClip.frames[0].joints.right_index_1 = [.44, 1.22, .29];
handClip.frames[0].joints.right_index_2 = [.46, 1.24, .28];
const originalHand = interpolateJointPositions(handClip, 0);
for (const energyFactor of [0, 1]) {
  const shiftedHand = interpolateJointPositions(handClip, 0, {
    expressiveModulation: true, energyFactor,
  });
  assert.notDeepEqual(shiftedHand.right_wrist, originalHand.right_wrist);
  for (const name of ["right_index_1", "right_index_2"]) {
    for (const axis of [0, 1, 2]) {
      const before = originalHand[name][axis] - originalHand.right_wrist[axis];
      const after = shiftedHand[name][axis] - shiftedHand.right_wrist[axis];
      assert(Math.abs(before - after) < 1e-9, "Audio energy preserves wrist-relative finger positions");
    }
  }
}
const a = makeClip(0),
  b = makeClip(1),
  before = JSON.stringify([a, b]);
const plan = {
  transition_ms: 333,
  segments: [{ motion_id: "a" }, { motion_id: "b" }],
  timing_mode: "original",
} as AnimationPlan;
const motion = stitchAnimationPlan(
  plan,
  new Map([
    ["a", a],
    ["b", b],
  ]),
);
assert.equal(motion.duration_ms, 533);
assert.deepEqual(
  motion.frames.find((f) => f.time_ms === 100)?.joints,
  a.frames[1].joints,
  "Keep the last frame of sign A",
);
assert.deepEqual(
  motion.frames.find((f) => f.time_ms === 433)?.joints,
  b.frames[0].joints,
  "Keep the first frame of sign B",
);
assert(
  motion.frames.every(
    (f, i) => i === 0 || f.time_ms > motion.frames[i - 1].time_ms,
  ),
);
assert(
  motion.frames
    .filter((f) => f.time_ms > 100 && f.time_ms < 433)
    .every((f) => f.joints.hips[0] > 0.1 && f.joints.hips[0] < 1),
);
assert.equal(
  JSON.stringify([a, b]),
  before,
  "Playback must not rewrite source captures",
);
const audioPlan = {
  ...plan,
  timing_mode: "audio_voiced",
  audio_features: {
    duration_ms: 1200,
    frames: [
      { start_ms: 0, end_ms: 400, is_pause: false },
      { start_ms: 400, end_ms: 800, is_pause: true },
      { start_ms: 800, end_ms: 1200, is_pause: false },
    ],
  },
} as AnimationPlan;
const warped = stitchAnimationPlan(
  audioPlan,
  new Map([
    ["a", a],
    ["b", b],
  ]),
);
assert.equal(warped.duration_ms, 1200);
assert.deepEqual(
  warped.frames.find((f) => f.time_ms === 400)?.joints,
  warped.frames.find((f) => f.time_ms === 760)?.joints,
  "Hold the same source position through an audio pause",
);
const root = new THREE.Group(),
  bones: THREE.Bone[] = [];
function bone(name: string, parent: THREE.Object3D, position: number[]) {
  const b = new THREE.Bone();
  b.name = name;
  b.position.fromArray(position);
  parent.add(b);
  bones.push(b);
  return b;
}
const neck = bone("Neck", root, [0, 1.5, 0]);
bone("Head", neck, [0, 0.12, 0]);
for (const side of ["Left", "Right"]) {
  const sign = side === "Left" ? 1 : -1;
  const arm = bone(side + "Arm", root, [sign * 0.18, 1.45, 0]);
  const fore = bone(side + "ForeArm", arm, [0, -0.28, 0]);
  const hand = bone(side + "Hand", fore, [0, -0.26, 0]);
  for (const [i, f] of [
    "Thumb",
    "Index",
    "Middle",
    "Ring",
    "Pinky",
  ].entries()) {
    const b1 = bone(side + "Hand" + f + "1", hand, [
      (i - 2) * 0.015,
      -0.05,
      0.01,
    ]);
    const b2 = bone(side + "Hand" + f + "2", b1, [0, -0.025, 0]);
    const b3 = bone(side + "Hand" + f + "3", b2, [0, -0.02, 0]);
    bone(side + "Hand" + f + "4", b3, [0, -0.015, 0]);
  }
}
const lengths = bones.map((b) => b.position.length()),
  retarget = new HumanoidRetargeter(root);
const joints = {
  hips: [0, 1, 0],
  neck: [0, 1.5, 0],
  left_shoulder: [0.18, 1.45, 0],
  right_shoulder: [-0.18, 1.45, 0],
  left_elbow: [0.35, 1.3, 0.1],
  right_elbow: [-0.35, 1.3, 0.1],
  left_wrist: [0.1, 1.55, 0.3],
  right_wrist: [-0.1, 1.1, 0.2],
} as Record<string, [number, number, number]>;
retarget.apply(joints);
const q = bones.map((b) => b.quaternion.toArray());
retarget.apply({ ...joints, left_wrist: [0.4, 1.3, 0.2] });
retarget.apply(joints);
assert.deepEqual(
  bones.map((b) => b.quaternion.toArray()),
  q,
  "Seeking back must restore exactly the same rig pose",
);
assert.deepEqual(
  bones.map((b) => b.position.length()),
  lengths,
  "Retargeting cannot stretch any bone",
);
assert(bones.every((b) => b.quaternion.toArray().every(Number.isFinite)));
console.log(
  "Motion checks passed: preserved sign endpoints, continuous transition time, pause hold, fixed bone lengths and deterministic IK seeking.",
);

// Authored software checks, not signed-language ground truth.
assert.deepEqual(browWeights(.38), { up: 0, down: 0 });
assert(browWeights(.44).up > 0 && browWeights(.44).up < .2, "Legacy brow ratios must not saturate the mesh");
assert(browWeights(.32).down > 0);
assert.deepEqual(browWeights(NaN), { up: 0, down: 0 });
assert.equal(browWeights(.44, .44).up, 0, "Explicit neutral reference overrides conservative fallback");
assert.equal(browWeights(.01).down, .18, "Uncalibrated recordings use restrained fallback expression");
assert.equal(faceCalibration({ ...a, frames: [{ time_ms: 0, joints, face: { brow_left: .1, brow_right: .2 } }] }), undefined, "Do not invent neutral references for unknown recordings");
const faceClip = { ...a, frames: [{ time_ms: 0, joints: {}, face: { brow_left: .34 } }, { time_ms: 100, joints: {}, face: { brow_left: .44 } }] };
assert.equal(sampleFace(faceClip, 50)?.brow_left, .39);
const missingFace = { ...faceClip, frames: [{ time_ms: 0, joints: {}, face: { brow_left: .5 } }, { time_ms: 300, joints: {}, face: null }] };
assert(sampleFace(missingFace, 60)!.brow_left > .38 && sampleFace(missingFace, 60)!.brow_left < .5);
assert.equal(sampleFace(missingFace, 180), null, "Do not hold a missing face indefinitely");
const rotation = (angle: number) => new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(1, 0, 0), angle);
const raw = [0, 0, 2, 0, 0].map((angle, i) => ({ time: i * 40, valid: true, rotations: [rotation(angle)] }));
const snapshot = JSON.stringify(raw);
const stable = stabilizeRotations(raw, [rotation(0)]);
assert.equal(stable.rejected, 1);
assert(stable.frames[2].rotations[0].angleTo(rotation(0)) < 1e-8);
assert.equal(JSON.stringify(raw), snapshot, "Raw rotation observations stay unchanged");
const gap = stabilizeRotations([
  { time: 0, valid: true, rotations: [rotation(.5)] },
  { time: 50, valid: false, rotations: [rotation(0)] },
  { time: 100, valid: true, rotations: [rotation(.7)] },
  { time: 600, valid: false, rotations: [rotation(0)] },
], [rotation(0)]);
assert.equal(gap.states[1], "interpolated");
assert.equal(gap.states[3], "untracked");
assert(gap.frames[1].rotations[0].angleTo(rotation(.6)) < 1e-7);
const sustained = stabilizeRotations([0, 2, 2, 2, 0].map((angle, i) => ({ time: i * 40, valid: true, rotations: [rotation(angle)] })), [rotation(0)]);
assert.equal(sustained.rejected, 0, "Do not reject a sustained handshape as an isolated spike");
const longGap = stabilizeRotations([0, 100, 200, 300, 400].map((time, i) => ({ time, valid: i === 0 || i === 4, rotations: [rotation(i === 0 ? 1 : 2)] })), [rotation(0)]);
assert.equal(longGap.states[2], "untracked");
assert(longGap.frames[2].rotations[0].angleTo(rotation(0)) < 1e-8, "Long-gap midpoint reaches rest continuously");
const point = (x: number, y = 0, z = 0) => new THREE.Vector3(x, y, z);
assert.equal(segmentDistance(point(0), point(1), point(.5, -1), point(.5, 1)).distance, 0);
assert.equal(segmentDistance(point(0), point(0), point(1), point(1)).distance, 1);
const leftCapsule = { a: point(0), b: point(0, 1), sourceA: point(-.1), sourceB: point(-.1, 1) };
const rightCapsule = { a: point(.005), b: point(.005, 1), sourceA: point(.1), sourceB: point(.1, 1) };
const separated = handContact([leftCapsule], [rightCapsule]);
assert(separated.correction.length() > 0 && separated.correction.length() <= .015);
const touching = handContact([leftCapsule], [{ ...rightCapsule, a: point(.013), b: point(.013, 1), sourceA: point(-.09), sourceB: point(-.09, 1) }]);
assert(touching.contact && touching.correction.length() === 0, "Preserve surface contact");
assert(handContact([leftCapsule], [{ ...rightCapsule, sourceA: point(-.09), sourceB: point(-.09, 1) }]).correction.length() > 0, "Deep intersection is corrected even during contact");
assert(handContact([{ ...leftCapsule, sourceObserved: false, radius: .022 }], [{ ...rightCapsule, sourceObserved: false, radius: .022 }]).correction.length() > 0, "Untracked rendered palms still collide");
assert(handContact([{ ...leftCapsule, sourceObserved: false }], [{ ...leftCapsule, sourceObserved: false }]).correction.length() > 0, "Coincident capsules need a deterministic separating normal");
const forearm = { ...leftCapsule, kind: "forearm" as const, radius: .028 };
assert(handContact([forearm], [forearm]).maxPenetration === 0, "Forearm crossings are not hand collisions");
assert(handContact([forearm], [{ ...rightCapsule, kind: "hand" as const }]).maxRawPenetration > 0, "Hands must be checked against the opposite forearm");
// Whole-track seeking must be independent of render history, including a gap.
const seekClip = { ...a, frames: [0, 40, 80].map((time_ms, i) => ({ time_ms, joints: { ...joints, left_wrist: [0.1 + i * .02, 1.5, .3] as [number, number, number] } })) };
retarget.applyMotion(seekClip, 20);
const seekPose = bones.map((b) => b.quaternion.toArray());
retarget.applyMotion(seekClip, 80);
retarget.applyMotion(seekClip, 20);
assert.deepEqual(bones.map((b) => b.quaternion.toArray()), seekPose);
assert.deepEqual(bones.map((b) => b.position.length()), lengths);
const calibrated = stitchAnimationPlan(plan, new Map([
  ["a", { ...a, face_calibration: { brow_left: .4, brow_right: .41 }, frames: a.frames.map((f) => ({ ...f, face: { brow_left: .4, brow_right: .41 } })) }],
  ["b", b],
]));
assert.equal(calibrated.frames[0].face?.brow_left_neutral, .4, "Keep explicit calibration when composing clips");
assert(calibrated.frames.some((f) => f.playback_origin === "transition"));
const fullHand = { ...joints };
for (const side of ["left", "right"]) for (const [i, finger] of ["thumb", "index", "middle", "ring", "pinky"].entries()) {
  const x = (side === "left" ? .1 : -.1) + (i - 2) * .02;
  fullHand[`${side}_${finger}_1`] = [x, 1.6, .3];
  fullHand[`${side}_${finger}_2`] = [x, 1.63, .3];
  fullHand[`${side}_${finger}_3`] = [x, 1.60, .3];
  fullHand[`${side}_${finger}_4`] = [x, 1.59, .29];
}
retarget.apply(fullHand);
for (const b of bones.filter((b) => /Hand(Index|Middle|Ring|Pinky)[23]$/.test(b.name)))
  assert(b.quaternion.angleTo(new THREE.Quaternion()) <= 1.900001, "Constrain extreme software-test finger curls without stretching bones");
console.log("Avatar checks passed: brows, immutable spike filtering, gap states, contact preservation and cached seeking.");

// --- Signing-core windows and continuous bridges -------------------------------------------
{
const walk = (x0: number, v: number, finger = 0.05): MotionFile => ({
  rig_version: "signaculture-v1", fps: 10, duration_ms: 400, joints: ["right_wrist"], synthetic: true,
  frames: [0, 100, 200, 300, 400].map((t) => ({ time_ms: t, joints: {
    right_wrist: [x0 + v * t, 1.2, 0.3], right_index_1: [x0 + v * t + finger, 1.2, 0.3] } })),
});
const core = { transition_ms: 333, timing_mode: "original", segments: [
  { motion_id: "p", clip_start_ms: 100, clip_end_ms: 300 },
  { motion_id: "q", clip_start_ms: 100, clip_end_ms: 300 },
] } as AnimationPlan;
const v = 0.001; // 1 m/s, in metres per millisecond
// A's window ends at x=0.3; B's window (from t=100) starts where uniform motion arrives 333 ms later.
const p = walk(0, v), q = walk(0.3 + 0.333 - 0.1, v);
const cored = stitchAnimationPlan(core, new Map([["p", p], ["q", q]]));
assert.equal(cored.duration_ms, 200 + 333 + 200, "Only each clip's window is played");
assert.deepEqual(cored.frames[0].joints, p.frames[1].joints, "A window starts on its captured frame");
assert.deepEqual(cored.frames.find((f) => f.time_ms === 200)?.joints, p.frames[3].joints, "A window ends on its captured frame");
assert.deepEqual(cored.frames.find((f) => f.time_ms === 533)?.joints, q.frames[1].joints, "B window starts on its captured frame");
const bridge = cored.frames.filter((f) => f.playback_origin === "transition");
const speeds = bridge.slice(1).map((f, i) => (f.joints.right_wrist[0] - bridge[i].joints.right_wrist[0]) / (f.time_ms - bridge[i].time_ms));
assert(speeds.every((s) => s > 0.6 * v && s < 1.4 * v), "Arms keep moving through the join instead of stopping at both ends");
assert(bridge.every((f) => Math.abs(f.joints.right_index_1[0] - f.joints.right_wrist[0] - 0.05) < 1e-9),
  "Fingers move with their wrist; the hand shape does not trail behind");
const morph = stitchAnimationPlan(core, new Map([["p", walk(0, 0, 0.02)], ["q", walk(0.2, 0, 0.08)]]));
const offsets = morph.frames.filter((f) => f.playback_origin === "transition").map((f) => f.joints.right_index_1[0] - f.joints.right_wrist[0]);
assert(offsets.every((o, i) => o > 0.02 && o < 0.08 && (i === 0 || o >= offsets[i - 1])), "Hand shape morphs monotonically between signs");
const still = stitchAnimationPlan(core, new Map([["p", walk(0, 0)], ["q", walk(0.2, 0)]]));
assert(still.frames.every((f) => f.joints.right_wrist[0] >= 0 && f.joints.right_wrist[0] <= 0.2), "No overshoot between resting endpoints");
console.log("Stitching checks passed: signing-core windows, continuous arm bridges, wrist-relative hand shapes.");
}

// Reversible quality metadata: software contracts, not LSF ground truth.
const rejectedHand = {time_ms:40,joints:fullHand,hand_tracking:{left:{observed:true,source:"actual detector"}},playback_quality:{left_hand_rejected:true}};
assert(!hasHand(rejectedHand,"left"),"Rejected tracking is interpolated only for playback");
assert(hasHand({...rejectedHand,playback_quality:undefined},"left"),"Keep the original observation available");
const libraryClip={...a,playback_window:{start_ms:0,end_ms:100}};
const libraryRaw=JSON.stringify(libraryClip);libraryPreview(libraryClip);
assert.equal(JSON.stringify(libraryClip),libraryRaw,"Preview windows never mutate original source data");

// MetaHuman profile: same geometry as the synthetic RPM rig above, UE5 joint names, corrective
// joints inserted as FIRST children (so "first child" would aim at the wrong bone), tip joints
// named <finger>_04_<side>. Authored software checks, not signed-language ground truth.
{
  const names: Record<string, string> = { Neck: "neck_01", Head: "head" };
  for (const [side, s] of [["Left", "l"], ["Right", "r"]] as const) {
    Object.assign(names, { [side + "Arm"]: `upperarm_${s}`, [side + "ForeArm"]: `lowerarm_${s}`, [side + "Hand"]: `hand_${s}` });
    for (const f of ["Thumb", "Index", "Middle", "Ring", "Pinky"])
      for (let i = 1; i <= 4; i++) names[`${side}Hand${f}${i}`] = `${f.toLowerCase()}_0${i}_${s}`;
  }
  const mhRoot = new THREE.Group(), mhBones: THREE.Bone[] = [];
  const clone = (src: THREE.Object3D, parent: THREE.Object3D) => {
    for (const child of src.children) {
      if (!(child instanceof THREE.Bone)) continue;
      const helper = new THREE.Bone();
      helper.name = names[child.name] + "_correctiveRoot";
      helper.position.set(0.02, 0.01, 0.03);
      const b = new THREE.Bone();
      b.name = names[child.name];
      b.position.copy(child.position);
      parent.add(helper);
      parent.add(b);
      mhBones.push(b);
      clone(child, b);
    }
  };
  // fresh RPM rig at bind pose for comparison
  const rpmRoot = new THREE.Group();
  const rpmNeck = new THREE.Bone(); rpmNeck.name = "Neck"; rpmNeck.position.set(0, 1.5, 0); rpmRoot.add(rpmNeck);
  const rpmHead = new THREE.Bone(); rpmHead.name = "Head"; rpmHead.position.set(0, 0.12, 0); rpmNeck.add(rpmHead);
  for (const b of bones.filter((x) => x.parent === root && x.name !== "Neck")) {
    const copy = b.clone(true);
    copy.traverse((o) => o.quaternion.identity());
    rpmRoot.add(copy);
  }
  clone(rpmRoot, mhRoot);
  const mh = new HumanoidRetargeter(mhRoot);
  assert.equal(mh.profile.id, "metahuman");
  assert.equal(detectRigProfile(rpmRoot).id, "rpm");
  const rpm = new HumanoidRetargeter(rpmRoot);
  const mhLengths = mhBones.map((b) => b.position.length());
  const motionJoints = { ...joints, ...Object.fromEntries(["left", "right"].flatMap((side) => {
    const w = joints[`${side}_wrist`];
    return ["thumb", "index", "middle", "ring", "pinky"].flatMap((f, k) => [1, 2, 3, 4].map((i) =>
      [`${side}_${f}_${i}`, [w[0] + (k - 2) * 0.015, w[1] - 0.05 - 0.02 * i, w[2] + 0.01 * i] as [number, number, number]]));
  })) } as Record<string, [number, number, number]>;
  rpm.apply(motionJoints);
  mh.apply(motionJoints);
  rpmRoot.updateMatrixWorld(true);
  mhRoot.updateMatrixWorld(true);
  for (const canonical of ["LeftHand", "RightHand", "LeftForeArm", "LeftHandIndex4", "RightHandThumb4"]) {
    const a = rpmRoot.getObjectByName(canonical)!.getWorldPosition(new THREE.Vector3());
    const b = mhRoot.getObjectByName(names[canonical])!.getWorldPosition(new THREE.Vector3());
    assert(a.distanceTo(b) < 1e-6, `MetaHuman ${names[canonical]} follows the same IK as ${canonical}`);
  }
  const pose = mhBones.map((b) => b.quaternion.toArray());
  mh.apply({ ...motionJoints, left_wrist: [0.4, 1.3, 0.2] });
  mh.apply(motionJoints);
  assert.deepEqual(mhBones.map((b) => b.quaternion.toArray()), pose, "MetaHuman seeking is deterministic");
  assert.deepEqual(mhBones.map((b) => b.position.length()), mhLengths, "MetaHuman retargeting cannot stretch bones");
  const helpers: THREE.Object3D[] = [];
  mhRoot.traverse((o) => { if (o.name.endsWith("_correctiveRoot")) helpers.push(o); });
  assert(helpers.every((o) => o.quaternion.equals(new THREE.Quaternion())), "Corrective joints are never driven");

  // Body layer: LSF-engine motions move the neck/head (and torso/clavicles when the rig has them) from the
  // source's deviation from the clip's median pose; captured clips keep the original arms-and-hands behaviour.
  {
    const base = { ...motionJoints, head: [0, 1.62, 0.09] as [number, number, number] };
    const nod = { ...base, neck: [0.03, 1.5, 0] as [number, number, number], head: [0.06, 1.58, 0.09] as [number, number, number] };
    const clip = (engine: boolean): MotionFile => ({
      rig_version: "signaculture-v1", fps: 25, duration_ms: 160, joints: ["neck", "head"], synthetic: true,
      ...(engine ? { capture_method: "lsf-engine-c0-2d-bone-length-lift" } : {}),
      frames: [0, 40, 80, 120, 160].map((t) => ({ time_ms: t, joints: t === 80 ? nod : base })),
    } as MotionFile);
    const headBone = mhRoot.getObjectByName("head")!, neckBone = mhRoot.getObjectByName("neck_01")!;
    const bindHead = headBone.quaternion.clone(), bindNeck = neckBone.quaternion.clone();
    const engineRig = new HumanoidRetargeter(mhRoot);
    engineRig.applyMotion(clip(true), 80);
    assert(headBone.quaternion.angleTo(bindHead) > 0.02 || neckBone.quaternion.angleTo(bindNeck) > 0.02, "Engine motion turns the neck/head");
    engineRig.applyMotion(clip(true), 0);
    assert(headBone.quaternion.angleTo(bindHead) < 1e-6 && neckBone.quaternion.angleTo(bindNeck) < 1e-6, "At the median pose the head is at rest");
    const capturedRig = new HumanoidRetargeter(mhRoot);
    capturedRig.applyMotion(clip(false), 80);
    assert(headBone.quaternion.angleTo(bindHead) < 1e-6 && neckBone.quaternion.angleTo(bindNeck) < 1e-6, "Captured clips keep the head at bind (unchanged behaviour)");
    console.log("Body layer checks passed: engine motions move the head, captured clips unchanged.");
  }

  // Face joints: no blendshapes, the driver moves FACIAL_* joints and resets them every call.
  const face = new THREE.Group();
  const facial = (name: string, parent: THREE.Object3D, p: number[]) => { const b = new THREE.Bone(); b.name = name; b.position.fromArray(p); parent.add(b); return b; };
  const jaw = facial("FACIAL_C_Jaw", face, [0, 1.62, 0.03]);
  const chin = facial("FACIAL_C_Chin", jaw, [0, -0.07, 0.06]);
  const lowerLip = facial("FACIAL_C_LowerLipRotation", face, [0, 1.62, 0.03]);
  const lid = facial("FACIAL_L_EyelidUpperA", face, [0.03, 1.666, 0.09]);
  const lidEdge = facial("FACIAL_L_EyelidUpperA1", lid, [0, 0.003, 0.012]);
  const brow = facial("FACIAL_L_ForeheadIn", face, [0.016, 1.688, 0.11]);
  face.updateMatrixWorld(true);
  const driver = MetaHumanFaceDriver.create(face)!;
  assert(driver, "MetaHuman facial joints are detected");
  assert.equal(MetaHumanFaceDriver.create(rpmRoot), null, "RPM rigs keep their blendshapes");
  const at = (o: THREE.Object3D) => { face.updateMatrixWorld(true); return o.getWorldPosition(new THREE.Vector3()); };
  const chin0 = at(chin), lid0 = at(lidEdge), brow0 = at(brow);
  driver.apply({ jawOpen: 1, eyeBlinkLeft: 1, browInnerUp: 1 });
  assert(at(chin).y < chin0.y - 0.01, "jawOpen lowers the chin");
  assert(at(lidEdge).y < lid0.y - 0.004, "blink lowers the upper lid");
  assert(at(brow).y > brow0.y + 0.004, "browInnerUp raises the inner brow");
  assert(!lowerLip.quaternion.equals(new THREE.Quaternion()), "the lower lip follows the jaw");
  driver.apply({});
  assert(at(chin).distanceTo(chin0) < 1e-9 && at(lidEdge).distanceTo(lid0) < 1e-9 && at(brow).distanceTo(brow0) < 1e-9, "Zero weights restore the bind face");
  console.log("MetaHuman checks passed: profile detection, explicit limb chains, same IK as RPM, deterministic seeking, facial joint driver.");
}

// Film mode: subtitle parsing (SRT and WebVTT), placement of one motion per cue, active motion at a time.
{
  const { parseSubtitles, placeMotions, activeAt } = await import("../src/utils/subtitles");
  const srt = "1\r\n00:00:01,000 --> 00:00:03,500\r\n<i>Bonjour</i> à tous.\r\n\r\n2\r\n00:00:04,000 --> 00:00:06,000\r\nLe musée\r\nest fermé.\r\n";
  const vtt = "WEBVTT\n\n00:01.000 --> 00:03.500 align:center\nBonjour à tous.\n\nnote\n00:04.000 --> 00:06.000\nLe musée est fermé.\n";
  for (const raw of [srt, vtt]) {
    const cues = parseSubtitles(raw);
    assert.deepEqual(cues, [
      { start_ms: 1000, end_ms: 3500, text: "Bonjour à tous." },
      { start_ms: 4000, end_ms: 6000, text: "Le musée est fermé." },
    ], "SRT and VTT give the same cues, tags removed, lines joined");
  }
  const cues = parseSubtitles(srt);
  const placed = placeMotions(cues, [4000, 1000]);
  assert.equal(placed[0].start_ms, 1000);
  assert.equal(placed[1].start_ms, 5150, "a motion still playing delays the next one (150 ms rest)");
  assert.equal(placed[1].delay_ms, 1150, "the delay is reported");
  assert.deepEqual(activeAt(placed, [true, true], 500), { index: 0, local_ms: 0 }, "before the first cue: first motion at rest");
  assert.deepEqual(activeAt(placed, [true, true], 4500), { index: 0, local_ms: 3500 });
  assert.deepEqual(activeAt(placed, [true, true], 9000), { index: 1, local_ms: 1000 }, "after a motion: its last pose is held");
  assert.equal(activeAt(placed, [false, false], 2000), null);
  console.log("Film mode checks passed: SRT/VTT parsing, cue placement with reported delay, active motion.");
}

// Engine face channels -> ARKit weights relative to the clip's neutral; grammatical markers kept separate.
{
  const { engineFace } = await import("../src/utils/engineFace");
  const neutral = { brow_left: .43, brow_right: .42, mouth_width: .57, mouth_corner: -.024, eye_open_left: .07, eye_open_right: .07 };
  const rest = engineFace({ mouth_width: .57, mouth_corner: -.024, eye_open_left: .07, eye_open_right: .07, mouth_open: .01 }, neutral);
  assert(Object.values(rest.weights).every((w) => w < 1e-9), "the neutral face gives no expression");
  const round = engineFace({ mouth_width: .45, mouth_open: .1, mouth_corner: -.024 }, neutral);
  assert(round.weights.mouthPucker > .3 && round.weights.mouthFunnel > .3, "narrow, open lips -> rounded");
  const spread = engineFace({ mouth_width: .7, mouth_corner: .05 }, neutral);
  assert(spread.weights.mouthStretchLeft > .3 && spread.weights.mouthSmileLeft > .3, "wide lips, raised corners -> spread, smile");
  const squint = engineFace({ eye_open_left: .03, eye_open_right: .07 }, neutral);
  assert(squint.weights.eyeSquintLeft > .3 && squint.weights.eyeSquintRight < 1e-9, "eyes are measured per side");
  const nm = engineFace({ nm_brow_raise: .65, nm_head_yaw: .9, nm_head_pitch: .1 }, neutral).nonmanual;
  assert(nm.browRaise === .65 && nm.headYaw === .4 && nm.headPitch === .1, "markers pass through, head motion bounded");
  console.log("Engine face checks passed: neutral, rounded/spread lips, per-eye squint, bounded grammatical markers.");
}

// Interpreter performance layer: forearm twist sharing, springs driven by the motion clock (deterministic on seek),
// emotion face below grammar, procedural emotion posture.
{
  const { PerformanceLayer } = await import("../src/utils/performanceLayer");
  const { EmotionStyle, emotionFace } = await import("../src/utils/emotionStyle");
  const { RPM_PROFILE } = await import("../src/utils/rigProfiles");
  const make = (name: string, parent: THREE.Object3D, p: number[]) => { const b = new THREE.Bone(); b.name = name; b.position.fromArray(p); parent.add(b); return b; };
  const root = new THREE.Group();
  const spine = make("Spine", root, [0, 1.1, 0]), spine1 = make("Spine1", spine, [0, 0.1, 0]), spine2 = make("Spine2", spine1, [0, 0.1, 0]);
  const neck = make("Neck", spine2, [0, 0.2, 0]), headB = make("Head", neck, [0, 0.1, 0]);
  const shoulder = make("LeftShoulder", spine2, [0.03, 0.15, 0]);
  const arm = make("LeftArm", shoulder, [0.12, 0, 0]), fore = make("LeftForeArm", arm, [0.28, 0, 0]);
  const hand = make("LeftHand", fore, [0.25, 0, 0]);
  const tw1 = make("lowerarm_twist_01_l", fore, [0.17, 0, 0]), tw2 = make("lowerarm_twist_02_l", fore, [0.08, 0, 0]);
  root.updateMatrixWorld(true);
  const layer = new PerformanceLayer(root, RPM_PROFILE);
  // twist: hand rolled 90 degrees about the forearm axis (+x here) -> twist bones get 60 and 30 degrees
  hand.quaternion.setFromAxisAngle(new THREE.Vector3(1, 0, 0), Math.PI / 2);
  layer.update(undefined, false);
  assert(Math.abs(2 * Math.acos(Math.min(1, tw1.quaternion.w)) - Math.PI / 3) < 1e-3, "twist_01 carries 2/3 of the roll");
  assert(Math.abs(2 * Math.acos(Math.min(1, tw2.quaternion.w)) - Math.PI / 6) < 1e-3, "twist_02 carries 1/3 of the roll");
  // springs: during playback the arm lags a sudden change, then settles; a seek gives the exact target at once
  hand.quaternion.identity();
  arm.quaternion.identity();
  layer.update(0, true);
  const target = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), 0.8);
  arm.quaternion.copy(target);
  layer.update(40, true);
  const lag = arm.quaternion.angleTo(target);
  assert(lag > 0.1, "the spring lags a sudden 0.8 rad change during playback");
  for (let t = 80; t <= 1200; t += 40) { arm.quaternion.copy(target); layer.update(t, true); }
  assert(arm.quaternion.angleTo(target) < 0.01, "and settles on the target");
  arm.quaternion.identity();
  layer.update(5000, true);                                   // seek: no simulation, exact pose
  assert(arm.quaternion.angleTo(new THREE.Quaternion()) < 1e-9, "a seek shows the retargeted pose exactly");
  // emotion face: presets, and brows cut to 30 % under a grammatical marker
  const happy = emotionFace("happy", 1, false);
  assert(happy.mouthSmileLeft > 0.4 && happy.cheekSquintLeft > 0.3, "happy = smile + cheek raise");
  assert(Math.abs(emotionFace("angry", 1, true).browDownLeft - 0.3 * emotionFace("angry", 1, false).browDownLeft) < 1e-9, "grammar wins over emotional brows");
  assert(Object.keys(emotionFace("neutral", 1, false)).length === 0, "neutral adds nothing");
  // procedural posture: sad tilts the head forward (world +x rotation moves the face down/forward)
  const style = new EmotionStyle(root, RPM_PROFILE);
  root.updateMatrixWorld(true);
  const before = headB.getWorldQuaternion(new THREE.Quaternion());
  const rest = [spine, spine1, spine2, neck, headB, shoulder].map((b) => b.quaternion.clone());
  for (let i = 0; i < 60; i++) {
    // the retargeter resets every bone each frame before the posture is added; do the same here
    [spine, spine1, spine2, neck, headB, shoulder].forEach((b, k) => b.quaternion.copy(rest[k]));
    root.updateMatrixWorld(true);
    style.apply(i / 30, "sad", 1, 1 / 30);
  }
  root.updateMatrixWorld(true);
  const tilt = new THREE.Euler().setFromQuaternion(headB.getWorldQuaternion(new THREE.Quaternion()).multiply(before.invert()));
  assert(tilt.x > 0.03, "sad posture lowers the head");
  // ponytail chain: lags a head turn, bounded per link, exact again after a seek
  const tailRoot = new THREE.Group();
  const tHead = make("Head", tailRoot, [0, 1.6, 0]);
  const links = [1, 2, 3, 4].map((i) => i).reduce<THREE.Bone[]>((acc, i) => [...acc, make(`hair_ponytail_0${i}`, acc[acc.length - 1] ?? tHead, [0, -0.025, 0.02])], []);
  tailRoot.updateMatrixWorld(true);
  const tail = new PerformanceLayer(tailRoot, RPM_PROFILE);
  const turn = new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), 0.6);
  const resetTail = () => links.forEach((b) => b.quaternion.identity());
  resetTail(); tail.update(0, true);
  tHead.quaternion.copy(turn);
  resetTail(); tail.update(33, true);
  const swing = links.map((b) => b.quaternion.angleTo(new THREE.Quaternion()));
  assert(swing[0] > 0.05, "the ponytail lags a quick head turn");
  assert(swing.every((a, i) => a <= THREE.MathUtils.degToRad(10 + 3 * i) + 1e-6), "each link stays within its angle limit");
  resetTail(); tail.update(9000, true);
  assert(links.every((b) => b.quaternion.angleTo(new THREE.Quaternion()) < 1e-9), "a seek shows the ponytail at rest under the head");
  console.log("Performance checks passed: forearm twist 2/3-1/3, springs lag/settle/seek-exact, emotion face below grammar, sad posture, ponytail chain.");
}

// Voice emotion -> ARKit blendshapes.
{
  const { emotionToBlendshapes, sampleEmotion, smoothWeights, SIGNING_BROW_CAP, SIGNING_FACE_CAP } =
    await import("../src/utils/emotionFace");
  const base = { angry: 0, disgusted: 0, fearful: 0, happy: 0, neutral: 0, other: 0, sad: 0, surprised: 0, unknown: 0 };
  const neutral = emotionToBlendshapes({ ...base, neutral: 0.95, happy: 0.05 }, "rest");
  assert(Object.values(neutral).every((w) => w === 0), "A near-neutral voice gives a neutral face");
  const happy = emotionToBlendshapes({ ...base, happy: 0.9, neutral: 0.1 }, "rest");
  assert(happy.mouthSmileLeft > 0.5 && happy.cheekSquintRight > 0, "Happy voice smiles at rest");
  for (const emotion of ["happy", "sad", "angry", "surprised", "fearful", "disgusted"]) {
    const scores = { ...base, [emotion]: 1 };
    const rest = emotionToBlendshapes(scores, "rest");
    const signing = emotionToBlendshapes(scores, "signing");
    assert(Object.keys(rest).length > 0, `${emotion} drives the face at rest`);
    for (const [name, w] of Object.entries({ ...rest, ...signing })) assert(w >= 0 && w <= 1, `${name} in [0,1]`);
    for (const [name, w] of Object.entries(signing)) {
      const cap = name.startsWith("brow") ? SIGNING_BROW_CAP : SIGNING_FACE_CAP;
      assert(w <= cap + 1e-9, `${emotion}: ${name} capped while signing`);
    }
    assert.equal(signing.jawOpen ?? 0, 0, "Mouthing jaw stays with captured data while signing");
  }
  assert.equal(emotionToBlendshapes({ ...base, happy: 1 }, "rest", 0).mouthSmileLeft, 0, "Intensity 0 disables");
  const timeline = [
    { start_ms: 0, end_ms: 2000, scores: { ...base, neutral: 1 } },
    { start_ms: 1000, end_ms: 3000, scores: { ...base, happy: 1 } },
  ];
  assert.equal(sampleEmotion(timeline, null, 0)!.neutral, 1);
  const mid = sampleEmotion(timeline, null, 1500)!;
  assert(Math.abs(mid.happy - 0.5) < 1e-9 && Math.abs(mid.neutral - 0.5) < 1e-9, "Interpolates between window centres");
  assert.equal(sampleEmotion(timeline, null, 9000)!.happy, 1);
  assert.equal(sampleEmotion(null, { ...base, sad: 0.7 }, 500)!.sad, 0.7, "Utterance score is the fallback");
  const eased = smoothWeights({}, { mouthSmileLeft: 1 }, 100);
  assert(eased.mouthSmileLeft > 0 && eased.mouthSmileLeft < 1, "Expression eases in");
  assert(smoothWeights({ mouthSmileLeft: 1 }, {}, 10000).mouthSmileLeft === undefined, "Expression fades out fully");
  // Anchoring: a noisy window cannot flip the clip's emotion (CaFE fear clip, windows said happy).
  const { anchorToClip, dominantEmotion } = await import("../src/utils/emotionFace");
  const fearClip = { ...base, fearful: 0.77, happy: 0.22, sad: 0.01 };
  const happyWindow = { ...base, happy: 1 };
  assert.equal(dominantEmotion(anchorToClip(happyWindow, fearClip)), "fearful", "Disagreeing window keeps the clip emotion");
  const lone = anchorToClip({ ...base, sad: 1 }, { ...base, happy: 1 });
  assert(Math.abs(lone.happy - 0.5) < 1e-9 && lone.sad === 0, "Fully disagreeing window halves, never flips");
  const agree = anchorToClip({ ...base, happy: 1 }, { ...base, happy: 1 });
  assert.equal(agree.happy, 1, "Agreeing window keeps full strength");
  const sampled = sampleEmotion([{ start_ms: 0, end_ms: 2000, scores: happyWindow }], fearClip, 1000)!;
  assert.equal(dominantEmotion(sampled), "fearful", "sampleEmotion anchors windows to the clip score");
  console.log("Emotion face checks passed: neutral gating, signing caps, timeline interpolation, easing, clip anchoring.");
}

// Voice pace, tonality and emphasis on signing.
{
  const { scalePath, nodAt, signScale, NOD_MS, EMPHASIS_SCALE } = await import("../src/utils/planStitcher");
  type V = [number, number, number];
  const centre: Record<string, V> = { right_wrist: [0.3, 1.2, 0.2], right_elbow: [0.25, 1.1, 0.05] };
  const pose: Record<string, V> = { right_wrist: [0.4, 1.3, 0.2], right_elbow: [0.25, 1.1, 0.05], right_index_1: [0.42, 1.33, 0.19], hips: [0, 1, 0] };
  const big = scalePath(pose, centre, 1.2);
  assert(Math.abs(big.right_wrist[0] - 0.42) < 1e-9 && Math.abs(big.right_wrist[1] - 1.32) < 1e-9, "Path widens around the sign's centre");
  assert.deepEqual(big.hips, pose.hips, "Body is untouched");
  const handBefore = pose.right_index_1.map((v, i) => v - pose.right_wrist[i]);
  const handAfter = big.right_index_1.map((v, i) => v - big.right_wrist[i]);
  assert(handBefore.every((v, i) => Math.abs(v - handAfter[i]) < 1e-9), "Handshape moves rigidly with the wrist");
  assert.deepEqual(scalePath(centre, centre, 1.5).right_wrist, centre.right_wrist, "A sign's location (its centre) never moves");
  const atChin: Record<string, V> = { head: [0, 1.6, 0.1], right_wrist: [0.02, 1.5, 0.15], right_elbow: [0.2, 1.2, 0.1] };
  const kept = scalePath(atChin, { right_wrist: [0.2, 1.2, 0.2], right_elbow: [0.25, 1.1, 0.05] }, 1.3, { right: 0.12 }).right_wrist;
  assert(kept.every((v, i) => Math.abs(v - atChin.right_wrist[i]) < 1e-9), "A hand touching the face is never moved");
  assert.equal(nodAt(-1, 1), 0); assert.equal(nodAt(NOD_MS + 1, 1), 0); assert.equal(nodAt(100, 0), 0);
  assert(Math.abs(nodAt(NOD_MS / 2, 0.8) - 1) < 1e-9, "Clear stress gets the full nod mid-way");
  assert(Math.abs(nodAt(NOD_MS / 2, 0.2) - 0.7) < 1e-9, "Weaker stress still nods visibly");
  assert(Math.abs(signScale({ emphasis: 1, liveliness: null } as never) - (1 + EMPHASIS_SCALE)) < 1e-9);
  assert(signScale({ emphasis: 0, liveliness: 0 } as never) < 1 && signScale({ emphasis: 0, liveliness: 1 } as never) > 1);

  // Word-timed plan: the sign starts at its word, holds at its end, nods at onset.
  const wristClip: MotionFile = {
    rig_version: "signaculture-v1", fps: 25, duration_ms: 400, joints: ["right_wrist"], synthetic: true,
    frames: [0, 100, 200, 300, 400].map((t) => ({ time_ms: t, joints: { right_wrist: [0.3 + t / 2000, 1.2, 0.2] as V } })),
  };
  const wordPlan = {
    plan_id: "p", source_text: "Bonjour", target_language: "LSF", unknowns: [], linguistic_review: "pending",
    animation_review: "pending", rights_status: "pending", render_allowed_for_publication: false,
    transition_ms: 0, timing_mode: "audio_words", duration_ms: 2000,
    segments: [{ gloss_id: "g", motion_id: "m", start_ms: 600, duration_ms: 800, hold_ms: 160, emphasis: 1, liveliness: 0.5 }],
  } as unknown as AnimationPlan;
  const timed = stitchAnimationPlan(wordPlan, new Map([["m", wristClip]]));
  const at = (t: number) => timed.frames.find((f) => f.time_ms === t)!;
  assert.equal(timed.duration_ms, 2000);
  assert.deepEqual(at(0).joints.right_wrist, at(560).joints.right_wrist, "Rest before the word starts");
  assert((at(760).nod ?? 0) > 0.9 && !at(1200).nod, "Nod at the onset of the stressed sign only");
  const mid = at(1000).joints.right_wrist, rest = stitchAnimationPlan({ ...wordPlan, segments: [{ ...wordPlan.segments[0], emphasis: 0, liveliness: null }] } as AnimationPlan, new Map([["m", wristClip]])).frames.find((f) => f.time_ms === 1000)!.joints.right_wrist;
  assert(Math.abs(mid[0] - 0.4) > Math.abs(rest[0] - 0.4) * 1.25, "Mid-sign path is clearly enlarged around the sign's centre");
  const holdOnly = { ...wordPlan, segments: [{ ...wordPlan.segments[0], emphasis: 0, liveliness: null }] } as AnimationPlan;
  const held = stitchAnimationPlan(holdOnly, new Map([["m", wristClip]]));
  const hAt = (t: number) => held.frames.find((f) => f.time_ms === t)!.joints.right_wrist;
  assert.deepEqual(hAt(1280), hAt(1400), "Source time is frozen during the end hold");
  assert(hAt(1000)[0] < hAt(1200)[0], "The sign itself still plays before the hold");
  // Retimed playback keeps hand-tracking gaps: a hand missing in either neighbouring frame is
  // flagged missing and its fingers are not blended in from the other frame.
  const { resampleFrame } = await import("../src/utils/planStitcher");
  const fingers = (x: number) => Object.fromEntries(["thumb", "index", "middle", "ring", "pinky"].flatMap((f) => [1, 2, 3, 4].map((i) => [`left_${f}_${i}`, [x, 1.2, 0.1] as V])));
  const gappy: MotionFile = { rig_version: "signaculture-v1", fps: 25, duration_ms: 80, joints: ["left_wrist"], synthetic: true, frames: [
    { time_ms: 0, joints: { left_wrist: [0, 1.2, 0.1], ...fingers(0) }, hand_tracking: { left: { observed: true, source: "holistic" } } },
    { time_ms: 40, joints: { left_wrist: [0.1, 1.2, 0.1] }, hand_tracking: { left: { observed: false, source: "missing" } } },
    { time_ms: 80, joints: { left_wrist: [0.2, 1.2, 0.1], ...fingers(0.2) }, hand_tracking: { left: { observed: true, source: "holistic" } } }] };
  const gapMid = resampleFrame(gappy, 20);
  assert.equal(gapMid.hand_tracking?.left?.observed, false, "A gap in either neighbour flags the hand missing");
  assert(!Object.keys(gapMid.joints).some((k) => k.startsWith("left_index")), "No one-sided fingers detached from the wrist");
  const solid = { ...gappy, frames: [gappy.frames[0], { ...gappy.frames[2], time_ms: 40 }] };
  assert.equal(resampleFrame(solid, 20).hand_tracking?.left?.observed, true, "A hand tracked on both sides stays tracked");
  console.log("Prosody checks passed: path scaling keeps location and handshape, nod envelope, word-timed holds, hand-gap resampling.");
}

// Live mode: phrases cut at pauses, long speech cut, WAV for the transcription endpoint.
{
  const { PhraseDetector, encodeWav } = await import("../src/utils/phraseDetector");
  const rate = 48000;
  const tone = (s: number, amp: number) => Float32Array.from({ length: Math.round(s * rate) }, (_, i) => amp * Math.sin((2 * Math.PI * 220 * i) / rate));
  const noise = (s: number) => Float32Array.from({ length: Math.round(s * rate) }, () => (Math.random() - 0.5) * 0.004);
  const phrases: number[] = [];
  const vad = new PhraseDetector({ sampleRate: rate }, (p) => phrases.push(p.length / rate));
  for (const chunk of [noise(1), tone(1.2, 0.2), noise(1), tone(0.5, 0.2), noise(1), tone(0.1, 0.2), noise(1)]) {
    for (let i = 0; i < chunk.length; i += 128) vad.push(chunk.subarray(i, i + 128));   // AudioWorklet-sized blocks
  }
  assert.equal(phrases.length, 2, "two phrases, the 0.1 s click is ignored");
  assert(phrases[0] > 1.2 && phrases[0] < 1.9, `first phrase keeps its pre-roll and a short tail (${phrases[0].toFixed(2)} s)`);
  const long: number[] = [];
  const vad2 = new PhraseDetector({ sampleRate: rate }, (p) => long.push(p.length / rate));
  for (const chunk of [noise(1), tone(10, 0.2), noise(1)]) for (let i = 0; i < chunk.length; i += 128) vad2.push(chunk.subarray(i, i + 128));
  assert(long.length === 2 && long[0] <= 8.05, "speech longer than 8 s is cut so the avatar never waits");
  const wav = new DataView(encodeWav(tone(1, 0.5), rate));
  assert.equal(String.fromCharCode(wav.getUint8(0), wav.getUint8(1), wav.getUint8(2), wav.getUint8(3)), "RIFF");
  assert.equal(wav.getUint32(24, true), 16000, "16 kHz");
  assert.equal(wav.getUint32(40, true), 32000, "1 s of 16-bit mono at 16 kHz");
  console.log("Live mode checks passed: pause-cut phrases with pre-roll, clicks ignored, 8 s cut, 16 kHz WAV.");
}
