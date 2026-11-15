import assert from "node:assert/strict";
import * as THREE from "three";
import { stitchAnimationPlan } from "../src/utils/planStitcher";
import { HumanoidRetargeter } from "../src/utils/retargetHumanoid";
import { browWeights, faceCalibration, sampleFace, stabilizeRotations } from "../src/utils/avatarPlayback";
import { handContact, segmentDistance } from "../src/utils/handContact";
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
const touching = handContact([leftCapsule], [{ ...rightCapsule, sourceA: point(-.09), sourceB: point(-.09, 1) }]);
assert(touching.contact && touching.correction.length() === 0, "Preserve source-supported contact");
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
