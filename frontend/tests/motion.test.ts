import assert from "node:assert/strict";
import * as THREE from "three";
import { stitchAnimationPlan } from "../src/utils/planStitcher";
import { HumanoidRetargeter } from "../src/utils/retargetHumanoid";
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
