import * as THREE from "three";
import type { InterpolatedJoints } from "./motionInterpolator";

/**
 * Default idle animation and rotation-space blending with signed motion.
 *
 * The idle is a real AnimationClip of bone rotations. If the avatar file ships a
 * clip whose name contains "idle", that clip is used. Otherwise one is built at
 * load: neutral-stance joint positions with slow breathing, weight shift and arm
 * drift are converted into rotations by the same retargeter that poses signs,
 * plus hip/spine/neck rotations the retargeter does not drive. All oscillations
 * divide the loop period, so the clip loops seamlessly.
 *
 * During playback, the sign's rotations are slerped over the idle per bone. The
 * blend weight eases in at the start of a phrase and out at its end, so signing
 * rises out of and settles back into the idle. Bones the retargeter does not
 * drive (hips, spine, legs) keep the idle throughout, so breathing continues.
 * This is a display layer: it never changes captured motion data.
 */
export const IDLE_PERIOD = 9;              // seconds
const SAMPLE_STEP = 0.1;                   // seconds between generated keyframes
export const BLEND_IN_MS = 400;
export const BLEND_OUT_MS = 500;

type Retargeter = { apply(joints: InterpolatedJoints): void };

const wave = (t: number, cycles: number, phase = 0) => Math.sin((t / IDLE_PERIOD) * cycles * Math.PI * 2 + phase);

/** Neutral stance with slow, looping drift (positions in the motion's metric space). */
export function idleJoints(base: InterpolatedJoints, t: number): InterpolatedJoints {
  const joints: InterpolatedJoints = { ...base };
  const shift = 0.008 * wave(t, 1);                   // weight shift side to side
  const breathe = 0.003 * wave(t, 2);                 // shoulders rise with each breath
  for (const side of ["left", "right"] as const) {
    const sign = side === "left" ? 1 : -1;
    joints[`${side}_shoulder`] = [sign * 0.18 + shift, 1.48 + breathe, 0];
    joints[`${side}_elbow`] = [sign * 0.25 + shift * 1.2, 1.2 + breathe * 0.5, 0.04 + 0.004 * wave(t, 1, 2)];
    // Hands resting loosely together; they drift a little as one unit.
    joints[`${side}_wrist`] = [sign * 0.17 + shift * 1.3 + 0.005 * wave(t, 1, 1),
      1.02 + 0.004 * wave(t, 3, sign), 0.1 + 0.005 * wave(t, 1, 2)];
  }
  return joints;
}

/** Rotations of bones the retargeter does not drive: hips, spine, neck and head. */
function addTorso(root: THREE.Object3D, t: number) {
  const turn = (name: string, x: number, y: number, z: number) => {
    const bone = root.getObjectByName(name);
    if (bone) bone.quaternion.multiply(new THREE.Quaternion().setFromEuler(new THREE.Euler(x, y, z)));
  };
  turn("Hips", 0, 0.008 * wave(t, 1, 1), 0.012 * wave(t, 1));
  turn("Spine", 0, 0, -0.006 * wave(t, 1));          // counter-roll keeps the shoulders level
  turn("Spine1", -0.006 * wave(t, 2), 0, -0.004 * wave(t, 1));
  turn("Spine2", -0.01 * wave(t, 2, -0.5), 0, 0);
  const pitch = 0.012 * wave(t, 1, 0.7) + 0.006 * wave(t, 3);
  const yaw = 0.02 * wave(t, 1, 2) + 0.008 * wave(t, 2, 1.3);
  const roll = 0.01 * wave(t, 1, 2.1);
  turn("Neck", pitch * 0.4, yaw * 0.4, roll * 0.4);
  turn("Head", pitch * 0.6, yaw * 0.6, roll * 0.6);
}

/** Snapshot of every bone's local rotation (call while the model is at its bind pose). */
export function bindPose(root: THREE.Object3D) {
  const pose = new Map<string, THREE.Quaternion>();
  root.traverse((o) => { if (o instanceof THREE.Bone) pose.set(o.name, o.quaternion.clone()); });
  return pose;
}

/** Bones the retargeter actually rotates when posing the neutral stance. */
export function drivenBones(root: THREE.Object3D, retarget: Retargeter, base: InterpolatedJoints, bind: Map<string, THREE.Quaternion>) {
  retarget.apply(idleJoints(base, 0));
  const driven = new Set<string>();
  root.traverse((o) => {
    if (o instanceof THREE.Bone && bind.has(o.name) && o.quaternion.angleTo(bind.get(o.name)!) > 1e-4) driven.add(o.name);
  });
  return driven;
}

/** Generate the default idle clip by converting idle poses into bone rotations. */
export function buildIdleClip(root: THREE.Object3D, retarget: Retargeter, base: InterpolatedJoints, bind: Map<string, THREE.Quaternion>) {
  const bones: THREE.Bone[] = [];
  root.traverse((o) => { if (o instanceof THREE.Bone) bones.push(o); });
  const steps = Math.round(IDLE_PERIOD / SAMPLE_STEP);
  const times = Array.from({ length: steps + 1 }, (_, i) => i * SAMPLE_STEP);
  const values = bones.map(() => [] as number[]);
  for (const t of times) {
    retarget.apply(idleJoints(base, t % IDLE_PERIOD));   // resets every bone, then poses arms and hands
    addTorso(root, t % IDLE_PERIOD);
    bones.forEach((bone, i) => values[i].push(...bone.quaternion.toArray()));
  }
  // Keep a track for every bone that ever leaves its bind rotation: moving bones and
  // bones held away from bind (arms, fingers), so the clip alone defines the idle pose.
  const tracks = bones.flatMap((bone, i) => {
    const v = values[i], rest = bind.get(bone.name);
    if (!rest) return [];
    const sample = new THREE.Quaternion();
    for (let k = 0; k < v.length; k += 4) {
      if (sample.fromArray(v, k).angleTo(rest) > 1e-4) return [new THREE.QuaternionKeyframeTrack(`${bone.name}.quaternion`, times, v)];
    }
    return [];
  });
  for (const bone of bones) bone.quaternion.copy(bind.get(bone.name) ?? bone.quaternion);
  return new THREE.AnimationClip("Idle (generated)", IDLE_PERIOD, tracks);
}

const smoother = (x: number) => {
  const t = THREE.MathUtils.clamp(x, 0, 1);
  return t * t * t * (t * (t * 6 - 15) + 10);
};

/** Signing weight for a phrase: 0 = idle, 1 = signed rotations. */
export function signingWeight(timeMs: number, durationMs: number) {
  return smoother(timeMs / BLEND_IN_MS) * smoother((durationMs - timeMs) / BLEND_OUT_MS);
}

/** Plays the idle clip and blends signed rotations over it, bone by bone. */
export class IdleBlender {
  private mixer: THREE.AnimationMixer;
  private bones: THREE.Bone[] = [];
  private idle: THREE.Quaternion[] = [];
  readonly clip: THREE.AnimationClip;

  constructor(private root: THREE.Object3D, clip: THREE.AnimationClip, private driven: Set<string>) {
    this.clip = clip;
    this.mixer = new THREE.AnimationMixer(root);
    this.mixer.clipAction(clip).play();
    const animated = new Set(clip.tracks.map((t) => t.name.replace(/\.quaternion$/, "")));
    root.traverse((o) => { if (o instanceof THREE.Bone && animated.has(o.name)) this.bones.push(o); });
    this.idle = this.bones.map(() => new THREE.Quaternion());
  }

  /** Pose the idle at `seconds` and remember it. */
  sampleIdle(seconds: number) {
    this.mixer.setTime(seconds % this.clip.duration);
    this.bones.forEach((b, i) => this.idle[i].copy(b.quaternion));
    this.root.updateMatrixWorld(true);
  }

  /** After the retargeter wrote signed rotations: blend driven bones, restore the idle elsewhere. */
  blendSigned(weight: number) {
    this.bones.forEach((b, i) => {
      if (this.driven.has(b.name)) b.quaternion.slerpQuaternions(this.idle[i], b.quaternion.clone(), weight);
      else b.quaternion.copy(this.idle[i]);
    });
    this.root.updateMatrixWorld(true);
  }

  dispose() {
    this.mixer.stopAllAction();
    this.mixer.uncacheRoot(this.root);
  }
}
