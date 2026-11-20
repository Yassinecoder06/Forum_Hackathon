import * as THREE from "three";
import { findRigBone, type RigProfile } from "./rigProfiles";

/**
 * Emotion as STYLE of the interpreter (presentation layer, below grammar):
 * - face: FACS-based ARKit expression presets per emotion (Ekman action-unit combinations), additive;
 *   when a grammatical marker (question / negation) is active, the emotional brows are cut to 30 % so the
 *   grammatical face always wins;
 * - body: an emotion "blendspace" of postures applied to torso, neck, head and clavicles only (arms are masked:
 *   they sign). Postures come from Mixamo idle clips when present (.runtime/emotion-idles/<label>.fbx, served by
 *   the API), otherwise from procedural postures. Weights: the emotion's confidence, neutral for the rest.
 * Heuristic presentation values, not a model of deaf signers' emotional signing.
 */

export const EMOTIONS = ["happy", "angry", "sad", "surprised", "fearful", "disgusted"] as const;
export type Emotion = (typeof EMOTIONS)[number] | "neutral";

const both = (name: string, v: number) => ({ [`${name}Left`]: v, [`${name}Right`]: v });
export const FACE_PRESETS: Record<Exclude<Emotion, "neutral">, Record<string, number>> = {
  happy: { ...both("mouthSmile", 0.45), ...both("cheekSquint", 0.35), ...both("eyeSquint", 0.12) },
  sad: { browInnerUp: 0.45, ...both("mouthFrown", 0.35), ...both("eyeSquint", 0.05) },
  angry: { ...both("browDown", 0.55), ...both("noseSneer", 0.2), ...both("mouthPress", 0.3), ...both("eyeSquint", 0.25) },
  fearful: { browInnerUp: 0.5, ...both("browOuterUp", 0.3), ...both("eyeWide", 0.45), ...both("mouthStretch", 0.25) },
  surprised: { browInnerUp: 0.45, ...both("browOuterUp", 0.45), ...both("eyeWide", 0.45), jawOpen: 0.18 },
  disgusted: { ...both("noseSneer", 0.45), ...both("mouthUpperUp", 0.3), ...both("browDown", 0.2) },
};
const BROWS = /^brow/;

/** Face weights for an emotion at a given intensity (0..1); emotional brows cut when grammar is active. */
export function emotionFace(label: string | undefined, intensity: number, grammarActive: boolean) {
  const preset = label && label in FACE_PRESETS ? FACE_PRESETS[label as keyof typeof FACE_PRESETS] : null;
  const out: Record<string, number> = {};
  if (!preset || intensity <= 0) return out;
  for (const [k, v] of Object.entries(preset)) out[k] = v * intensity * (grammarActive && BROWS.test(k) ? 0.3 : 1);
  return out;
}

/** Procedural postures: world-space rotation vectors (x = pitch forward, y = yaw, z = roll) per canonical bone,
 * and a shoulder lift (radians, + = up). */
type Posture = { bones: Partial<Record<string, [number, number, number]>>; shoulderLift: number };
export const POSTURES: Record<Exclude<Emotion, "neutral">, Posture> = {
  sad: { bones: { Spine2: [0.07, 0, 0], Neck: [0.05, 0, 0], Head: [0.08, 0, 0] }, shoulderLift: -0.06 },
  happy: { bones: { Spine2: [-0.03, 0, 0], Head: [-0.04, 0, 0.02] }, shoulderLift: 0.02 },
  angry: { bones: { Spine2: [0.04, 0, 0], Head: [0.03, 0, 0] }, shoulderLift: 0.04 },
  fearful: { bones: { Spine2: [-0.03, 0, 0], Head: [0.02, 0, 0] }, shoulderLift: 0.08 },
  surprised: { bones: { Spine2: [-0.03, 0, 0], Head: [-0.06, 0, 0] }, shoulderLift: 0.03 },
  disgusted: { bones: { Spine2: [-0.02, 0, 0], Head: [0, 0.05, -0.02] }, shoulderLift: 0 },
};

const MIXAMO: Record<string, string> = {
  Hips: "Hips", Spine: "Spine", Spine1: "Spine1", Spine2: "Spine2", Neck: "Neck", Head: "Head",
  LeftShoulder: "LeftShoulder", RightShoulder: "RightShoulder",
};
const MASKED_WEIGHT: Record<string, number> = { Hips: 0.4, Spine: 0.8, Spine1: 0.8, Spine2: 0.8, Neck: 0.7, Head: 0.6, LeftShoulder: 0.6, RightShoulder: 0.6 };

/** World-space rotation of a bone in a Mixamo clip at time t, relative to the clip skeleton's bind pose. */
type IdleClip = { duration: number; sample: (t: number) => Map<string, THREE.Quaternion> };

function idleFromFbx(group: THREE.Object3D, clip: THREE.AnimationClip): IdleClip {
  const bones = new Map<string, THREE.Object3D>();
  group.traverse((o) => {
    const name = o.name.replace(/^mixamorig[:_]?/i, "");
    if (name in MIXAMO) bones.set(name, o);
  });
  group.updateMatrixWorld(true);
  const bind = new Map([...bones].map(([k, b]) => [k, b.getWorldQuaternion(new THREE.Quaternion())]));
  const mixer = new THREE.AnimationMixer(group);
  mixer.clipAction(clip).play();
  return {
    duration: clip.duration,
    sample(t: number) {
      mixer.setTime(t % Math.max(clip.duration, 1e-3));
      group.updateMatrixWorld(true);
      const out = new Map<string, THREE.Quaternion>();
      for (const [k, b] of bones) out.set(k, b.getWorldQuaternion(new THREE.Quaternion()).multiply(bind.get(k)!.clone().invert()));
      return out;
    },
  };
}

const logQ = (q: THREE.Quaternion, out = new THREE.Vector3()) => {
  const w = THREE.MathUtils.clamp(q.w < 0 ? -q.w : q.w, -1, 1), s = Math.sqrt(1 - w * w), sign = q.w < 0 ? -1 : 1;
  return s < 1e-8 ? out.set(0, 0, 0) : out.set(sign * q.x, sign * q.y, sign * q.z).multiplyScalar(2 * Math.acos(w) / s);
};
const expV = (v: THREE.Vector3) => {
  const a = v.length();
  return a < 1e-9 ? new THREE.Quaternion() : new THREE.Quaternion().setFromAxisAngle(v.clone().divideScalar(a), a);
};

export class EmotionStyle {
  private bones = new Map<string, THREE.Object3D>();
  private idles = new Map<string, IdleClip>();
  private neutralMean?: Map<string, THREE.Quaternion>;
  /** Smoothed emotion weights (eased over ~0.6 s so a new emotion does not pop). */
  private weights = new Map<string, number>();

  constructor(root: THREE.Object3D, profile: RigProfile) {
    for (const name of Object.keys(MIXAMO)) {
      const bone = findRigBone(root, profile, name);
      if (bone) this.bones.set(name, bone);
    }
  }

  get loadedIdles() {
    return [...this.idles.keys()];
  }

  /** Load the Mixamo idles listed by the API (missing ones keep the procedural posture). */
  async loadIdles(list: { label: string; url: string }[]) {
    const { FBXLoader } = await import("three/examples/jsm/loaders/FBXLoader.js");
    const loader = new FBXLoader();
    for (const { label, url } of list) {
      try {
        const group = await loader.loadAsync(url);
        if (group.animations[0]) this.idles.set(label, idleFromFbx(group, group.animations[0]));
      } catch {
        /* keep the procedural posture for this emotion */
      }
    }
    const neutral = this.idles.get("neutral");
    if (neutral) {
      // reference = mean pose of the neutral idle, so emotional clips are applied as offsets from it
      const sums = new Map<string, THREE.Vector3>();
      const n = 24;
      for (let i = 0; i < n; i++)
        for (const [k, q] of neutral.sample((i / n) * neutral.duration)) sums.set(k, (sums.get(k) ?? new THREE.Vector3()).add(logQ(q)));
      this.neutralMean = new Map([...sums].map(([k, v]) => [k, expV(v.divideScalar(n))]));
    }
  }

  /** Apply the posture blendspace after retargeting. label/confidence: the motion's voice emotion. */
  apply(time: number, label: string | undefined, confidence: number, dt: number) {
    const target = new Map<string, number>();
    if (label && label !== "neutral" && label in POSTURES) target.set(label, THREE.MathUtils.clamp(confidence, 0, 1));
    for (const e of EMOTIONS) {
      const current = this.weights.get(e) ?? 0, goal = target.get(e) ?? 0;
      this.weights.set(e, current + (goal - current) * Math.min(1, dt * 1.7));
    }
    const neutralIdle = this.idles.get("neutral");
    const neutralPose = neutralIdle?.sample(time);
    for (const [name, bone] of this.bones) {
      const rv = new THREE.Vector3();
      // base life from the neutral idle (weight shift, breathing posture) relative to its own mean
      if (neutralPose && this.neutralMean?.has(name)) rv.add(logQ(neutralPose.get(name)!.clone().multiply(this.neutralMean.get(name)!.clone().invert())));
      for (const e of EMOTIONS) {
        const w = this.weights.get(e) ?? 0;
        if (w < 1e-3) continue;
        const idle = this.idles.get(e);
        if (idle && this.neutralMean?.has(name)) {
          const q = idle.sample(time).get(name);
          if (q) rv.addScaledVector(logQ(q.clone().multiply(this.neutralMean.get(name)!.clone().invert())), w);
        } else {
          const p = POSTURES[e];
          const v = p.bones[name];
          if (v) rv.add(new THREE.Vector3(...v).multiplyScalar(w));
          if (name === "LeftShoulder") rv.z += p.shoulderLift * w;
          if (name === "RightShoulder") rv.z -= p.shoulderLift * w;
        }
      }
      rv.multiplyScalar(MASKED_WEIGHT[name] ?? 0.6);
      if (rv.lengthSq() < 1e-10) continue;
      rv.clampLength(0, 0.25);
      const world = bone.getWorldQuaternion(new THREE.Quaternion());
      const parent = bone.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion();
      bone.quaternion.copy(parent.invert().multiply(expV(rv).multiply(world)));
      bone.updateMatrixWorld(true);
    }
  }
}
