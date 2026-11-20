import * as THREE from "three";
import type { MotionFile, MotionFrame } from "../types/contract";

export const HAND_GAP_MS = 180;
export type HandState = "observed" | "interpolated" | "held" | "untracked";
export type RotationFrame = { time: number; rotations: THREE.Quaternion[]; valid: boolean };

/** Native capture files remain immutable. These are derived, timestamp-based tracks. */
export function stabilizeRotations(
  frames: RotationFrame[],
  rest: THREE.Quaternion[],
  /** Per-frame neutral pose. Invalid frames fade to this instead of the authored bind pose. */
  neutral?: (THREE.Quaternion[] | undefined)[],
) {
  const observed = frames.map((f) => ({ ...f, rotations: f.rotations.map((q) => q.clone()) }));
  let rejected = 0;
  // Remove only isolated reversals with agreeing neighbours, never sustained gestures.
  for (let i = 1; i < observed.length - 1; i++) {
    const a = frames[i - 1], b = frames[i], c = frames[i + 1];
    if (!a.valid || !b.valid || !c.valid || c.time - a.time > HAND_GAP_MS) continue;
    b.rotations.forEach((q, j) => {
      if (a.rotations[j].angleTo(c.rotations[j]) < .45 &&
          q.angleTo(a.rotations[j]) > 1.2 && q.angleTo(c.rotations[j]) > 1.2) {
        observed[i].rotations[j].slerpQuaternions(a.rotations[j], c.rotations[j], (b.time - a.time) / (c.time - a.time));
        rejected++;
      }
    });
  }
  const states: HandState[] = [];
  let previous = -1;
  const nextValid = frames.map(() => -1);
  let next = -1;
  for (let i = frames.length - 1; i >= 0; i--) {
    nextValid[i] = next;
    if (frames[i].valid) next = i;
  }
  const result = observed.map((f, i) => {
    const rotations = f.rotations.map((q) => q.clone());
    if (f.valid) {
      states.push("observed");
      previous = i;
      const a = observed[i - 1], c = observed[i + 1];
      if (a?.valid && c?.valid && c.time - a.time <= HAND_GAP_MS) {
        // Low-pass small noise; retain large coherent movements. Symmetric and seek-safe.
        rotations.forEach((q, j) => {
          const midpoint = a.rotations[j].clone().slerp(c.rotations[j], (f.time - a.time) / (c.time - a.time));
          const weight = .3 * Math.max(0, 1 - a.rotations[j].angleTo(c.rotations[j]) / .7);
          q.slerp(midpoint, weight);
        });
      }
    } else {
      const a = observed[previous], b = observed[nextValid[i]];
      if (a && b && b.time - a.time <= HAND_GAP_MS) {
        rotations.forEach((q, j) => q.slerpQuaternions(a.rotations[j], b.rotations[j], (f.time - a.time) / (b.time - a.time)));
        states.push("interpolated");
      } else if (a) {
        // Hold only a past observation, then fade to a per-frame neutral pose.
        // Never borrow a future hand pose: that shows a sign before it happens.
        const age = f.time - a.time;
        const window = b ? Math.min(320, (b.time - a.time) / 2) : 320;
        const fade = THREE.MathUtils.smoothstep(age, Math.min(80, window / 4), window);
        rotations.forEach((q, j) => {
          const target = neutral?.[i]?.[j] ?? rest[j];
          q.copy(a.rotations[j]).slerp(target, fade);
        });
        states.push(fade < 1 ? "held" : "untracked");
      } else {
        // No past observation (start of clip or first appearance later):
        // use the neutral hand rather than previewing the next observed pose.
        rotations.forEach((q, j) => q.copy(neutral?.[i]?.[j] ?? rest[j]));
        states.push("untracked");
      }
    }
    return { time: f.time, rotations, valid: f.valid };
  });
  return { frames: result, states, rejected };
}

export function frameInterval(frames: { time_ms: number }[], time: number) {
  let lo = 0, hi = frames.length - 1;
  while (lo < hi) {
    const mid = Math.ceil((lo + hi) / 2);
    if (frames[mid].time_ms <= time) lo = mid; else hi = mid - 1;
  }
  const next = Math.min(lo + 1, frames.length - 1);
  const alpha = THREE.MathUtils.clamp((time - frames[lo].time_ms) / Math.max(1, frames[next].time_ms - frames[lo].time_ms), 0, 1);
  return { lo, next, alpha };
}

export function sampleFace(motion: MotionFile | null, time: number) {
  if (!motion?.frames.length) return null;
  const { lo, next, alpha } = frameInterval(motion.frames, time);
  const a = motion.frames[lo].face, b = motion.frames[next].face;
  if (!a || !b) {
    const available = a || b;
    if (!available) return null;
    const age = a ? time - motion.frames[lo].time_ms : motion.frames[next].time_ms - time;
    const weight = 1 - THREE.MathUtils.smoothstep(Math.max(0, age), 0, 120);
    if (weight <= 0) return null;
    const reference = faceCalibration(motion);
    const face: Record<string, number> = {};
    for (const [key, value] of Object.entries(available)) {
      if (!Number.isFinite(value)) continue;
      if (key.endsWith("_neutral") || key === "brow_reference_blend") face[key] = value;
      else {
        const neutral = key === "brow_left" || key === "brow_right" ? available[`${key}_neutral`] ?? reference?.[key] ?? .38 : 0;
        face[key] = neutral + (value - neutral) * weight;
      }
    }
    return face;
  }
  const face: Record<string, number> = {};
  for (const key of Object.keys(a))
    if (Number.isFinite(a[key]) && Number.isFinite(b[key])) face[key] = a[key] + (b[key] - a[key]) * alpha;
  return face;
}

/** Conservative fallback in the existing capture ratio's units, NOT per-clip centering.
 * Explicit neutral references override it without erasing sustained brow expressions. */
export function browWeights(ratio: number | undefined, neutral?: number, limit = Number.isFinite(neutral) ? .65 : .18) {
  if (ratio === undefined || !Number.isFinite(ratio)) return { up: 0, down: 0 };
  if (neutral === undefined || !Number.isFinite(neutral)) neutral = .38;
  const delta = ratio - neutral;
  return {
    up: THREE.MathUtils.clamp((delta - .01) * 2, 0, limit),
    down: THREE.MathUtils.clamp((-delta - .01) * 2, 0, limit),
  };
}

/** These six source recordings' first frames were visually inspected at rest.
 * This is an engineering reference estimate, NOT linguistic neutral approval.
 * Never infer a neutral expression from a whole utterance's median. */
export function faceCalibration(motion: MotionFile) {
  if (motion.face_calibration && Object.values(motion.face_calibration).every(Number.isFinite)) return motion.face_calibration;
  const url = motion.attribution?.source_url || "";
  const reference = url.match(/^https:\/\/www\.sign-lang\.uni-hamburg\.de\/dicta-sign\/portal\/concepts\/cs\/cs_(380|490|905|535|926|988)\.html$/);
  const first = motion.frames[0];
  if (!reference || !first || first.time_ms !== 0 || !first.face) return undefined;
  const resting = ["left", "right"].every((s) => first.joints[`${s}_wrist`] && first.joints[`${s}_shoulder`] && first.joints[`${s}_wrist`][1] < first.joints[`${s}_shoulder`][1] - .2);
  if (!resting || !Number.isFinite(first.face.brow_left) || !Number.isFinite(first.face.brow_right)) return undefined;
  return { brow_left: first.face.brow_left, brow_right: first.face.brow_right };
}

export function hasHand(frame: MotionFrame, side: string) {
  return frame.playback_quality?.[`${side}_hand_rejected`] !== true && frame.hand_tracking?.[side]?.observed !== false && ["index", "middle", "ring", "pinky", "thumb"].every((finger) =>
    [1, 2, 3, 4].every((i) => frame.joints[`${side}_${finger}_${i}`]?.every(Number.isFinite)));
}
