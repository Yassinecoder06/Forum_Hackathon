import * as THREE from "three";
import { roleShiftAt, roleShiftSteps, type RoleShiftPose, type RoleStep } from "./roleShift";

/**
 * AR companion helpers (pure, unit-tested): camera frame size for the brain, scene-change measure, gesture maths,
 * facing the person, and the "look at it" overlay (head + eyes turn toward the side of the picture where the thing
 * the brain talks about is: the minimal start of pointing, reusing the role-shift overlay of the film mode).
 */

/** The brain sees a JPEG whose longest side is at most `max` px (about 512: enough for objects, small to send). */
export function fitSize(width: number, height: number, max = 512) {
  if (!(width > 0 && height > 0)) return { width: 0, height: 0 };
  const k = Math.min(1, max / Math.max(width, height));
  return { width: Math.max(1, Math.round(width * k)), height: Math.max(1, Math.round(height * k)) };
}

/** RGBA pixels -> luminance (0-255), for the tiny scene thumbnails. */
export function toGray(rgba: Uint8Array | Uint8ClampedArray) {
  const out = new Uint8Array(rgba.length >> 2);
  for (let i = 0, j = 0; j < out.length; i += 4, j++) out[j] = (rgba[i] * 77 + rgba[i + 1] * 150 + rgba[i + 2] * 29) >> 8;
  return out;
}

/** Mean absolute difference of two thumbnails of the same size, 0 (same) .. 1 (opposite); 1 when sizes differ. */
export function sceneDifference(a: Uint8Array | null, b: Uint8Array | null) {
  if (!a || !b || a.length !== b.length || !a.length) return 1;
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += Math.abs(a[i] - b[i]);
  return sum / (a.length * 255);
}

/** Thumbnail size and change threshold of the scene watcher (one thumbnail every SCENE_CHECK_MS). */
export const SCENE_THUMB = { width: 16, height: 12 };
export const SCENE_CHANGE = 0.16;
export const SCENE_CHECK_MS = 1500;
/** Never comment on the scene more often than this. */
export const SCENE_MIN_GAP_MS = 20000;

/** WebGL readPixels rows are bottom-up: flip them in place. */
export function flipRows(pixels: Uint8Array, width: number, height: number) {
  const row = width * 4, tmp = new Uint8Array(row);
  for (let y = 0; y < height >> 1; y++) {
    const a = y * row, b = (height - 1 - y) * row;
    tmp.set(pixels.subarray(a, a + row));
    pixels.copyWithin(a, b, b + row);
    pixels.set(tmp, b);
  }
  return pixels;
}

export type Touch2 = { x0: number; y0: number; x1: number; y1: number };
export type Placement = { scale: number; yaw: number };
export const SCALE_RANGE = [0.15, 2.5] as const;

/** Two-finger gesture: pinch scales, twist turns, relative to the placement when both fingers came down. */
export function pinchPlacement(start: Touch2, now: Touch2, base: Placement): Placement {
  const dist = (t: Touch2) => Math.hypot(t.x1 - t.x0, t.y1 - t.y0);
  const angle = (t: Touch2) => Math.atan2(t.y1 - t.y0, t.x1 - t.x0);
  const ratio = dist(start) > 1 ? dist(now) / dist(start) : 1;
  let turn = angle(now) - angle(start);
  turn = Math.atan2(Math.sin(turn), Math.cos(turn));
  return { scale: THREE.MathUtils.clamp(base.scale * ratio, SCALE_RANGE[0], SCALE_RANGE[1]), yaw: base.yaw - turn };
}

/** Yaw (about +Y) that turns a model at `from` (whose front is +Z) to face the point `to`. */
export function facingYaw(from: { x: number; z: number }, to: { x: number; z: number }) {
  return Math.atan2(to.x - from.x, to.z - from.z);
}

/** Shortest-way exponential easing of an angle (the avatar turns to follow the person smoothly). */
export function easeAngle(current: number, target: number, dt: number, rate = 4) {
  const d = Math.atan2(Math.sin(target - current), Math.cos(target - current));
  return current + d * (1 - Math.exp(-rate * Math.max(0, dt)));
}

export type Focus = "left" | "centre" | "right" | null | undefined;

/** Looking at a real object: a bigger head turn and more eyes than a role shift, the torso barely follows. */
export const LOOK = { torsoDeg: 6, headDeg: 32, gaze: 0.55, easeMs: 420, leadMs: 0, holdGapMs: 0, releaseMs: 0 };

/**
 * Brain focus -> head/eye steps over a signed reply: look toward the thing during the first part of the reply (its
 * naming), then come back to the person. Side in the viewer's frame (left of the picture = left of the screen),
 * the same convention as the role shift. Centre / none: no turn (the avatar keeps facing the person).
 */
export function lookSteps(focus: Focus, durationMs: number): RoleStep[] {
  const side = focus === "left" ? -1 : focus === "right" ? 1 : 0;
  if (!side || !(durationMs > 0)) return [];
  const end = Math.min(durationMs * 0.6, Math.max(1600, durationMs * 0.4));
  return roleShiftSteps([{ start_ms: 0, end_ms: end, side }], LOOK);
}

export function lookAt(steps: RoleStep[], t: number): RoleShiftPose | null {
  return steps.length ? roleShiftAt(steps, t, LOOK) : null;
}

/** The hidden prompt of a spontaneous remark when the scene changed a lot (never shown as the person's turn). */
export const SCENE_REMARK = "La scène vient de changer. Dis-moi en une phrase ce que tu remarques de nouveau.";
