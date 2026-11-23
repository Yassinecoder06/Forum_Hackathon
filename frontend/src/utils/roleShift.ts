import * as THREE from "three";
import { findRigBone, type RigProfile } from "./rigProfiles";
import type { Locus } from "./speakers";

/**
 * Role shift (prise de rôle), as interpreters mark who is speaking: while the lines of speaker S are signed, the
 * upper body turns slightly toward S's locus in signing space, the head turns a little more, and the eyes follow;
 * the narrator is frontal. Turn changes ease in and out (smoothstep, ~0.3 s), so the body never snaps.
 *
 * - Time curve: a sum of eased steps, a pure function of the film time (deterministic: same t -> same pose, whatever
 *   the order of evaluation, so seeking and frame-stepped export give the same picture).
 * - Pose overlay: applied in AvatarCanvas AFTER retargeting, the performance layer and the grammatical head motion;
 *   the retargeter resets every bone each frame, so nothing accumulates. It only ADDS a yaw about the vertical axis:
 *   a negation head shake still oscillates around the shifted heading, a question's head tilt is kept (it turns with
 *   the body).
 *
 * Side convention: viewer's frame. side -1 = left of the screen (the avatar's right), +1 = right of the screen.
 * Angles are a first calibration (torso 12 deg, head 22 deg in total, gaze 0.3), to be tuned with an LSF reader.
 */
export type RoleShiftPose = {
  /** Upper-body yaw (radians, about +Y; positive turns toward the right of the screen). */
  torsoYaw: number;
  /** Total head yaw in the world (radians), including the torso's. */
  headYaw: number;
  /** Horizontal gaze in eyeLook* weight units (positive = toward the right of the screen). */
  gazeX: number;
};
export type RoleSegment = { start_ms: number; end_ms: number; side: number };
export type RoleStep = { at_ms: number; delta: number };

export const ROLE_SHIFT = {
  torsoDeg: 12,
  headDeg: 22,
  gaze: 0.3,
  /** Duration of a turn change (ease in/out). */
  easeMs: 320,
  /** The shift is complete when the segment starts (it starts easeMs/2 before). */
  leadMs: 160,
  /** Between two segments of the same side, hold the shift across gaps up to this long… */
  holdGapMs: 1500,
  /** …otherwise return to the narrator stance this long after the end of the segment. */
  releaseMs: 300,
};

export const FRONTAL: RoleShiftPose = { torsoYaw: 0, headYaw: 0, gazeX: 0 };

/** Locus of the cast sheet -> side (-1 left, 0 centre / narrator, +1 right of the screen). */
export function locusSide(locus?: Locus | string | null) {
  return locus === "L" ? -1 : locus === "R" ? 1 : 0;
}

const smoothstep = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));

/** Segments (any order, may overlap: the later start wins) -> eased steps of the side value. */
export function roleShiftSteps(segments: RoleSegment[], opts = ROLE_SHIFT): RoleStep[] {
  const segs = segments.filter((s) => s.end_ms > s.start_ms && Number.isFinite(s.side))
    .map((s) => ({ ...s, side: THREE.MathUtils.clamp(s.side, -1, 1) }))
    .sort((a, b) => a.start_ms - b.start_ms || a.end_ms - b.end_ms);
  const steps: RoleStep[] = [];
  let value = 0;
  const to = (at_ms: number, side: number) => {
    if (Math.abs(side - value) < 1e-9) return;
    steps.push({ at_ms, delta: side - value });
    value = side;
  };
  segs.forEach((s, i) => {
    to(s.start_ms - opts.leadMs, s.side);
    const next = segs[i + 1];
    const end = Math.min(s.end_ms, next ? Math.max(next.start_ms, s.start_ms) : s.end_ms);
    if (!next || next.start_ms - end > opts.holdGapMs) to(end + opts.releaseMs, 0);
  });
  return steps;
}

/** Side value (-1..1) at time t: sum of smoothstep transitions centred on each step. */
export function roleShiftValue(steps: RoleStep[], t: number, easeMs = ROLE_SHIFT.easeMs) {
  let v = 0;
  for (const s of steps) v += s.delta * smoothstep((t - (s.at_ms - easeMs / 2)) / easeMs);
  return THREE.MathUtils.clamp(v, -1, 1);
}

/** Pose at time t (radians / gaze weight). */
export function roleShiftAt(steps: RoleStep[], t: number, opts = ROLE_SHIFT): RoleShiftPose {
  const v = roleShiftValue(steps, t, opts.easeMs);
  if (Math.abs(v) < 1e-6) return FRONTAL;
  const rad = THREE.MathUtils.degToRad;
  return { torsoYaw: v * rad(opts.torsoDeg), headYaw: v * rad(opts.headDeg), gazeX: v * opts.gaze };
}

/** Bones the overlay turns: the spine chain (shared yaw), neck and head. */
export type RoleShiftRig = { spine: { bone: THREE.Object3D; share: number }[]; neck?: THREE.Object3D; head?: THREE.Object3D };

export function createRoleShiftRig(root: THREE.Object3D, profile: RigProfile): RoleShiftRig | null {
  const shares: [string, number][] = [["Spine", 0.3], ["Spine1", 0.35], ["Spine2", 0.35]];
  const spine = shares.map(([name, share]) => ({ bone: findRigBone(root, profile, name), share }))
    .filter((s): s is { bone: THREE.Object3D; share: number } => Boolean(s.bone));
  const total = spine.reduce((n, s) => n + s.share, 0);
  spine.forEach((s) => (s.share /= total || 1));
  const neck = findRigBone(root, profile, "Neck"), head = findRigBone(root, profile, "Head");
  return spine.length || head ? { spine, neck, head } : null;
}

const UP = new THREE.Vector3(0, 1, 0);
const qa = new THREE.Quaternion(), qb = new THREE.Quaternion(), qc = new THREE.Quaternion();

/** Rotate a bone by a world-space yaw about +Y, keeping its world position (children follow). */
export function yawWorld(bone: THREE.Object3D, angle: number) {
  if (!angle || !bone.parent) return;
  bone.parent.updateWorldMatrix(true, false);
  const world = bone.getWorldQuaternion(qa);
  const parent = bone.parent.getWorldQuaternion(qb).invert();
  bone.quaternion.copy(parent.multiply(qc.setFromAxisAngle(UP, angle).multiply(world)));
  bone.updateWorldMatrix(false, true);
}

/** Additive overlay on the current pose: torso yaw over the spine, the head's extra yaw over neck (40 %) + head (60 %). */
export function applyRoleShift(rig: RoleShiftRig | null, pose: RoleShiftPose | null | undefined) {
  if (!rig || !pose || (!pose.torsoYaw && !pose.headYaw)) return;
  const torso = rig.spine.length ? pose.torsoYaw : 0;
  for (const s of rig.spine) yawWorld(s.bone, torso * s.share);
  const extra = pose.headYaw - torso;
  if (rig.neck && rig.head) {
    yawWorld(rig.neck, extra * 0.4);
    yawWorld(rig.head, extra * 0.6);
  } else if (rig.head) yawWorld(rig.head, extra);
}

/** Film timeline -> role-shift segments: one per signed cue (side of its speaker), with the cue's quoted / reported
 * clauses (motion.role_shift) as sub-segments toward their own speaker. */
export function filmSegments(
  placed: { start_ms: number; end_ms: number }[],
  sides: (number | null | undefined)[],
  inner: ({ start_ms: number; end_ms: number; side: number }[] | undefined)[] = [],
): RoleSegment[] {
  const out: RoleSegment[] = [];
  placed.forEach((p, i) => {
    if (sides[i] === null || sides[i] === undefined || p.end_ms <= p.start_ms) return;
    const base = sides[i] as number;
    let t = p.start_ms;
    for (const span of [...(inner[i] ?? [])].sort((a, b) => a.start_ms - b.start_ms)) {
      const a = Math.max(t, p.start_ms + span.start_ms), b = Math.min(p.end_ms, p.start_ms + span.end_ms);
      if (b <= a) continue;
      if (a > t) out.push({ start_ms: t, end_ms: a, side: base });
      out.push({ start_ms: a, end_ms: b, side: span.side });
      t = b;
    }
    if (p.end_ms > t) out.push({ start_ms: t, end_ms: p.end_ms, side: base });
  });
  return out;
}
