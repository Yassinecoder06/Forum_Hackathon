import * as THREE from "three";
import type { HandConstraints, HandJoint } from "./handConstraints";
import { findRigBone, type RigProfile } from "./rigProfiles";
import { locusSide, type RoleShiftPose } from "./roleShift";

/**
 * Pointing (index): in LSF a signer refers to a person or a thing by pointing the index finger at it. The overlay is
 * `pointAt(rig, command)`, with a TARGET that is either
 *  - a locus of signing space ({ side } -1 viewer's left ... +1 viewer's right; L / C / R of the cast sheet), for Film and
 *    Studio, or
 *  - a point of the real world ({ world }, in the avatar's world space, metres), for the AR companion: the arm, head and
 *    eyes go toward that position, whichever side it is on.
 * `motion.pointing` (lsf_order.pointing_spans) schedules the loci: each span points back to an existing `index` /
 * `mention` event of the gloss analysis (clause + gloss_index + referent id) and adds the time the event is signed.
 * `pointingAt` turns the spans into commands at a time t; an AR page can instead pass its own command per frame
 * (AvatarFrame.point). This module blends such a point on top of the generated motion:
 *
 * - Time curve (`pointingAt`): a pure function of the motion time. Each span has a weight that rises over `leadMs` before
 *   it starts and falls over `releaseMs` after it ends (smoothstep), so the arm never snaps. Spans of one hand blend
 *   their target side by weight. Same t -> same pose (seeking and frame-stepped export agree).
 * - Arm (`pointAt`): the arm on the target's own side points (viewer's left = the avatar's right arm; the centre uses the
 *   right arm). Two-bone IK puts the wrist on the line from the shoulder toward the target (body frame read from the
 *   CURRENT skeleton, so it follows the torso turn of the role shift), the elbow falls down and outward, the hand is
 *   turned so the index finger lies on that line, palm down and inward. `pointLook` gives the matching head and eye turn.
 * - Hand: handshape "index point" (index straight, other fingers curled, thumb across), set through the same
 *   anatomical angles as the hand limits, then `HandConstraints.constrain("hard")`, so no joint leaves its range.
 *
 * Only ADDS a pose over the retargeted one (the retargeter resets every bone each frame): nothing accumulates, and with
 * weight 0 the skeleton is not touched at all.
 */
export type PointSpan = {
  start_ms: number; end_ms: number; locus?: string | null; referent?: string | null; kind?: string;
  /** The gloss-analysis event this span comes from: clause index and gloss_index of its `index` / `mention` event. */
  clause?: number; gloss_index?: number | null;
};
export type PointTarget = { side: number } | { world: THREE.Vector3 };
/** One arm's point: how much (0..1) and where. `hand` is chosen from the target's side when absent. */
export type PointCommand = { weight: number; target: PointTarget; hand?: "Left" | "Right" };
export type PointPose = PointCommand[];

export const POINTING = {
  /** The arm is on its way this long before the point (the point itself is complete at the span start). */
  leadMs: 320,
  /** Return to the generated motion this long after the span end. */
  releaseMs: 380,
  /** Yaw of the pointing line from straight ahead for a left / right locus. */
  lateralDeg: 52,
  /** The line rises a little above the horizontal. */
  raiseDeg: 10,
  /** Wrist reach as a share of the straight arm: a natural, not locked, elbow. */
  reach: 0.93,
  /** Head turn as a share of the target's angle (max headMaxRad), and eye turn (eyeLook weight units at 60 degrees). */
  headShare: 0.5,
  headMaxRad: 0.45,
  gaze: 0.4,
};

/** Handshape "index point" (degrees): [MCP, PIP, DIP] flexion of the fingers, thumb [CMC flexion, CMC abduction, MCP, IP]. */
export const INDEX_POINT = {
  fingers: { Index: [0, 0, 0], Middle: [88, 100, 70], Ring: [88, 100, 70], Pinky: [88, 100, 70] } as Record<string, number[]>,
  thumb: [35, 5, 40, 25],
};

const smoothstep = (u: number) => (u <= 0 ? 0 : u >= 1 ? 1 : u * u * (3 - 2 * u));

/** The arm that points toward a side: the one on that side (viewer's left = the avatar's right arm); the centre uses the right. */
export const handForSide = (side: number): "Left" | "Right" => (side > 0 ? "Left" : "Right");

/** Weight (0..1) of a span at time t: eased in before its start, held, eased out after its end. */
export function spanWeight(s: { start_ms: number; end_ms: number }, t: number, opts = POINTING) {
  if (!(s.end_ms > s.start_ms)) return 0;
  return smoothstep((t - (s.start_ms - opts.leadMs)) / opts.leadMs) * (1 - smoothstep((t - s.end_ms) / opts.releaseMs));
}

/** Commands (one per arm at most) at time t, from the spans. Pure function of the spans and t; null when nothing points. */
export function pointingAt(spans: PointSpan[] | undefined | null, t: number, opts = POINTING): PointPose | null {
  if (!spans?.length) return null;
  const acc = { Left: { w: 0, ws: 0, sum: 0 }, Right: { w: 0, ws: 0, sum: 0 } };
  for (const s of spans) {
    if (s.locus !== "L" && s.locus !== "C" && s.locus !== "R") continue;
    const w = spanWeight(s, t, opts);
    if (w <= 1e-4) continue;
    const side = locusSide(s.locus), a = acc[handForSide(side)];
    a.w = Math.max(a.w, w);
    a.ws += w;
    a.sum += w * side;
  }
  const out: PointPose = [];
  for (const hand of ["Left", "Right"] as const) {
    const a = acc[hand];
    if (a.w > 1e-4) out.push({ weight: a.w, target: { side: a.sum / a.ws }, hand });
  }
  return out.length ? out : null;
}

type ArmBones = { upper: THREE.Object3D; fore: THREE.Object3D; hand: THREE.Object3D; index: THREE.Object3D[]; middle1: THREE.Object3D; pinky1: THREE.Object3D };
export type PointingRig = {
  arms: Record<"Left" | "Right", ArmBones>;
  hands: HandConstraints | null;
  /** Largest angle (radians) between the index finger and the pointing line at the end of the last applied point. */
  lastError: number;
};

export function createPointingRig(root: THREE.Object3D, profile: RigProfile, hands: HandConstraints | null): PointingRig | null {
  const arms: Partial<PointingRig["arms"]> = {};
  for (const side of ["Left", "Right"] as const) {
    const find = (n: string) => findRigBone(root, profile, side + n);
    const upper = find("Arm"), fore = find("ForeArm"), hand = find("Hand");
    const index = [1, 2, 3, 4].map((i) => find(`HandIndex${i}`)).filter((b): b is THREE.Object3D => Boolean(b));
    const middle1 = find("HandMiddle1"), pinky1 = find("HandPinky1");
    if (!upper || !fore || !hand || index.length < 3 || !middle1 || !pinky1) return null;
    arms[side] = { upper, fore, hand, index, middle1, pinky1 };
  }
  return { arms: arms as PointingRig["arms"], hands, lastError: 0 };
}

const UP = new THREE.Vector3(0, 1, 0);
const q1 = new THREE.Quaternion(), q2 = new THREE.Quaternion(), q3 = new THREE.Quaternion();
const wp = (b: THREE.Object3D, out = new THREE.Vector3()) => b.getWorldPosition(out);

/** Rotate a bone by a world-space rotation about its own position (children follow). */
function rotateWorld(bone: THREE.Object3D, delta: THREE.Quaternion) {
  if (!bone.parent) return;
  bone.parent.updateWorldMatrix(true, false);
  const world = bone.getWorldQuaternion(q1).premultiply(delta);
  bone.quaternion.copy(bone.parent.getWorldQuaternion(q2).invert().multiply(world));
  bone.updateWorldMatrix(false, true);
}

function setFingers(joints: HandJoint[], hands: HandConstraints, side: string) {
  const joint = (n: string) => joints.find((j) => j.name === side + n);
  const D = Math.PI / 180;
  const set = (n: string, flex: number, abd = 0) => {
    const j = joint(n);
    if (j) j.bone.quaternion.copy(hands.compose(j, { flex: flex * D, abd: abd * D, twist: 0 }));
  };
  for (const [finger, [mcp, pip, dip]] of Object.entries(INDEX_POINT.fingers)) {
    set(`Hand${finger}1`, mcp);
    set(`Hand${finger}2`, pip);
    set(`Hand${finger}3`, dip);
  }
  const [cmc, abd, mcp, ip] = INDEX_POINT.thumb;
  set("HandThumb1", cmc, abd);
  set("HandThumb2", mcp);
  set("HandThumb3", ip);
}

/** Body frame from the current skeleton: side = toward the viewer's right (the avatar's left shoulder), forward = facing. */
function bodyFrame(rig: PointingRig) {
  const side = wp(rig.arms.Left.upper).sub(wp(rig.arms.Right.upper));
  side.y = 0;
  if (side.lengthSq() < 1e-8) side.set(1, 0, 0);
  side.normalize();
  const forward = new THREE.Vector3().crossVectors(side, UP).normalize();
  return { side, forward };
}

/** Pointing line (unit, world) from the shoulder for a signed side: -1 viewer's left ... +1 viewer's right. */
export function pointingLine(side: number, frame: { side: THREE.Vector3; forward: THREE.Vector3 }, opts = POINTING) {
  const yaw = THREE.MathUtils.degToRad(opts.lateralDeg) * THREE.MathUtils.clamp(side, -1, 1);
  const raise = THREE.MathUtils.degToRad(opts.raiseDeg);
  const flat = frame.forward.clone().multiplyScalar(Math.cos(yaw)).addScaledVector(frame.side, Math.sin(yaw));
  return flat.multiplyScalar(Math.cos(raise)).addScaledVector(UP, Math.sin(raise)).normalize();
}

/** Direction of the index finger (wrist -> last index joint), world. */
function indexDirection(arm: ArmBones) {
  return wp(arm.index[arm.index.length - 1]).sub(wp(arm.hand)).normalize();
}

/** Body frame, pointing line and arm of a command (shared by the arm and the look). */
function resolve(rig: PointingRig, cmd: PointCommand) {
  const frame = bodyFrame(rig), t = cmd.target;
  let hand = cmd.hand;
  if (!hand) {
    const chest = wp(rig.arms.Left.upper).add(wp(rig.arms.Right.upper)).multiplyScalar(0.5);
    hand = handForSide("world" in t ? t.world.clone().sub(chest).dot(frame.side) : t.side);
  }
  const shoulder = wp(rig.arms[hand].upper);
  let line: THREE.Vector3;
  if ("world" in t) {
    line = t.world.clone().sub(shoulder);
    // a target at (or behind) the shoulder has no direction: straight ahead
    line = line.lengthSq() < 1e-6 ? frame.forward.clone() : line.normalize();
  } else line = pointingLine(t.side, frame);
  return { frame, hand, line, shoulder };
}

/**
 * Head and eye turn toward the target(s), as a RoleShiftPose to ADD to the role shift (torsoYaw 0). Call it before the
 * gaze weights are set and after the retargeting (it reads the current skeleton).
 */
export function pointLook(rig: PointingRig | null, pose: PointPose | PointCommand | null | undefined): RoleShiftPose | null {
  const cmds = Array.isArray(pose) ? pose : pose ? [pose] : [];
  const cmd = cmds.filter((c) => c.weight > 1e-4).sort((a, b) => b.weight - a.weight)[0];
  if (!rig || !cmd) return null;
  rig.arms.Left.upper.updateWorldMatrix(true, false);
  rig.arms.Right.upper.updateWorldMatrix(true, false);
  const { frame, line } = resolve(rig, cmd);
  const yaw = Math.atan2(line.dot(frame.side), line.dot(frame.forward));
  const w = THREE.MathUtils.clamp(cmd.weight, 0, 1);
  return {
    torsoYaw: 0,
    headYaw: w * THREE.MathUtils.clamp(yaw * POINTING.headShare, -POINTING.headMaxRad, POINTING.headMaxRad),
    gazeX: w * THREE.MathUtils.clamp(yaw / (60 * Math.PI / 180), -1, 1) * POINTING.gaze,
  };
}

/** Sum of two role-shift style poses (role shift + pointing look). */
export function addLook(a: RoleShiftPose | null | undefined, b: RoleShiftPose | null | undefined): RoleShiftPose | null {
  if (!a || !b) return a ?? b ?? null;
  return { torsoYaw: a.torsoYaw + b.torsoYaw, headYaw: a.headYaw + b.headYaw, gazeX: a.gazeX + b.gazeX };
}

/**
 * pointAt: blend the pointing pose over the current skeleton. Per arm: elbow and wrist by two-bone IK, handshape, hand
 * turned onto the line, the blend with the generated pose by `weight`, then the hand limits (hard) once at the end.
 * With weight 0 (or null) the skeleton is not touched.
 */
export function pointAt(rig: PointingRig | null, pose: PointPose | PointCommand | null | undefined) {
  if (!rig || !pose) return;
  const cmds = Array.isArray(pose) ? pose : [pose];
  for (const cmd of cmds) {
    if (!(cmd.weight > 1e-4)) continue;
    rig.arms.Left.upper.updateWorldMatrix(true, true);
    rig.arms.Right.upper.updateWorldMatrix(true, true);
    const { frame, hand: name, line } = resolve(rig, cmd);
    const arm = rig.arms[name], hands = rig.hands, weight = THREE.MathUtils.clamp(cmd.weight, 0, 1);
    const bones = [arm.upper, arm.fore, arm.hand, ...(hands ? hands.joints.filter((j) => j.side === name).map((j) => j.bone) : [])];
    const original = bones.map((b) => b.quaternion.clone());
    // --- two-bone IK: wrist on the line, elbow down and outward
    const S = wp(arm.upper), E = wp(arm.fore), W = wp(arm.hand);
    const a = E.distanceTo(S), b = W.distanceTo(E);
    const c = THREE.MathUtils.clamp((a + b) * POINTING.reach, Math.abs(a - b) + 1e-3, a + b - 1e-3);
    const T = S.clone().addScaledVector(line, c);
    const outward = frame.side.clone().multiplyScalar(name === "Right" ? -1 : 1);
    const pole = UP.clone().multiplyScalar(-1).addScaledVector(outward, 0.45).addScaledVector(frame.forward, -0.15);
    const along = line.clone();
    pole.addScaledVector(along, -pole.dot(along)).normalize();
    const x = (a * a - b * b + c * c) / (2 * c), h = Math.sqrt(Math.max(0, a * a - x * x));
    const E2 = S.clone().addScaledVector(along, x).addScaledVector(pole, h);
    rotateWorld(arm.upper, q3.setFromUnitVectors(E.clone().sub(S).normalize(), E2.clone().sub(S).normalize()).clone());
    const E3 = wp(arm.fore), W3 = wp(arm.hand);
    rotateWorld(arm.fore, q3.setFromUnitVectors(W3.sub(E3).normalize(), T.clone().sub(E3).normalize()).clone());
    // --- handshape, then the hand turned so the index lies on the line (palm down and inward)
    if (hands) setFingers(hands.joints, hands, name);
    arm.hand.updateWorldMatrix(false, true);
    rotateWorld(arm.hand, q3.setFromUnitVectors(indexDirection(arm), line).clone());
    const H = wp(arm.hand);
    const radial = wp(arm.index[0]).sub(wp(arm.pinky1));
    const palm = radial.cross(wp(arm.middle1).sub(H)).normalize().multiplyScalar(name === "Right" ? 1 : -1);
    const want = UP.clone().multiplyScalar(-0.85).addScaledVector(outward, -0.5);
    want.addScaledVector(line, -want.dot(line)).normalize();
    palm.addScaledVector(line, -palm.dot(line)).normalize();
    const roll = Math.atan2(line.dot(palm.clone().cross(want)), palm.dot(want));
    rotateWorld(arm.hand, q3.setFromAxisAngle(line, THREE.MathUtils.clamp(roll, -1.4, 1.4)).clone());
    // --- blend with the generated pose, then keep every hand joint inside its range
    bones.forEach((bone, i) => {
      const target = bone.quaternion.clone();
      bone.quaternion.copy(original[i]).slerp(target, weight);
    });
    arm.upper.updateWorldMatrix(false, true);
    hands?.constrain("hard");
    arm.upper.updateWorldMatrix(false, true);
    rig.lastError = Math.acos(THREE.MathUtils.clamp(indexDirection(arm).dot(line), -1, 1));
  }
}
