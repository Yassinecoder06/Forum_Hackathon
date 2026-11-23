import * as THREE from "three";
import { findRigBone, type RigProfile } from "./rigProfiles";

/**
 * Hand realism layer: anatomical joint limits for the wrist and every finger joint, plus temporal smoothing of the
 * finger rotations. Applied to the retargeter's offline track (soft limits + smoothing, before hand contacts are
 * resolved) and again at runtime after the performance-layer springs (hard limits, which only catch overshoot).
 *
 * Each joint is described in its bone's bind-relative frame F = (parent pose) x (bind local rotation). F moves with
 * the parent segment, so the joint's local rotation relative to bind (`bindLocal^-1 * q`) is exactly the
 * articulation of that joint. Inside F, three axes come from the bind geometry (never from axis letters):
 *  - u: direction of the parent segment (forearm for the wrist, wrist->knuckle for the MCP, previous phalanx else);
 *  - p: the flexion direction, perpendicular to u: the palm normal (palmar side) for the wrist and the fingers,
 *       refined by the mesh's own bind curl for the interphalangeal joints; toward the little-finger base for the thumb;
 *  - s: the radial direction (toward the thumb), s = +-(u x p).
 * The bone's rotation is split into swing (where its child joint points) and twist (roll about the bone); for the
 * wrist the roll comes first (pronation/supination of the forearm, which the forearm twist bones display).
 * Flexion = atan2(d.p, d.u) in the (u, p) plane, abduction = asin(d.s) out of it, twist = roll about the bone axis.
 *
 * Limits (degrees). Source: anatomical references, not the UE5 rig (the Unreal editor was not reachable from this
 * runtime; a UE5 Control Rig / IK Rig export can replace JOINT_LIMITS later, same frame conventions):
 *  - American Academy of Orthopaedic Surgeons, "Joint Motion: Method of Measuring and Recording" (1965):
 *    wrist flexion 80 / extension 70, radial 20 / ulnar deviation 30; MCP 0-90; PIP 0-100; DIP 0-90;
 *    thumb MCP 0-50, thumb IP 0-80.
 *  - I. A. Kapandji, "The Physiology of the Joints", vol. 1 (upper limb): MCP hyperextension up to 20-30 (passive),
 *    lateral MCP motion largest for index and little finger, smallest for the middle finger, and lost in flexion
 *    (collateral ligaments tighten); interphalangeal joints are pure hinges.
 *  - M. C. Hume et al., "Functional range of motion of the joints of the hand", J. Hand Surg. 15A (1990).
 *  - J. Lee, T. L. Kunii, "Model-based analysis of hand posture", IEEE CG&A (1995): hinge IP joints, coupled
 *    abduction, small axial rotation.
 * Each range ends a few degrees beyond the active range of those references (the soft zone then starts inside it),
 * so a straight finger or a full fist is kept as is and only poses a signer cannot make are bent back. The extreme
 * handshapes of LSF (fist, flat hand, index point, pinch, thumb-up, spread-5) stay inside the limits.
 */

export type JointType = "wrist" | "mcp" | "pip" | "dip" | "thumbCmc" | "thumbMcp" | "thumbIp";
export type Range = readonly [number, number];
export type JointLimit = {
  /** Flexion range (positive = toward the palm). `anatomy`: angle to the parent segment; `bind`: relative to bind. */
  flex: Range;
  flexFrom: "anatomy" | "bind";
  /** Abduction relative to the bind pose (positive = radial, toward the thumb). Shrinks with flexion when `abdFlexLoss`. */
  abd: Range;
  /** Fraction of the abduction range lost at full flexion (MCP collateral ligaments). */
  abdFlexLoss?: number;
  /** Axial twist relative to the bind pose, +-. */
  twist: number;
  /** Width of the soft zone at each end of a range (the limit is approached asymptotically, never snapped). */
  soft: number;
};

const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"] as const;
type Finger = (typeof FINGERS)[number];

const MCP_ABDUCTION: Record<Exclude<Finger, "Thumb">, number> = { Index: 20, Middle: 14, Ring: 17, Pinky: 22 };

export function jointLimit(type: JointType, finger?: Finger): JointLimit {
  switch (type) {
    // Pronation/supination is carried by the hand bone in these rigs (the forearm twist bones only display it).
    case "wrist": return { flex: [-70, 80], flexFrom: "anatomy", abd: [-30, 20], twist: 100, soft: 10 };
    case "mcp": {
      const a = MCP_ABDUCTION[(finger ?? "Index") as Exclude<Finger, "Thumb">] ?? 18;
      return { flex: [-20, 95], flexFrom: "anatomy", abd: [-a, a], abdFlexLoss: 0.65, twist: 10, soft: 8 };
    }
    case "pip": return { flex: [-5, 110], flexFrom: "anatomy", abd: [-5, 5], twist: 8, soft: 8 };
    case "dip": return { flex: [-5, 85], flexFrom: "anatomy", abd: [-5, 5], twist: 8, soft: 8 };
    // Saddle joint: a wide swing cone around the relaxed thumb, no backwards (dorsal) collapse, bounded opposition roll.
    case "thumbCmc": return { flex: [-30, 55], flexFrom: "bind", abd: [-45, 45], twist: 25, soft: 10 };
    case "thumbMcp": return { flex: [-10, 65], flexFrom: "anatomy", abd: [-10, 10], twist: 10, soft: 8 };
    case "thumbIp": return { flex: [-15, 85], flexFrom: "anatomy", abd: [-6, 6], twist: 8, soft: 8 };
  }
}

export type HandJoint = {
  /** Canonical bone name (LeftHand, LeftHandIndex1...). */
  name: string;
  side: "Left" | "Right";
  finger: Finger | null;
  type: JointType;
  bone: THREE.Object3D;
  bindLocal: THREE.Quaternion;
  /** Axes in the bone's bind-relative frame F (unit vectors). */
  t: THREE.Vector3;
  u: THREE.Vector3;
  p: THREE.Vector3;
  s: THREE.Vector3;
  /** Anatomical flexion and abduction of the bind pose (radians). */
  flex0: number;
  abd0: number;
  limit: JointLimit;
};

export type JointAngles = { flex: number; abd: number; twist: number };
export type Violation = "flexion" | "hyperextension" | "abduction" | "twist";

const DEG = Math.PI / 180;
const clamp = THREE.MathUtils.clamp;

/** C1-continuous limit: identity inside [lo + k, hi - k], asymptotic to the limit beyond (never exceeds it). */
export function softClamp(x: number, lo: number, hi: number, k: number) {
  k = Math.min(k, (hi - lo) / 2);
  if (k <= 1e-9) return clamp(x, lo, hi);
  if (x > hi - k) return hi - k + k * Math.tanh((x - (hi - k)) / k);
  if (x < lo + k) return lo + k - k * Math.tanh((lo + k - x) / k);
  return x;
}

const tmpQ = new THREE.Quaternion(), tmpQ2 = new THREE.Quaternion(), tmpV = new THREE.Vector3();

/** q = swing * twist, twist about the unit axis `axis`. Returns the signed twist angle (radians). */
function twistAngle(q: THREE.Quaternion, axis: THREE.Vector3) {
  const d = q.x * axis.x + q.y * axis.y + q.z * axis.z;
  let w = q.w, x = d;
  if (w < 0) { w = -w; x = -x; }
  if (w * w + x * x < 1e-12) return 0;
  return 2 * Math.atan2(x, w);
}

/** The 2 pi branch of `angle` closest to `ref`. */
function closest(angle: number, ref: number) {
  return angle + 2 * Math.PI * Math.round((ref - angle) / (2 * Math.PI));
}

/** Wrap an angle by 2 pi when that brings it closer to [lo, hi] (a full curl is not a backwards bend). */
function nearest(angle: number, lo: number, hi: number) {
  const gap = (a: number) => (a < lo ? lo - a : a > hi ? a - hi : 0);
  for (const c of [angle - 2 * Math.PI, angle + 2 * Math.PI]) if (gap(c) < gap(angle)) angle = c;
  return angle;
}

export class HandConstraints {
  readonly joints: HandJoint[] = [];
  /** Joint-samples changed by the last constrainTrack call. */
  limitedSamples = 0;

  /** Build from the skeleton in its BIND pose (the retargeter's constructor contract). */
  constructor(root: THREE.Object3D, profile: RigProfile) {
    root.updateMatrixWorld(true);
    const find = (name: string) => findRigBone(root, profile, name);
    const pos = (name: string) => find(name)?.getWorldPosition(new THREE.Vector3());
    for (const side of ["Left", "Right"] as const) {
      const hand = find(side + "Hand"), wrist = pos(side + "Hand"), fore = pos(side + "ForeArm");
      const index1 = pos(side + "HandIndex1"), middle1 = pos(side + "HandMiddle1"), pinky1 = pos(side + "HandPinky1");
      if (!hand || !wrist || !fore || !index1 || !middle1 || !pinky1) continue;
      // Palm frame. Anatomy is mirrored: for the right hand (index - pinky) x (middle - wrist) points out of the palm,
      // for the left hand it points out of the back of the hand.
      const radial = index1.clone().sub(pinky1).normalize();
      const palmar = radial.clone().cross(middle1.clone().sub(wrist)).normalize().multiplyScalar(side === "Right" ? 1 : -1);
      /** flexDir: flexion direction (world, any length); hinge: a hinge axis to use instead (world, unit). */
      const add = (name: string, type: JointType, finger: Finger | null, parent: THREE.Vector3, child: THREE.Vector3 | undefined, hinge: THREE.Vector3 | null, side0: THREE.Vector3) => {
        const bone = find(name), at = pos(name);
        if (!bone || !at) return;
        const u = at.clone().sub(parent).normalize();
        const t = child ? child.clone().sub(at).normalize() : u.clone();
        // Flexion direction perpendicular to the parent segment: rotating u about the hinge h moves it toward h x u.
        const flexDir = hinge ? hinge.clone().cross(u) : side0.clone();
        flexDir.addScaledVector(u, -flexDir.dot(u)).normalize();
        if (flexDir.dot(side0) < 0) flexDir.negate();
        const s = u.clone().cross(flexDir).normalize();
        if (s.dot(radial) < 0) s.negate();
        const world = bone.getWorldQuaternion(new THREE.Quaternion()).invert();
        const local = (v: THREE.Vector3) => v.clone().applyQuaternion(world).normalize();
        const joint: HandJoint = {
          name, side, finger, type, bone, bindLocal: bone.quaternion.clone(),
          t: local(t), u: local(u), p: local(flexDir), s: local(s), flex0: 0, abd0: 0, limit: jointLimit(type, finger ?? undefined),
        };
        joint.flex0 = Math.atan2(joint.t.dot(joint.p), joint.t.dot(joint.u));
        joint.abd0 = Math.asin(clamp(joint.t.dot(joint.s), -1, 1));
        this.joints.push(joint);
      };
      add(side + "Hand", "wrist", null, fore, middle1, null, palmar);
      for (const finger of FINGERS) {
        const names = [1, 2, 3, 4].map((i) => `${side}Hand${finger}${i}`), p = names.map(pos);
        if (!p[0] || !p[1] || !p[2]) continue;
        const thumb = finger === "Thumb";
        const types: JointType[] = thumb ? ["thumbCmc", "thumbMcp", "thumbIp"] : ["mcp", "pip", "dip"];
        const tip = p[3] ?? p[2].clone().add(p[2].clone().sub(p[1]));
        // The finger's hinge: the mesh's bind curl at the PIP (thumb: MCP) joint when it is visible and closes toward
        // the palm (thumb: toward the little-finger base); else perpendicular to the segment and the palm normal.
        // Real fingers flex in one plane per finger (they converge in a fist), so MCP, PIP and DIP share this hinge.
        const toward = (at: THREE.Vector3) => (thumb ? pinky1.clone().sub(at) : palmar.clone());
        const a = p[1].clone().sub(p[0]).normalize(), b = p[2].clone().sub(p[1]).normalize();
        const curl = b.clone().addScaledVector(a, -b.dot(a));
        const ref = toward(p[1]).addScaledVector(a, -toward(p[1]).dot(a)).normalize();
        const hinge = curl.length() > Math.sin(4 * DEG) && curl.clone().normalize().dot(ref) > Math.cos(50 * DEG)
          ? a.clone().cross(curl).normalize()
          : a.clone().cross(ref).normalize();
        // Knuckles keep the palm as their reference when the finger's own plane is far from it (unusual binds).
        const mcpRef = toward(p[0]).addScaledVector(a, -toward(p[0]).dot(a)).normalize();
        const mcpHinge = hinge.clone().cross(a).normalize().dot(mcpRef) > Math.cos(35 * DEG) ? hinge : null;
        add(names[0], types[0], finger, wrist, p[1], mcpHinge, toward(p[0]));
        add(names[1], types[1], finger, p[0], p[2], hinge, toward(p[1]));
        add(names[2], types[2], finger, p[1], tip, hinge, toward(p[2]));
      }
    }
  }

  /**
   * Anatomical angles (radians) of a local rotation `q` of joint `j`. Angles are defined modulo 2 pi: without `near`
   * the branch closest to the limits is taken; with `near` (the previous frame of a track) the branch closest to it,
   * so a hand that keeps turning past 180 degrees is seen as turning on, not as flipping to the other side.
   */
  angles(j: HandJoint, q: THREE.Quaternion, near?: JointAngles): JointAngles {
    const delta = tmpQ.copy(j.bindLocal).invert().multiply(q);
    let twist: number;
    if (j.type === "wrist") {
      // Pronation/supination happens before the wrist (radius about ulna): delta = roll(t) * swing, and the
      // flexion/deviation are read in the rolled frame.
      twist = -twistAngle(tmpQ2.copy(delta).invert(), j.t);
      delta.premultiply(tmpQ2.setFromAxisAngle(j.t, -twist));
    } else twist = twistAngle(delta, j.t);
    const d = tmpV.copy(j.t).applyQuaternion(delta);
    let flex = Math.atan2(d.dot(j.p), d.dot(j.u));
    if (j.limit.flexFrom === "bind") flex -= j.flex0;
    const lo = j.limit.flex[0] * DEG, hi = j.limit.flex[1] * DEG;
    flex = near ? closest(flex, near.flex) : nearest(flex, lo, hi);
    twist = near ? closest(twist, near.twist) : twist;
    const abd = Math.asin(clamp(d.dot(j.s), -1, 1)) - j.abd0;
    return { flex, abd, twist };
  }

  /** Abduction range (radians, relative to bind) at a given flexion. */
  private abdRange(j: HandJoint, flex: number): [number, number] {
    const loss = j.limit.abdFlexLoss ? j.limit.abdFlexLoss * THREE.MathUtils.smoothstep(flex / DEG, 25, 90) : 0;
    return [j.limit.abd[0] * DEG * (1 - loss), j.limit.abd[1] * DEG * (1 - loss)];
  }

  /** Limit violations of a local rotation beyond the hard limits (with `tolerance` degrees of slack). */
  violations(j: HandJoint, q: THREE.Quaternion, tolerance = 1): Violation[] {
    const a = this.angles(j, q), tol = tolerance * DEG, out: Violation[] = [];
    if (a.flex > j.limit.flex[1] * DEG + tol) out.push("flexion");
    if (a.flex < j.limit.flex[0] * DEG - tol) out.push("hyperextension");
    const [lo, hi] = this.abdRange(j, clamp(a.flex, j.limit.flex[0] * DEG, j.limit.flex[1] * DEG));
    if (a.abd > hi + tol || a.abd < lo - tol) out.push("abduction");
    if (Math.abs(a.twist) > j.limit.twist * DEG + tol) out.push("twist");
    return out;
  }

  /**
   * Local rotation of joint `j` with its limits enforced. `soft`: smooth asymptotic limits (offline track);
   * `hard`: plain clamping at the limits (idempotent; runtime guard after springs). Returns `q` itself when it is
   * already inside the limits, so valid poses are kept bit for bit.
   */
  limitRotation(j: HandJoint, q: THREE.Quaternion, mode: "soft" | "hard" = "soft", a: JointAngles = this.angles(j, q)): THREE.Quaternion {
    const L = j.limit, k = (mode === "soft" ? L.soft : 0) * DEG;
    const fit = (x: number, lo: number, hi: number) => (mode === "soft" ? softClamp(x, lo, hi, Math.min(k, (hi - lo) / 4)) : clamp(x, lo, hi));
    const flex = fit(a.flex, L.flex[0] * DEG, L.flex[1] * DEG);
    const [alo, ahi] = this.abdRange(j, flex);
    const abd = fit(a.abd, alo, ahi);
    const twist = fit(a.twist, -L.twist * DEG, L.twist * DEG);
    if (Math.abs(flex - a.flex) < 1e-6 && Math.abs(abd - a.abd) < 1e-6 && Math.abs(twist - a.twist) < 1e-6) return q;
    return this.compose(j, { flex, abd, twist });
  }

  /** Local rotation of joint `j` with the given anatomical angles (radians): the inverse of `angles`. */
  compose(j: HandJoint, a: JointAngles): THREE.Quaternion {
    // Child direction from (flex, abd), minimal swing from the bind axis, then the twist about the bone.
    const f = a.flex + (j.limit.flexFrom === "bind" ? j.flex0 : 0), b = a.abd + j.abd0;
    const d = j.u.clone().multiplyScalar(Math.cos(f)).addScaledVector(j.p, Math.sin(f)).multiplyScalar(Math.cos(b)).addScaledVector(j.s, Math.sin(b)).normalize();
    const swing = new THREE.Quaternion().setFromUnitVectors(j.t, d);
    const roll = new THREE.Quaternion().setFromAxisAngle(j.t, a.twist);
    return j.type === "wrist" ? j.bindLocal.clone().multiply(roll).multiply(swing) : j.bindLocal.clone().multiply(swing).multiply(roll);
  }

  /** Enforce the limits on the skeleton's current pose (local rotations only; no world update is needed). */
  constrain(mode: "soft" | "hard" = "hard") {
    let changed = 0;
    for (const j of this.joints) {
      const q = this.limitRotation(j, j.bone.quaternion, mode);
      if (q !== j.bone.quaternion) { j.bone.quaternion.copy(q); changed++; }
    }
    return changed;
  }

  /**
   * Offline pass over a whole rotation track (the retargeter's per-keyframe local rotations, indexed by `names`):
   * soft limits on the wrist and all finger joints, then zero-phase One-Euro smoothing of the finger rotations on the
   * motion clock, then a hard guard. A pure function of the track: seeking any time gives the same pose.
   */
  constrainTrack(frames: { time: number; rotations: THREE.Quaternion[] }[], names: string[], smoothing: OneEuroParams | null = FINGER_SMOOTHING,
    types?: JointType[]) {
    let limited = 0;
    for (const j of this.joints) {
      const index = names.indexOf(j.name);
      if (index < 0 || (types && !types.includes(j.type))) continue;
      let previous: JointAngles | undefined;
      frames.forEach((f, i) => {
        // continuity over time, restarted after a gap (the first frame takes the branch closest to the limits)
        const a = this.angles(j, f.rotations[index], i && f.time - frames[i - 1].time <= 250 ? previous : undefined);
        previous = a;
        const q = this.limitRotation(j, f.rotations[index], "soft", a);
        if (q !== f.rotations[index]) { f.rotations[index] = q.clone(); limited++; }
      });
      if (smoothing && j.type !== "wrist") {
        const smoothed = smoothRotations(frames.map((f) => f.time), frames.map((f) => f.rotations[index]), smoothing);
        frames.forEach((f, i) => { f.rotations[index] = this.limitRotation(j, smoothed[i], "hard").clone(); });
      }
    }
    this.limitedSamples = limited;
    return limited;
  }
}

export type OneEuroParams = { minCutoff: number; beta: number; derivativeCutoff: number };
/** Finger jitter of monocular capture sits around 5-12 Hz; real handshape changes are fast, so the cutoff rises with
 * speed (One-Euro): about 3.5 Hz on holds, about 14 Hz on a 90-degree change in 150 ms. */
export const FINGER_SMOOTHING: OneEuroParams = { minCutoff: 3.5, beta: 1.0, derivativeCutoff: 1.5 };

const alpha = (cutoff: number, dt: number) => 1 / (1 + 1 / (2 * Math.PI * cutoff * dt));

function oneEuroPass(times: number[], rotations: THREE.Quaternion[], params: OneEuroParams) {
  const out = rotations.map((q) => q.clone());
  let speed = 0;
  for (let i = 1; i < out.length; i++) {
    const dt = Math.abs(times[i] - times[i - 1]) / 1000;
    if (dt <= 0) { out[i].copy(out[i - 1]); continue; }
    if (dt > 0.25) { speed = 0; continue; }     // a gap in the track restarts the filter
    const target = rotations[i];
    const rawSpeed = out[i - 1].angleTo(target) / dt;
    speed += (rawSpeed - speed) * alpha(params.derivativeCutoff, dt);
    out[i].copy(out[i - 1]).slerp(target, alpha(params.minCutoff + params.beta * Math.abs(speed), dt));
  }
  return out;
}

/** Zero-phase (forward then backward) One-Euro filter of a rotation track: no lag, sign timing is kept. */
export function smoothRotations(times: number[], rotations: THREE.Quaternion[], params: OneEuroParams = FINGER_SMOOTHING) {
  if (rotations.length < 3) return rotations.map((q) => q.clone());
  const forward = oneEuroPass(times, rotations, params);
  return oneEuroPass([...times].reverse(), forward.reverse(), params).reverse();
}

/** Count limit violations of the skeleton's current pose, per joint type and kind (measurement tool, tests). */
export function countViolations(hands: HandConstraints, tally: Record<string, number> = {}, tolerance = 1) {
  for (const j of hands.joints) {
    tally[`${j.type}.samples`] = (tally[`${j.type}.samples`] ?? 0) + 1;
    for (const v of hands.violations(j, j.bone.quaternion, tolerance)) tally[`${j.type}.${v}`] = (tally[`${j.type}.${v}`] ?? 0) + 1;
  }
  return tally;
}
