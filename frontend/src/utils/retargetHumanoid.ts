import * as THREE from "three";
import { interpolateJointPositions, type InterpolatedJoints } from "./motionInterpolator";
import type { MotionFile } from "../types/contract";
import { frameInterval, hasHand, stabilizeRotations, type HandState } from "./avatarPlayback";
import { handContactSets, type CapsuleSet, type CapsuleSource, type CapsuleSpec, type ContactResult } from "./handContact";
import { detectRigProfile, rigBoneName, type RigProfile } from "./rigProfiles";
import { HandConstraints, softClamp } from "./handConstraints";
const fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const SIDES = ["Left", "Right"] as const;
type Side = (typeof SIDES)[number];
type Bind = {
  bone: THREE.Bone;
  local: THREE.Quaternion;
  world: THREE.Quaternion;
  direction: THREE.Vector3;
};
const vec = (a: number[]) => new THREE.Vector3(a[0], a[1], a[2]);
/** Reference (median) body pose of a generated clip: the body layer moves the torso, shoulders and head by the
 * DEVIATION from it, so a clip's average posture keeps the avatar's own rest posture. */
type BodyRef = { neck: THREE.Vector3; shrug: [number, number]; head: THREE.Vector3; eyeRoll: number | null };
const median = (a: number[]) => { const s = [...a].sort((x, y) => x - y); return s.length ? s[s.length >> 1] : 0; };
/** Body following is enabled for motions produced by the LSF engines (whole-body generated poses). Captured clips
 * keep the original arms-and-hands behaviour. */
export function followsBody(motion: MotionFile) {
  const m = motion as MotionFile & { capture_method?: string };
  return Boolean(m.generation) || Boolean(m.capture_method?.startsWith("lsf-engine"));
}
function bodyReference(motion: MotionFile): BodyRef | null {
  const f = motion.frames.map((x) => x.joints).filter((j) => j.neck && j.left_shoulder && j.right_shoulder && j.head);
  if (!f.length) return null;
  const eyes = f.filter((j) => j.left_eye && j.right_eye);
  return {
    neck: new THREE.Vector3(median(f.map((j) => j.neck[0])), median(f.map((j) => j.neck[1])), median(f.map((j) => j.neck[2]))),
    shrug: [median(f.map((j) => j.left_shoulder[1] - j.neck[1])), median(f.map((j) => j.right_shoulder[1] - j.neck[1]))],
    head: new THREE.Vector3(median(f.map((j) => j.head[0] - j.neck[0])), median(f.map((j) => j.head[1] - j.neck[1])), median(f.map((j) => j.head[2] - j.neck[2]))),
    eyeRoll: eyes.length ? median(eyes.map((j) => Math.atan2(j.left_eye[1] - j.right_eye[1], j.left_eye[0] - j.right_eye[0]))) : null,
  };
}
function basis(
  wrist: THREE.Vector3,
  index: THREE.Vector3,
  middle: THREE.Vector3,
  pinky: THREE.Vector3,
) {
  const x = index.clone().sub(pinky), y = middle.clone().sub(wrist);
  if (x.length() < .008 || y.length() < .015) return null;
  x.normalize(); y.normalize();
  const z = new THREE.Vector3().crossVectors(x, y);
  if (z.lengthSq() < .04) return null;
  z.normalize();
  y.crossVectors(z, x).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(x, y, z),
  );
}

/**
 * Torso volume measured on the avatar's own meshes (body + clothes) in the rest pose: the vertices skinned mostly to
 * the pelvis / spine bones, cut in 2 cm horizontal slices; each slice keeps its centre and, per direction around it,
 * the farthest vertex (a star-shaped outline: chest, breasts, belly, back). Arm points are tested in the chest bone's
 * frame, so the volume follows the body when it leans or turns.
 */
type Torso = { chest: THREE.Object3D; restChest: THREE.Matrix4; y0: number; step: number; slices: ({ cx: number; cz: number; r: Float32Array } | null)[] };
const TORSO_BONE = /(^|[^a-z])(pelvis|hips|spine|chest)/i, TORSO_BINS = 48;
function measureTorso(root: THREE.Object3D, chest: THREE.Object3D | undefined): Torso | null {
  if (!chest) return null;
  root.updateMatrixWorld(true);
  const points: THREE.Vector3[] = [];
  root.traverse((o) => {
    const mesh = o as THREE.SkinnedMesh;
    if (!mesh.isSkinnedMesh || !mesh.geometry?.attributes.skinIndex) return;
    mesh.skeleton.update();
    const index = mesh.geometry.attributes.skinIndex, weight = mesh.geometry.attributes.skinWeight;
    const count = mesh.geometry.attributes.position.count, every = Math.max(1, Math.floor(count / 40000));
    for (let i = 0; i < count; i += every) {
      let best = -1, bw = 0;
      for (let k = 0; k < 4; k++) { const w = weight.getComponent(i, k); if (w > bw) { bw = w; best = index.getComponent(i, k); } }
      const bone = mesh.skeleton.bones[best];
      if (!bone || !TORSO_BONE.test(bone.name)) continue;
      points.push(mesh.getVertexPosition(i, new THREE.Vector3()).applyMatrix4(mesh.matrixWorld));
    }
  });
  if (points.length < 200) return null;
  const step = 0.02, y0 = Math.min(...points.map((p) => p.y)), n = Math.ceil((Math.max(...points.map((p) => p.y)) - y0) / step) + 1;
  const groups: THREE.Vector3[][] = Array.from({ length: n }, () => []);
  for (const p of points) groups[Math.floor((p.y - y0) / step)].push(p);
  const slices = groups.map((g) => {
    if (g.length < 20) return null;
    const xs = g.map((p) => p.x), zs = g.map((p) => p.z);
    const cx = (Math.min(...xs) + Math.max(...xs)) / 2, cz = (Math.min(...zs) + Math.max(...zs)) / 2;
    const r = new Float32Array(TORSO_BINS);
    for (const p of g) {
      const b = Math.floor(((Math.atan2(p.z - cz, p.x - cx) + Math.PI) / (2 * Math.PI)) * TORSO_BINS) % TORSO_BINS;
      r[b] = Math.max(r[b], Math.hypot(p.x - cx, p.z - cz));
    }
    for (let pass = 0; pass < TORSO_BINS && r.some((v) => !v); pass++) // empty directions take their neighbours' outline
      r.forEach((v, b) => { if (!v) r[b] = Math.max(r[(b + 1) % TORSO_BINS], r[(b + TORSO_BINS - 1) % TORSO_BINS]); });
    return { cx, cz, r };
  });
  return { chest, restChest: chest.matrixWorld.clone(), y0, step, slices };
}

/**
 * World transforms of the bones the retargeter reads (its mapped bones and their ancestors, ~75 of MetaHuman's 912),
 * computed lazily, parent before child, with three.js's own operations (compose, multiplyMatrices, decompose), so the
 * numbers are exactly those of `getWorldPosition` / `updateMatrixWorld`. Changing a bone only marks it and its cached
 * descendants stale; nothing else of the skeleton (face joints, correctives, meshes) is touched while deriving.
 * Bones outside the cache keep stale world matrices until the next full update (applyMotion / the renderer).
 */
class WorldCache {
  readonly index = new Map<THREE.Object3D, number>();
  private nodes: THREE.Object3D[] = [];
  private parent: number[] = [];
  private end: number[] = [];
  private valid: Uint8Array;
  private qValid: Uint8Array;
  private wq: THREE.Quaternion[];
  private static p = new THREE.Vector3();
  private static s = new THREE.Vector3();
  constructor(private root: THREE.Object3D, bones: THREE.Object3D[]) {
    const keep = new Set<THREE.Object3D>();
    for (const b of bones) {
      let o: THREE.Object3D | null = b;
      while (o && o !== root) { keep.add(o); o = o.parent; }
      if (o !== root) throw new Error("bone outside the avatar");
    }
    // preorder: a node's descendants are the contiguous range [i, end[i])
    const visit = (o: THREE.Object3D, parent: number) => {
      const i = this.nodes.length;
      this.nodes.push(o); this.parent.push(parent); this.end.push(0); this.index.set(o, i);
      for (const c of o.children) if (keep.has(c)) visit(c, i);
      this.end[i] = this.nodes.length;
    };
    visit(root, -1);
    this.valid = new Uint8Array(this.nodes.length);
    this.qValid = new Uint8Array(this.nodes.length);
    this.wq = this.nodes.map(() => new THREE.Quaternion());
  }
  /** Everything may have changed (start of a pose, or another layer moved bones): recompute on demand. */
  invalidateAll() {
    this.root.parent?.updateWorldMatrix(true, false);
    this.valid.fill(0);
    this.qValid.fill(0);
  }
  /** A node's local transform changed: it and its descendants are stale. */
  invalidate(i: number) {
    this.valid.fill(0, i, this.end[i]);
    this.qValid.fill(0, i, this.end[i]);
  }
  ensure(i: number) {
    if (this.valid[i]) return;
    const p = this.parent[i];
    if (p >= 0) this.ensure(p);
    const o = this.nodes[i];
    // exactly Object3D.updateWorldMatrix(false, false)
    if (o.matrixAutoUpdate) o.updateMatrix();
    if (o.matrixWorldAutoUpdate === true) {
      if (o.parent === null) o.matrixWorld.copy(o.matrix);
      else o.matrixWorld.multiplyMatrices(o.parent.matrixWorld, o.matrix);
    }
    this.valid[i] = 1;
  }
  quaternion(i: number, target: THREE.Quaternion) {
    this.ensure(i);
    if (!this.qValid[i]) { this.nodes[i].matrixWorld.decompose(WorldCache.p, this.wq[i], WorldCache.s); this.qValid[i] = 1; }
    return target.copy(this.wq[i]);
  }
}

type CapsuleBone = { a: THREE.Object3D; b: THREE.Object3D; ia: number; ib: number; sa: string; sb: string };
type FingerJoint = {
  bind: Bind; finger: string; i: number; a: string; b: string; previous: string | null;
  curlWorld: THREE.Vector3; flexAxis: THREE.Vector3; spreadAxis: THREE.Vector3; straight: number;
};
const noop = () => {};

/** Arm IK and calibrated local finger articulation: bind translations never stretch.
 * Every call starts from bind transforms, so seeking is deterministic. */
export class HumanoidRetargeter {
  private binds = new Map<string, Bind>();
  private palms = new Map<string, THREE.Quaternion>();
  /** Elbow hinge axis per side, in the upper arm's local frame (from the bind pose). */
  private hinges = new Map<string, THREE.Vector3>();
  private neckOrigin = new THREE.Vector3();
  private torso: Torso | null = null;
  private toRest = new THREE.Matrix4();
  private fromRest = new THREE.Quaternion();
  /** Keep the arms out of the torso (measured on the avatar's meshes). Off only for comparisons. */
  torsoGuard = true;
  private torsoFixed = 0;
  private torsoMaxDepth = 0;
  private shoulderWidth = 0.36;
  private tracks = new WeakMap<MotionFile, ReturnType<HumanoidRetargeter["prepare"]>>();
  /** Set while preparing an engine motion; null = arms/hands only (original behaviour). */
  private bodyRef: BodyRef | null = null;
  diagnostics = { preparedMs: 0, rejectedSpikes: 0, correctedFrames: 0, contactFrames: 0, maxCorrection: 0, maxPenetration: 0, handLimited: 0 };
  /** Anatomical wrist/finger limits and finger smoothing (handConstraints.ts); null when disabled. */
  readonly hands: HandConstraints | null;
  /** Bind segment lengths for explicitly mapped rigs (RPM keeps local translations). */
  private lengths = new Map<string, number>();
  // Static data derived once from the bind pose (see the constructor's end).
  private names: string[];
  private bindList: Bind[];
  private world: WorldCache;
  private capsuleBones = {} as Record<Side, CapsuleBone[]>;
  private capsuleSpec = {} as Record<Side, CapsuleSpec[]>;
  /** Capsules without the forearm (the forearm segment is always last). */
  private capsuleHands = {} as Record<Side, number>;
  private fingerJoints = {} as Record<Side, FingerJoint[]>;
  private palmInverse = new Map<string, THREE.Quaternion>();
  private foreArmRest = new Map<string, THREE.Vector3>();
  private bestPose: THREE.Quaternion[];
  private hitA: ContactResult = { correction: new THREE.Vector3(), maxPenetration: 0, maxRawPenetration: 0, contact: false };
  private hitB: ContactResult = { correction: new THREE.Vector3(), maxPenetration: 0, maxRawPenetration: 0, contact: false };
  private quietDepth = 0;
  private quietSaved: (() => void)[] = [];
  private quietBones: THREE.Bone[];
  constructor(private root: THREE.Object3D, readonly profile: RigProfile = detectRigProfile(root), options: { handConstraints?: boolean } = {}) {
    root.updateMatrixWorld(true);
    // Built from the bind pose, before any bone moves.
    this.hands = options.handConstraints === false ? null : new HandConstraints(root, profile);
    // Bones are keyed by canonical (Ready Player Me) names. RPM rigs keep every bone
    // (identity mapping, original behaviour); mapped rigs only expose their mapped joints,
    // so MetaHuman's corrective/facial joints are never touched by the retargeter.
    const canonical = new Map<string, string>();
    if (profile.bones) for (const [name, rig] of Object.entries(profile.bones)) canonical.set(rig, name);
    const rigBones = new Map<string, THREE.Bone>();
    root.traverse((o) => { if (o instanceof THREE.Bone) rigBones.set(o.name, o); });
    const position = (o: THREE.Object3D) => o.getWorldPosition(new THREE.Vector3());
    root.traverse((o) => {
      if (o instanceof THREE.Bone) {
        const name = profile.bones ? canonical.get(o.name) : o.name;
        if (!name) return;
        const world = o.getWorldQuaternion(new THREE.Quaternion());
        let direction: THREE.Vector3;
        if (profile.chain) {
          const next = profile.chain[name] ? rigBones.get(rigBoneName(profile, profile.chain[name])) : undefined;
          if (next) direction = position(next).sub(position(o)).normalize();
          else if (o.parent instanceof THREE.Bone) direction = position(o).sub(position(o.parent)).normalize();
          else direction = new THREE.Vector3(0, 1, 0).applyQuaternion(world);
          if (o.parent instanceof THREE.Bone) this.lengths.set(name, position(o).distanceTo(position(o.parent)));
        } else {
          const child = o.children.find((c) => c instanceof THREE.Bone);
          direction = child
            ? position(child).sub(position(o)).normalize()
            : new THREE.Vector3(0, 1, 0).applyQuaternion(world);
        }
        this.binds.set(name, {
          bone: o,
          local: o.quaternion.clone(),
          world,
          direction,
        });
      }
    });
    const required = [
      "Neck",
      "Head",
      ...["Left", "Right"].flatMap((side) =>
        [
          "Arm",
          "ForeArm",
          "Hand",
          ...fingers.flatMap((f) => [1, 2, 3].map((i) => `Hand${f}${i}`)),
        ].map((n) => side + n),
      ),
    ];
    if (required.some((name) => !this.binds.has(name)))
      throw new Error("Humanoid body/finger rig is incomplete");
    const neck = this.bone("Neck");
    if (neck) this.neckOrigin.copy(neck.getWorldPosition(new THREE.Vector3()));
    const l = this.bone("LeftArm"),
      r = this.bone("RightArm");
    if (l && r)
      this.shoulderWidth = l
        .getWorldPosition(new THREE.Vector3())
        .distanceTo(r.getWorldPosition(new THREE.Vector3()));
    for (const side of ["Left", "Right"]) {
      const positions = ["Hand", "HandIndex1", "HandMiddle1", "HandPinky1"].map(
        (n) => this.bone(side + n)?.getWorldPosition(new THREE.Vector3()),
      );
      if (positions.every(Boolean)) {
        const q = basis(
          ...(positions as [
            THREE.Vector3,
            THREE.Vector3,
            THREE.Vector3,
            THREE.Vector3,
          ]),
        );
        if (q) this.palms.set(side, q);
      }
      // Elbow hinge: the bind arm's bend plane (shoulder -> elbow -> wrist), or, for a straight bind arm, the axis
      // that flexes the forearm forward (+z, the avatar's facing direction).
      const [s, e, w] = ["Arm", "ForeArm", "Hand"].map((n) => this.bone(side + n)?.getWorldPosition(new THREE.Vector3()));
      const upperBone = this.bone(side + "Arm");
      if (s && e && w && upperBone) {
        const u = e.clone().sub(s).normalize(), f = w.clone().sub(e).normalize();
        const hinge = u.clone().cross(f);
        if (hinge.length() < Math.sin(THREE.MathUtils.degToRad(5))) hinge.copy(u).cross(new THREE.Vector3(0, 0, 1));
        if (hinge.lengthSq() > 1e-8)
          this.hinges.set(side, hinge.normalize().applyQuaternion(upperBone.getWorldQuaternion(new THREE.Quaternion()).invert()));
      }
    }
    // ---- static data, computed once (the same expressions the per-frame code used to evaluate every time) ----
    this.names = [...this.binds.keys()];
    this.bindList = this.names.map((n) => this.binds.get(n)!);
    this.quietBones = [...new Set(this.bindList.map((b) => b.bone))];
    this.world = new WorldCache(root, this.bindList.map((b) => b.bone));
    this.bestPose = this.bindList.map(() => new THREE.Quaternion());
    for (const side of SIDES) {
      const lower = side.toLowerCase();
      // hand capsules (fingers, palm), then the forearm: the order of the contact pairs is kept
      const bones: CapsuleBone[] = [], spec: CapsuleSpec[] = [];
      const segment = (an: string, bn: string, saName: string, sbName: string, radius: number) => {
        const a = this.bone(side + an), b = this.bone(side + bn);
        if (!a || !b) return;
        bones.push({ a, b, ia: this.world.index.get(a)!, ib: this.world.index.get(b)!, sa: `${lower}_${saName}`, sb: `${lower}_${sbName}` });
        spec.push({ radius, forearm: an === "ForeArm" });
      };
      for (const finger of fingers) for (const i of [1, 2, 3]) segment(`Hand${finger}${i}`, `Hand${finger}${i + 1}`, `${finger.toLowerCase()}_${i}`, `${finger.toLowerCase()}_${i + 1}`, .008);
      for (const f of ["Index", "Middle", "Pinky"]) segment("Hand", `Hand${f}1`, "wrist", `${f.toLowerCase()}_1`, .022);
      segment("ForeArm", "Hand", "elbow", "wrist", .028);
      this.capsuleBones[side] = bones; this.capsuleSpec[side] = spec;
      this.capsuleHands[side] = spec.filter((c) => !c.forearm).length;
      const palmBind = this.palms.get(side);
      if (palmBind) this.palmInverse.set(side, palmBind.clone().invert());
      const lowerBind = this.binds.get(side + "ForeArm");
      if (lowerBind) this.foreArmRest.set(side, lowerBind.direction.clone().applyQuaternion(lowerBind.world.clone().invert()));
      const joints: FingerJoint[] = [];
      if (palmBind) for (const finger of fingers) for (let i = 1; i <= 3; i++) {
        const bind = this.binds.get(`${side}Hand${finger}${i}`);
        if (!bind) continue;
        const f = finger.toLowerCase();
        const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(palmBind);
        const first = this.binds.get(`${side}Hand${finger}1`)!, second = this.binds.get(`${side}Hand${finger}2`)!;
        // The mesh's slight bind curl defines the anatomical closing direction.
        // Mirrored hands cannot share a hard-coded signed palm-normal rule.
        const curlWorld = first.direction.clone().cross(second.direction);
        if (curlWorld.lengthSq() < 1e-5) curlWorld.copy(bind.direction).cross(normal).multiplyScalar(side === "Left" ? -1 : 1);
        curlWorld.normalize();
        const previousBind = i > 1 ? this.binds.get(`${side}Hand${finger}${i - 1}`)! : null;
        joints.push({
          bind, finger, i, a: `${lower}_${f}_${i}`, b: `${lower}_${f}_${i + 1}`, previous: i > 1 ? `${lower}_${f}_${i - 1}` : null,
          curlWorld,
          flexAxis: curlWorld.clone().applyQuaternion(bind.world.clone().invert()),
          spreadAxis: normal.clone().applyQuaternion(bind.world.clone().invert()),
          // relative to the bind curl; a straight source finger may straighten the (slightly curled) bind finger
          straight: previousBind ? previousBind.direction.angleTo(bind.direction) : 0,
        });
      }
      this.fingerJoints[side] = joints;
    }
    this.torso = measureTorso(this.root, this.bone("Spine2") ?? this.bone("Spine1"));
  }

  /** How deep a world point is inside the torso (with `margin` of clearance), and the way out, or null. */
  private torsoDepth(p: THREE.Vector3, margin: number) {
    const t = this.torso!, q = p.clone().applyMatrix4(this.toRest);
    const slice = t.slices[Math.floor((q.y - t.y0) / t.step)];
    if (!slice) return null;
    const dx = q.x - slice.cx, dz = q.z - slice.cz, r = Math.hypot(dx, dz);
    const a = ((Math.atan2(dz, dx) + Math.PI) / (2 * Math.PI)) * TORSO_BINS, b = Math.floor(a) % TORSO_BINS, f = a - Math.floor(a);
    const outline = slice.r[b] * (1 - f) + slice.r[(b + 1) % TORSO_BINS] * f;
    const depth = outline + margin - r;
    if (depth <= 0) return null;
    // The way out is always toward the front (signers keep their arms in front of the body), never through the back:
    // the outward direction around the torso, tilted forward.
    const out = (r > 1e-4 ? new THREE.Vector3(dx / r, 0, dz / r) : new THREE.Vector3()).add(new THREE.Vector3(0, 0, 1)).normalize();
    const forward = new THREE.Vector3(0, 0, 1).applyQuaternion(this.fromRest);
    return { depth, dir: out.applyQuaternion(this.fromRest), forward };
  }

  /**
   * After an arm is placed: if the upper arm, elbow, forearm or hand is inside the torso (monocular depth can put a
   * forearm through the chest when the hand reaches the other side), move the elbow out (forward / sideways) or the
   * wrist out, and place the arm again. A pure function of the pose, so seeking and sliced preparation agree.
   */
  private keepArmOutOfTorso(side: string, target: THREE.Vector3, elbowHint: THREE.Vector3, preservePalm = false) {
    if (!this.torso) return;
    const chest = this.torso.chest, ci = this.world.index.get(chest);
    if (ci === undefined) chest.updateWorldMatrix(true, false); else this.world.ensure(ci);
    this.toRest.copy(this.torso.restChest).multiply(chest.matrixWorld.clone().invert());
    this.fromRest.setFromRotationMatrix(this.toRest).invert();
    const upper = this.bone(side + "Arm")!, lower = this.bone(side + "ForeArm")!, hand = this.bone(side + "Hand")!, mid = this.bone(side + "HandMiddle1");
    let last = Infinity;
    for (let pass = 0; pass < 6; pass++) {
      const s = this.wpos(upper, new THREE.Vector3()), e = this.wpos(lower, new THREE.Vector3()), w = this.wpos(hand, new THREE.Vector3());
      const m = mid ? this.wpos(mid, new THREE.Vector3()) : w.clone();
      const samples: [THREE.Vector3, number, boolean][] = [   // point, clearance (m), is it the hand
        [s.clone().lerp(e, .75), .03, false], [e, .035, false],
        ...[.25, .5, .75].map((k): [THREE.Vector3, number, boolean] => [e.clone().lerp(w, k), .03, false]),
        [w, .025, true], [w.clone().lerp(m, .6), .015, true]];
      let worst: { depth: number; dir: THREE.Vector3; forward: THREE.Vector3; isHand: boolean } | null = null;
      for (const [p, margin, isHand] of samples) {
        const hit = this.torsoDepth(p, margin);
        if (hit && (!worst || hit.depth > worst.depth)) worst = { ...hit, isHand };
      }
      if (!worst) return;
      if (pass === 0) { this.torsoFixed++; this.torsoMaxDepth = Math.max(this.torsoMaxDepth, worst.depth); }
      if (worst.isHand) target.addScaledVector(worst.dir, worst.depth + .005);
      else {
        elbowHint.addScaledVector(worst.dir, worst.depth * 2 + .03);
        // a nearly straight arm across the body cannot bend its way out: bring the hand forward too
        if (worst.depth > last * 0.7) target.addScaledVector(worst.forward, Math.min(.06, worst.depth + .01));
      }
      last = worst.depth;
      this.placeArm(side, target, elbowHint, preservePalm);
    }
  }

  /** placeArm, then the torso check (every arm placement of the derivation goes through here). */
  private placeArmOutside(side: string, target: THREE.Vector3, elbow: THREE.Vector3, preservePalm = false) {
    target = target.clone(); elbow = elbow.clone();
    this.placeArm(side, target, elbow, preservePalm);
    if (this.torsoGuard) this.keepArmOutOfTorso(side, target, elbow, preservePalm);
  }
  private bone(name: string) {
    return this.binds.get(name)?.bone;
  }
  // ---- world transforms through the cache (identical numbers to getWorldPosition / getWorldQuaternion) ----
  private wpos(o: THREE.Object3D, target: THREE.Vector3) {
    const i = this.world.index.get(o);
    if (i === undefined) return o.getWorldPosition(target);
    this.world.ensure(i);
    return target.setFromMatrixPosition(o.matrixWorld);
  }
  private wquat(o: THREE.Object3D, target: THREE.Quaternion) {
    const i = this.world.index.get(o);
    return i === undefined ? o.getWorldQuaternion(target) : this.world.quaternion(i, target);
  }
  /** A bone's local rotation changed (was: bone.updateWorldMatrix(false, true)). */
  private touch(o: THREE.Object3D) {
    const i = this.world.index.get(o);
    if (i === undefined) o.updateWorldMatrix(false, true);
    else this.world.invalidate(i);
  }
  /** Was: root.updateMatrixWorld(true) inside the derivation. */
  private refresh() {
    this.world.invalidateAll();
  }
  /** While deriving, the mapped bones' quaternions do not update their Euler `rotation` on every write (nobody
   * reads it); it is synchronised once at the end. */
  private quiet() {
    if (this.quietDepth++ > 0) return;
    this.quietSaved = this.quietBones.map((b) => (b.quaternion as unknown as { _onChangeCallback: () => void })._onChangeCallback);
    for (const b of this.quietBones) (b.quaternion as unknown as { _onChangeCallback: () => void })._onChangeCallback = noop;
  }
  private loud() {
    if (--this.quietDepth > 0) return;
    this.quietBones.forEach((b, i) => {
      (b.quaternion as unknown as { _onChangeCallback: () => void })._onChangeCallback = this.quietSaved[i];
      this.quietSaved[i]();
    });
  }
  private aim(name: string, direction: THREE.Vector3) {
    const bind = this.binds.get(name);
    if (bind) this.aimBind(bind, direction);
  }
  private aimBind(bind: Bind, direction: THREE.Vector3) {
    if (direction.lengthSq() < 1e-10) return;
    const desired = new THREE.Quaternion()
      .setFromUnitVectors(bind.direction, direction.clone().normalize())
      .multiply(bind.world);
    const parent = bind.bone.parent ? this.wquat(bind.bone.parent, new THREE.Quaternion()) : new THREE.Quaternion();
    bind.bone.quaternion.copy(parent.invert().multiply(desired));
    this.touch(bind.bone);
  }
  private placeArm(side: string, target: THREE.Vector3, elbow: THREE.Vector3, preservePalm = false) {
    const upper = this.bone(side + "Arm")!, lower = this.bone(side + "ForeArm")!, hand = this.bone(side + "Hand")!;
    const palm = this.wquat(hand, new THREE.Quaternion());
    const shoulder = this.wpos(upper, new THREE.Vector3());
    const l1 = this.lengths.get(side + "ForeArm") ?? lower.position.length(),
      l2 = this.lengths.get(side + "Hand") ?? hand.position.length();
    const axis = target.clone().sub(shoulder);
    const d = THREE.MathUtils.clamp(axis.length(), Math.abs(l1 - l2) + .005, l1 + l2 - .005);
    axis.normalize();
    const pole = elbow.clone().sub(shoulder);
    pole.addScaledVector(axis, -pole.dot(axis));
    if (pole.lengthSq() < .00001)
      pole.set(side === "Left" ? 1 : -1, -1, -.3).addScaledVector(axis, -pole.dot(axis));
    pole.normalize();
    const along = (l1 * l1 - l2 * l2 + d * d) / (2 * d), height = Math.sqrt(Math.max(0, l1 * l1 - along * along));
    const bend = shoulder.clone().addScaledVector(axis, along).addScaledVector(pole, height);
    const reachable = shoulder.clone().addScaledVector(axis, d);
    const upperDir = bend.clone().sub(shoulder).normalize();
    this.aim(side + "Arm", upperDir);
    const hinge = this.hinges.get(side), lowerBind = this.binds.get(side + "ForeArm");
    if (hinge && lowerBind) {
      // The elbow is a hinge (as in a UE5 IK rig): roll the upper arm about its own axis so that its elbow axis lies
      // in the bend plane, then turn the forearm from its bind pose under the upper arm about that axis only. The
      // forearm never twists or bends sideways at the elbow, so the hand's roll about the forearm is a true
      // pronation/supination angle. Nearly straight arms have no reliable bend plane: the roll fades out there.
      const carried = hinge.clone().applyQuaternion(this.wquat(upper, new THREE.Quaternion()));
      const wanted = pole.clone().cross(axis);
      carried.addScaledVector(upperDir, -carried.dot(upperDir));
      wanted.addScaledVector(upperDir, -wanted.dot(upperDir));
      if (carried.lengthSq() > 1e-8 && wanted.lengthSq() > 1e-8) {
        carried.normalize(); wanted.normalize();
        const roll = Math.atan2(upperDir.dot(carried.clone().cross(wanted)), carried.dot(wanted));
        const bent = THREE.MathUtils.smoothstep(Math.asin(THREE.MathUtils.clamp(height / l1, 0, 1)), 0.1, 0.45);
        // The humerus turns at most ~50 degrees from the aimed pose: beyond, MetaHuman's shoulder skinning (no
        // corrective joints driven) shows the body through the sleeve; the rest stays as a small elbow twist.
        const limit = THREE.MathUtils.degToRad(50);
        this.turn(side + "Arm", new THREE.Quaternion().setFromAxisAngle(upperDir, softClamp(roll * bent, -limit, limit, THREE.MathUtils.degToRad(15))));
      }
      lower.quaternion.copy(lowerBind.local);
      this.touch(lower);
      const lowerWorld = this.wquat(lower, new THREE.Quaternion());
      const current = this.foreArmRest.get(side)!.clone().applyQuaternion(lowerWorld);
      const target = reachable.sub(this.wpos(lower, new THREE.Vector3())).normalize();
      this.turn(side + "ForeArm", new THREE.Quaternion().setFromUnitVectors(current, target));
    } else this.aim(side + "ForeArm", reachable.sub(this.wpos(lower, new THREE.Vector3())));
    if (preservePalm) {
      hand.quaternion.copy(this.wquat(hand.parent!, new THREE.Quaternion()).invert().multiply(palm));
      this.touch(hand);
    }
  }
  /** Rotate a bone by a world-space rotation (keeps its children attached). */
  private turn(name: string, rotation: THREE.Quaternion) {
    const bone = this.bone(name);
    if (!bone) return;
    const world = this.wquat(bone, new THREE.Quaternion());
    const parent = bone.parent ? this.wquat(bone.parent, new THREE.Quaternion()) : new THREE.Quaternion();
    bone.quaternion.copy(parent.invert().multiply(rotation.clone().multiply(world)));
    this.touch(bone);
  }
  /** Body layer (engine motions): torso lean/tilt, shoulder shrug and head orientation follow the source's
   * deviation from the clip's median pose. Rotations are bounded; yaw is not observable from 2D shoulders. */
  private applyBody(joints: InterpolatedJoints, ref: BodyRef) {
    const { neck, left_shoulder: ls, right_shoulder: rs, head } = joints;
    if (!neck || !ls || !rs || !head) return;
    const clamp = THREE.MathUtils.clamp, torso = 0.5;
    const d = vec(neck).sub(ref.neck);
    // lateral sway -> roll about the forward (z) axis; forward/back -> pitch about x. The shoulder-line tilt is
    // NOT added here: the clavicles below reproduce it (the neck is the shoulders' midpoint, so the per-side
    // lift relative to it is exactly that tilt; counting it in the spine too would apply it twice).
    const roll = clamp(-Math.atan2(d.x, torso), -0.25, 0.25), pitch = clamp(Math.atan2(d.z, torso), -0.2, 0.2);
    // Follow-through (animation cheat): the chest turns a little toward hands that work off-centre, as a real
    // signer's torso does when reaching across; at most ~5 degrees.
    const lw = joints.left_wrist, rw = joints.right_wrist;
    const reach = lw && rw ? (lw[0] + rw[0]) / 2 - neck[0] : 0;
    const yaw = clamp(0.45 * reach, -0.09, 0.09);
    const spine = ["Spine", "Spine1", "Spine2"].filter((n) => this.binds.has(n));
    if (spine.length) {
      const part = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch / spine.length, yaw / spine.length, roll / spine.length, "XYZ"));
      for (const n of spine) this.turn(n, part);
    }
    // shoulder shrug: raise/lower the clavicle so the shoulder moves like the source's (relative to the neck)
    for (const [side, s, i] of [["Left", ls, 0], ["Right", rs, 1]] as const) {
      const clavicle = this.bone(side + "Shoulder"), arm = this.bone(side + "Arm");
      if (!clavicle || !arm) continue;
      // + automatic elevation: real clavicles rise when the hand goes above shoulder height (up to 3.5 cm)
      const wrist = side === "Left" ? lw : rw;
      const rise = wrist ? THREE.MathUtils.smoothstep(wrist[1] - s[1], -0.05, 0.25) * 0.035 : 0;
      const lift = clamp((s[1] - neck[1]) - ref.shrug[i] + rise, -0.04, 0.07);
      const c = this.wpos(clavicle, new THREE.Vector3()), a = this.wpos(arm, new THREE.Vector3());
      const from = a.clone().sub(c), to = from.clone().add(new THREE.Vector3(0, lift, 0));
      if (from.lengthSq() > 1e-6) this.turn(side + "Shoulder", new THREE.Quaternion().setFromUnitVectors(from.normalize(), to.normalize()));
    }
    // head: rotation taking the median neck->nose direction to the current one, plus the eye-line tilt;
    // the torso rotation already applied is removed so the head is not turned twice. 40 % neck, 60 % head.
    const now = vec(head).sub(vec(neck)), rest = ref.head.clone();
    if (now.lengthSq() < 1e-6 || rest.lengthSq() < 1e-6) return;
    const look = new THREE.Quaternion().setFromUnitVectors(rest.normalize(), now.normalize());
    const torsoRot = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, roll, "XYZ"));
    look.premultiply(torsoRot.invert());
    if (ref.eyeRoll !== null && joints.left_eye && joints.right_eye) {
      const eyeRoll = Math.atan2(joints.left_eye[1] - joints.right_eye[1], joints.left_eye[0] - joints.right_eye[0]) - ref.eyeRoll - roll;
      look.premultiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 0, 1), clamp(eyeRoll, -0.3, 0.3)));
    }
    look.rotateTowards(new THREE.Quaternion(), Math.max(0, 2 * Math.acos(Math.min(1, Math.abs(look.w))) - 0.45));   // at most ~26 degrees
    const identity = new THREE.Quaternion();
    this.turn("Neck", identity.clone().slerp(look, 0.4));
    this.turn("Head", identity.clone().slerp(look, 0.6));
  }
  /** Pose the rig from joint positions (no motion: the rest stance). All world matrices are up to date afterwards. */
  apply(joints: InterpolatedJoints) {
    this.quiet();
    try { this.applyPose(joints); } finally { this.loud(); }
    this.root.updateMatrixWorld(true);
  }
  /** The pose itself; only the cached world transforms are kept up to date. */
  private applyPose(joints: InterpolatedJoints) {
    for (const b of this.bindList) b.bone.quaternion.copy(b.local);
    this.refresh();
    if (this.bodyRef) this.applyBody(joints, this.bodyRef);
    // arm targets are placed relative to the avatar's CURRENT neck, so the arms travel with the body layer
    const neckNow = this.bodyRef && this.bone("Neck") ? this.wpos(this.bone("Neck")!, new THREE.Vector3()) : this.neckOrigin;
    const srcNeck = vec(joints.neck || [0, 1.5, 0]);
    const sourceWidth =
      joints.left_shoulder && joints.right_shoulder
        ? vec(joints.left_shoulder).distanceTo(vec(joints.right_shoulder))
        : 0.36;
    const scale = THREE.MathUtils.clamp(
      this.shoulderWidth / Math.max(sourceWidth, 0.2),
      0.75,
      1.3,
    );
    const map = (a: number[]) =>
      vec(a).sub(srcNeck).multiplyScalar(scale).add(neckNow);
    for (const side of SIDES) {
      const prefix = side.toLowerCase();
      const upper = this.bone(side + "Arm"),
        lower = this.bone(side + "ForeArm"),
        hand = this.bone(side + "Hand");
      const wrist = joints[prefix + "_wrist"],
        elbow = joints[prefix + "_elbow"];
      if (!upper || !lower || !hand || !wrist || !elbow) continue;
      const target = map(wrist);
      // Monocular depth can place a hand inside the torso. A local torso envelope
      // prevents mesh penetration without rewriting the captured motion asset.
      if (
        target.y < this.neckOrigin.y + 0.04 &&
        target.y > this.neckOrigin.y - 0.65 &&
        Math.abs(target.x) < 0.28
      ) {
        const front =
          this.neckOrigin.z +
          0.18 * Math.sqrt(Math.max(0, 1 - (target.x / 0.3) ** 2)) +
          0.045;
        target.z = Math.max(target.z, front);
      }
      // Only source-supported non-contact gets a head guard. Face-touching signs
      // and uncertain monocular contact must not be pushed away automatically.
      if (joints.head && vec(wrist).distanceTo(vec(joints.head)) > .22) {
        const head = this.wpos(this.bone("Head")!, new THREE.Vector3()).add(new THREE.Vector3(0, .08, .015));
        const delta = target.clone().sub(head), distance = delta.length();
        if (distance > .001 && distance < .13)
          target.addScaledVector(delta.normalize(), Math.min(.03, .13 - distance));
      }
      this.placeArmOutside(side, target, map(elbow));
      const points = ["wrist", "index_1", "middle_1", "pinky_1"].map(
        (n) => joints[prefix + "_" + n],
      );
      const trackedHand = points.every(Boolean);
      if (!trackedHand) {
        // Without a hand observation, continue the forearm instead of leaving
        // the authored bind hand (often a clasp/fist) at the wrist. Straighten
        // the four fingers so the fallback reads as a neutral hand, not a sign.
        const handPosition = this.wpos(hand, new THREE.Vector3());
        const forearm = handPosition.sub(this.wpos(lower, new THREE.Vector3()));
        this.aim(side + "Hand", forearm);
        for (const finger of fingers.slice(1)) for (let i = 1; i <= 3; i++)
          this.aim(`${side}Hand${finger}${i}`, forearm);
      }
      const palmBind = this.palms.get(side),
        handBind = this.binds.get(side + "Hand");
      if (trackedHand && palmBind && handBind) {
        const q = basis(
          ...(points.map((p) => map(p)) as [
            THREE.Vector3,
            THREE.Vector3,
            THREE.Vector3,
            THREE.Vector3,
          ]),
        );
        if (q) {
          const desired = q
            .multiply(this.palmInverse.get(side)!)
            .multiply(handBind.world);
          const parent = this.wquat(hand.parent!, new THREE.Quaternion())
            .invert();
          hand.quaternion.copy(parent.multiply(desired));
          this.touch(hand);
        }
      }
      // Calibrated local bend/spread axes: no arbitrary per-phalanx axial twist.
      if (trackedHand && palmBind) {
        const sourcePalm = basis(...(points.map((p) => vec(p)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3]));
        if (sourcePalm) {
          const sourceInverse = sourcePalm.clone().invert();
          for (const f of this.fingerJoints[side]) {
            const { bind, i } = f;
            const a = joints[f.a], b = joints[f.b];
            if (!a || !b) continue;
            const direction = vec(b).sub(vec(a));
            if (direction.lengthSq() < 1e-8) continue;
            direction.normalize().applyQuaternion(sourceInverse);
            let flex = 0, spread = 0, straight = 0;
            if (i > 1) {
              const previous = joints[f.previous!];
              if (!previous) continue;
              const before = vec(a).sub(vec(previous)).normalize().applyQuaternion(sourceInverse);
              straight = f.straight;
              flex = before.angleTo(direction) - straight;
            } else if (f.finger === "Thumb") {
              // Thumb CMC has a distinct opposition axis; retain bounded 3D swing.
              const observedBind = direction.clone().applyQuaternion(palmBind);
              const swing = new THREE.Quaternion().setFromUnitVectors(bind.direction, observedBind);
              const limit = new THREE.Quaternion().rotateTowards(swing, 1.2);
              const local = bind.world.clone().invert().multiply(limit).multiply(bind.world);
              bind.bone.quaternion.copy(bind.local).multiply(local);
              this.touch(bind.bone);
              continue;
            } else {
              const observedBind = direction.clone().applyQuaternion(palmBind);
              flex = Math.atan2(f.curlWorld.dot(bind.direction.clone().cross(observedBind)), bind.direction.dot(observedBind));
              spread = THREE.MathUtils.clamp(-Math.atan2(direction.x, direction.y), -.45, .45);
            }
            flex = THREE.MathUtils.clamp(flex, i === 1 ? -.2 : -straight, f.finger === "Thumb" ? 1.4 : i === 2 ? 1.9 : 1.65);
            bind.bone.quaternion.copy(bind.local)
              .multiply(new THREE.Quaternion().setFromAxisAngle(f.spreadAxis, spread))
              .multiply(new THREE.Quaternion().setFromAxisAngle(f.flexAxis, flex));
            this.touch(bind.bone);
          }
        }
      }
    }
  }

  /** Source points of the contact capsules at one instant (they depend only on the motion). */
  private capsuleSource(joints: InterpolatedJoints): CapsuleSource {
    const side = (s: Side) => {
      const bones = this.capsuleBones[s], n = bones.length;
      const p = new Float64Array(n * 6), hasA = new Uint8Array(n), observed = new Uint8Array(n);
      bones.forEach((c, k) => {
        const sa = joints[c.sa], sb = joints[c.sb];
        if (sa) { p[k * 6] = sa[0]; p[k * 6 + 1] = sa[1]; p[k * 6 + 2] = sa[2]; hasA[k] = 1; }
        if (sb) { p[k * 6 + 3] = sb[0]; p[k * 6 + 4] = sb[1]; p[k * 6 + 5] = sb[2]; }
        observed[k] = sa && sb ? 1 : 0;
      });
      return { p, hasA, observed };
    };
    const left = side("Left"), right = side("Right");
    return { left, right, gaps: new Float64Array(this.capsuleBones.Left.length * this.capsuleBones.Right.length).fill(NaN), rightCount: this.capsuleBones.Right.length };
  }
  /** The posed capsules of one side (with or without the forearm). `into`: a scratch set to overwrite (for sets
   * that are used at once), else a new one. */
  private capsules(side: Side, forearms = true, into?: CapsuleSet): CapsuleSet {
    const bones = this.capsuleBones[side], n = forearms ? bones.length : this.capsuleHands[side], world = this.world;
    const p = into ? into.p : new Float64Array(n * 6);
    for (let k = 0; k < n; k++) {
      const c = bones[k];
      world.ensure(c.ia);
      let e = c.a.matrixWorld.elements;
      p[k * 6] = e[12]; p[k * 6 + 1] = e[13]; p[k * 6 + 2] = e[14];
      world.ensure(c.ib);
      e = c.b.matrixWorld.elements;
      p[k * 6 + 3] = e[12]; p[k * 6 + 4] = e[13]; p[k * 6 + 5] = e[14];
    }
    if (into) { into.n = n; return into; }
    return { p, base: p, n };
  }
  /** Scratch capsule sets (full, with forearms) for contact checks whose capsules are not kept. */
  private scratchL?: CapsuleSet;
  private scratchR?: CapsuleSet;
  private scratch(side: Side) {
    const key = side === "Left" ? "scratchL" : "scratchR";
    let set = this[key];
    if (!set) { const p = new Float64Array(this.capsuleBones[side].length * 6); set = this[key] = { p, base: p, n: 0 }; }
    return this.capsules(side, true, set);
  }
  /** Contact of the current pose (full capsules), on scratch sets. */
  private poseContact(src: CapsuleSource, out: ContactResult) {
    return this.contact(this.scratch("Left"), this.scratch("Right"), src, out);
  }
  private contact(left: CapsuleSet, right: CapsuleSet, src: CapsuleSource, out: ContactResult) {
    return handContactSets(left, right, this.capsuleSpec.Left, this.capsuleSpec.Right, src, out);
  }

  /** Offline derivation of a motion, all at once (blocks for about 15-40 ms per second of motion). */
  private prepare(motion: MotionFile) {
    const steps = this.prepareSteps(motion);
    this.quiet();
    try {
      for (;;) {
        const r = steps.next();
        if (r.done) return r.value;
      }
    } finally { this.loud(); }
  }

  /**
   * The same derivation cut into small steps: it pauses (yield) only at the top of a frame's iteration, where the
   * iteration sets every pose it reads, so the rig may be posed and drawn between steps (sliced prewarm). After a
   * pause the world cache is dropped (other layers may have moved bones meanwhile).
   */
  private *prepareSteps(motion: MotionFile) {
    const start = performance.now(), names = this.names, binds = this.bindList;
    const ref = followsBody(motion) ? bodyReference(motion) : null;
    this.torsoFixed = 0; this.torsoMaxDepth = 0;
    const all: { time: number; rotations: THREE.Quaternion[] }[] = [];
    for (const frame of motion.frames) {
      yield; this.refresh();
      this.bodyRef = ref;
      this.applyPose(frame.joints);
      this.bodyRef = null;
      all.push({ time: frame.time_ms, rotations: binds.map((b) => b.bone.quaternion.clone()) });
    }
    let rejected = 0;
    const states: Record<string, HandState[]> = {};
    for (const side of SIDES) {
      const indexes = names.flatMap((n, i) => n.startsWith(side + "Hand") ? [i] : []);
      const rest = indexes.map((i) => binds[i].local.clone());
      // `all` already holds the forearm-aligned fallback for untracked frames.
      // Use it as the per-frame neutral target instead of the authored bind pose.
      const neutral = all.map((f) => indexes.map((j) => f.rotations[j].clone()));
      const track = stabilizeRotations(all.map((f, i) => ({ time: f.time, rotations: indexes.map((j) => f.rotations[j]), valid: hasHand(motion.frames[i], side.toLowerCase()) })), rest, neutral);
      track.frames.forEach((f, i) => indexes.forEach((j, k) => { all[i].rotations[j] = f.rotations[k]; }));
      states[side] = track.states.map((state, i) => state === "observed" && motion.frames[i].playback_origin && motion.frames[i].playback_origin !== "source" ? "interpolated" : state);
      rejected += track.rejected;
    }
    // Hand realism: anatomical limits and zero-phase finger smoothing on the keyframe track, BEFORE contacts are
    // resolved, so the contact pass sees the final hand shapes. Local rotations only; a pure function of the motion.
    yield; this.refresh();
    const handLimited = this.hands ? this.hands.constrainTrack(all, names) : 0;
    yield; this.refresh();
    // Resolve contacts AFTER rotation filtering. Densify to 100 Hz so a collision
    // between source keyframes is also checked; runtime only samples this cache.
    const dense: { time_ms: number; rotations: THREE.Quaternion[] }[] = [], denseStates: Record<string, HandState[]> = { Left: [], Right: [] };
    const geometry: { src: CapsuleSource; original: Record<Side, THREE.Vector3> }[] = [];
    const hands = { Left: this.bone("LeftHand")!, Right: this.bone("RightHand")! };
    const foreArms = { Left: this.bone("LeftForeArm")!, Right: this.bone("RightForeArm")! };
    const handPositions = () => ({ Left: this.wpos(hands.Left, new THREE.Vector3()), Right: this.wpos(hands.Right, new THREE.Vector3()) });
    const { hitA, hitB, bestPose } = this;
    let correctedFrames = 0, contactFrames = 0, maxCorrection = 0, maxPenetration = 0;
    for (let time = 0; all.length && time <= motion.duration_ms; time = Math.min(motion.duration_ms, time + 10)) {
      yield; this.refresh();
      const interval = frameInterval(motion.frames, time);
      const qa = all[interval.lo].rotations, qb = all[interval.next].rotations;
      for (let i = 0; i < binds.length; i++) binds[i].bone.quaternion.slerpQuaternions(qa[i], qb[i], interval.alpha);
      this.refresh();
      const joints = interpolateJointPositions(motion, time);
      const original = handPositions();
      const src = this.capsuleSource(joints);
      let corrected = false;
      const resolveContacts = (includeForearms: boolean) => {
      const activeCaps = (side: Side) => this.capsules(side, includeForearms);
      let bestScore = Infinity, hasBest = false;
      const remember = () => {
        if (!includeForearms) return;
        const score = this.poseContact(src, hitB).maxRawPenetration;
        if (score < bestScore) { bestScore = score; hasBest = true; for (let i = 0; i < binds.length; i++) bestPose[i].copy(binds[i].bone.quaternion); }
      };
      const restore = () => {
        if (!includeForearms || !hasBest) return;
        for (let i = 0; i < binds.length; i++) binds[i].bone.quaternion.copy(bestPose[i]);
        this.refresh();
      };
      remember();

      for (let iteration = 0; iteration < 8; iteration++) {
        const left = activeCaps("Left"), right = activeCaps("Right");
        if (!left.n || !right.n) break;
        const hit = this.contact(left, right, src, hitA);
        if (iteration === 0 && hit.contact) contactFrames++;
        if (hit.correction.lengthSq() < 1e-10) break;
        for (const side of SIDES) {
          const target = this.wpos(hands[side], new THREE.Vector3()).addScaledVector(hit.correction, side === "Left" ? 1 : -1);
          // Keep the volume of the palm/fingers in front of the torso, rather
          // than correcting one hand backwards through the character's chest.
          let bodyPush = 0;
          const set = side === "Left" ? left : right, spec = this.capsuleSpec[side], p = set.p;
          for (let k = 0; k < set.n; k++) for (let e = k * 6; e <= k * 6 + 3; e += 3) {
            const px = p[e], py = p[e + 1], pz = p[e + 2];
            if (py < this.neckOrigin.y - .04 && py > this.neckOrigin.y - .55 && Math.abs(px) < .25) {
              const front = this.neckOrigin.z + .14 * Math.sqrt(Math.max(0, 1 - (px / .28) ** 2));
              bodyPush = Math.max(bodyPush, front + spec[k].radius - pz);
            }
          }
          target.z += Math.min(.04, bodyPush);
          const offset = target.clone().sub(original[side]).clampLength(0, .06);
          this.placeArmOutside(side, original[side].clone().add(offset), this.wpos(foreArms[side], new THREE.Vector3()), true);
        }
        remember();
        corrected = true;
      }
      restore();
      // A closest-pair projection can get stuck between several fingers. Search
      // bounded rigid translations for a better separating direction, keeping
      // fingers/palm intact and never moving a hand backwards into the torso.
      const left = activeCaps("Left"), right = activeCaps("Right");
      let best = this.contact(left, right, src, hitB).maxPenetration;
      if (best > .006) {
        const current = handPositions();
        let chosen: Record<Side, THREE.Vector3> | null = null;
        const shifted = (cs: CapsuleSet, delta: THREE.Vector3): CapsuleSet => {
          const p = new Float64Array(cs.p.length);
          for (let k = 0; k < p.length; k += 3) { p[k] = cs.p[k] + delta.x; p[k + 1] = cs.p[k + 1] + delta.y; p[k + 2] = cs.p[k + 2] + delta.z; }
          return { p, base: cs.base, n: cs.n };
        };
        for (const axis of SHIFT_AXES) for (const amount of [.01, .02, .03]) {
          const shifts = {} as Record<Side, THREE.Vector3>;
          for (const s of SIDES) {
            const target = current[s].clone().addScaledVector(axis, amount * (s === "Left" ? 1 : -1));
            shifts[s] = target.sub(original[s]).clampLength(0, .06).add(original[s]).sub(current[s]);
          }
          const penetration = this.contact(shifted(left, shifts.Left), shifted(right, shifts.Right), src, hitB).maxPenetration;
          if (penetration < best - .0001) { best = penetration; chosen = shifts; }
        }
        if (chosen) for (const side of SIDES) {
          const target = current[side].clone().add(chosen[side]);
          this.placeArmOutside(side, target, this.wpos(foreArms[side], new THREE.Vector3()), true);
          remember();
        corrected = true;
        }
      }

      remember(); restore();
      };
      resolveContacts(false); // Preserve the established hand-to-hand solution.
      resolveContacts(true); // Accept only actual geometric improvements.
      for (const side of SIDES)
        maxCorrection = Math.max(maxCorrection, this.wpos(hands[side], new THREE.Vector3()).distanceTo(original[side]));
      maxPenetration = Math.max(maxPenetration, this.poseContact(src, hitB).maxPenetration);
      if (corrected) correctedFrames++;
      dense.push({ time_ms: time, rotations: binds.map((b) => b.bone.quaternion.clone()) });
      geometry.push({ src, original });
      for (const side of SIDES) denseStates[side].push(states[side][interval.alpha < .5 ? interval.lo : interval.next]);
      if (time === motion.duration_ms) break;
    }
    const setPose = (rotations: THREE.Quaternion[]) => {
      for (let i = 0; i < binds.length; i++) binds[i].bone.quaternion.copy(rotations[i]);
      this.refresh();
    };
    const baselineWrists: Record<Side, THREE.Vector3>[] = [];
    for (const frame of dense) { yield; setPose(frame.rotations); baselineWrists.push(handPositions()); }
    const acceptedWrists = [...baselineWrists];
    let trajectoryRefinedFrames = 0;
    // Offline, deterministic pass: require both adjacent wrist steps to remain
    // within the existing trajectory's lengths. The next frame's fallback pose
    // is always feasible, so an improvement cannot force a later teleport.
    for (let i = 1; i < dense.length - 1; i++) {
      yield;
      const frame = dense[i], { src, original } = geometry[i];
      setPose(frame.rotations);
      let best = this.poseContact(src, hitB).maxRawPenetration;
      if (best <= .006) continue;
      const seed = frame.rotations, seedWrists = baselineWrists[i];
      const elbows = { Left: this.wpos(foreArms.Left, new THREE.Vector3()), Right: this.wpos(foreArms.Right, new THREE.Vector3()) };
      let chosen: THREE.Quaternion[] | null = null;
      for (const side of SIDES) for (const axis of TRAJECTORY_AXES) for (const amount of [.01,.02,.04,.06]) {
        setPose(seed);
        const target = seedWrists[side].clone().addScaledVector(axis, amount);
        target.sub(original[side]).clampLength(0,.06).add(original[side]);
        if (target.z < seedWrists[side].z - .0001) continue;
        this.placeArmOutside(side,target,elbows[side],true);
        const actual = this.wpos(hands[side], new THREE.Vector3());
        const previousLimit = seedWrists[side].distanceTo(baselineWrists[i-1][side]);
        const nextLimit = seedWrists[side].distanceTo(baselineWrists[i+1][side]);
        if (actual.distanceTo(acceptedWrists[i-1][side]) > previousLimit + 1e-6 ||
            actual.distanceTo(baselineWrists[i+1][side]) > nextLimit + 1e-6) continue;
        const score = this.poseContact(src, hitB).maxRawPenetration;
        if (score < best - .0001) { best=score; chosen=binds.map(b => b.bone.quaternion.clone()); }
      }
      if (chosen) { frame.rotations=chosen; trajectoryRefinedFrames++; }
      setPose(frame.rotations); acceptedWrists[i]=handPositions();
    }
    // Contacts move the arms with the palm kept in place, which changes the wrist angle: limit the wrists again.
    yield; this.refresh();
    if (this.hands) this.hands.constrainTrack(dense.map((f) => ({ time: f.time_ms, rotations: f.rotations })), names, null, ["wrist"]);
    maxPenetration=0;
    for (let i = 0; i < dense.length; i++) {
      yield;
      setPose(dense[i].rotations);
      maxPenetration=Math.max(maxPenetration,this.poseContact(geometry[i].src,hitB).maxPenetration);
      for (const side of SIDES) maxCorrection=Math.max(maxCorrection,this.wpos(hands[side], new THREE.Vector3()).distanceTo(geometry[i].original[side]));
    }
    const diagnostics = { preparedMs: performance.now() - start, rejectedSpikes: rejected, correctedFrames, contactFrames, maxCorrection, maxPenetration, trajectoryRefinedFrames, handLimited,
      torsoFixedKeyframes: this.torsoFixed, torsoMaxDepth: this.torsoMaxDepth };
    return { all: dense, names, states: denseStates, diagnostics };
  }

  /** True when the motion's offline derivation is cached (applyMotion will not block). */
  isPrepared(motion: MotionFile) {
    return this.tracks.has(motion);
  }
  /** Derive and cache a motion ahead of playback (blocks for about 15-40 ms per second of motion). Leaves the
   * rig in an arbitrary pose: the caller re-applies the current motion afterwards. */
  prime(motion: MotionFile) {
    if (!this.tracks.has(motion)) this.tracks.set(motion, this.prepare(motion));
  }
  private slicing = new WeakMap<MotionFile, ReturnType<HumanoidRetargeter["prepareSteps"]>>();
  /** Derive a motion a few milliseconds at a time (`budgetMs` per call), so drawing never stalls. True once cached.
   * Like prime(), leaves the rig in an arbitrary pose: the caller poses it again before drawing. */
  primeSlice(motion: MotionFile, budgetMs: number) {
    if (this.tracks.has(motion)) return true;
    let steps = this.slicing.get(motion);
    if (!steps) this.slicing.set(motion, (steps = this.prepareSteps(motion)));
    const until = performance.now() + budgetMs;
    this.quiet();
    try {
      do {
        const r = steps.next();
        if (r.done) {
          this.tracks.set(motion, r.value);
          this.slicing.delete(motion);
          return true;
        }
      } while (performance.now() < until);
    } finally { this.loud(); }
    return false;
  }
  /** Cached offline derivation; seeking never depends on past render calls. */
  applyMotion(motion: MotionFile, time: number) {
    let track = this.tracks.get(motion);
    if (!track) { track = this.prepare(motion); this.tracks.set(motion, track); }
    this.diagnostics = track.diagnostics;
    if (!track.all.length) return { Left: "untracked", Right: "untracked" };
    const { lo, next, alpha } = frameInterval(track.all, time);
    const qa = track.all[lo].rotations, qb = track.all[next].rotations, binds = this.bindList;
    // slerp into a temporary, then one write (one Euler update per bone instead of two)
    for (let i = 0; i < binds.length; i++) binds[i].bone.quaternion.copy(slerpTmp.slerpQuaternions(qa[i], qb[i], alpha));
    this.root.updateMatrixWorld(true);
    return { Left: track.states.Left[alpha < .5 ? lo : next], Right: track.states.Right[alpha < .5 ? lo : next] };
  }
}
const slerpTmp = new THREE.Quaternion();
const SHIFT_AXES = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(-1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, -1, 0)];
const TRAJECTORY_AXES = [new THREE.Vector3(1,0,0),new THREE.Vector3(-1,0,0),
  new THREE.Vector3(0,1,0),new THREE.Vector3(0,-1,0),new THREE.Vector3(0,0,1)];
