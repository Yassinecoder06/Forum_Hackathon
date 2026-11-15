import * as THREE from "three";
import { interpolateJointPositions, type InterpolatedJoints } from "./motionInterpolator";
import type { MotionFile } from "../types/contract";
import { frameInterval, hasHand, stabilizeRotations, type HandState } from "./avatarPlayback";
import { handContact, type Capsule } from "./handContact";
const fingers = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
type Bind = {
  bone: THREE.Bone;
  local: THREE.Quaternion;
  world: THREE.Quaternion;
  direction: THREE.Vector3;
};
const vec = (a: number[]) => new THREE.Vector3(a[0], a[1], a[2]);
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
/** Arm IK and calibrated local finger articulation: bind translations never stretch.
 * Every call starts from bind transforms, so seeking is deterministic. */
export class HumanoidRetargeter {
  private binds = new Map<string, Bind>();
  private palms = new Map<string, THREE.Quaternion>();
  private neckOrigin = new THREE.Vector3();
  private shoulderWidth = 0.36;
  private tracks = new WeakMap<MotionFile, ReturnType<HumanoidRetargeter["prepare"]>>();
  diagnostics = { preparedMs: 0, rejectedSpikes: 0, correctedFrames: 0, contactFrames: 0, maxCorrection: 0, maxPenetration: 0 };
  constructor(private root: THREE.Object3D) {
    root.updateMatrixWorld(true);
    root.traverse((o) => {
      if (o instanceof THREE.Bone) {
        const child = o.children.find((c) => c instanceof THREE.Bone);
        const world = o.getWorldQuaternion(new THREE.Quaternion());
        const direction = child
          ? child
              .getWorldPosition(new THREE.Vector3())
              .sub(o.getWorldPosition(new THREE.Vector3()))
              .normalize()
          : new THREE.Vector3(0, 1, 0).applyQuaternion(world);
        this.binds.set(o.name, {
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
    }
  }
  private bone(name: string) {
    return this.binds.get(name)?.bone;
  }
  private aim(name: string, direction: THREE.Vector3) {
    const bind = this.binds.get(name);
    if (!bind || direction.lengthSq() < 1e-10) return;
    const desired = new THREE.Quaternion()
      .setFromUnitVectors(bind.direction, direction.clone().normalize())
      .multiply(bind.world);
    const parent =
      bind.bone.parent?.getWorldQuaternion(new THREE.Quaternion()) ||
      new THREE.Quaternion();
    bind.bone.quaternion.copy(parent.invert().multiply(desired));
    bind.bone.updateWorldMatrix(false, true);
  }
  private placeArm(side: string, target: THREE.Vector3, elbow: THREE.Vector3, preservePalm = false) {
    const upper = this.bone(side + "Arm")!, lower = this.bone(side + "ForeArm")!, hand = this.bone(side + "Hand")!;
    const palm = hand.getWorldQuaternion(new THREE.Quaternion());
    const shoulder = upper.getWorldPosition(new THREE.Vector3());
    const l1 = lower.position.length(), l2 = hand.position.length();
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
    this.aim(side + "Arm", bend.sub(shoulder));
    this.aim(side + "ForeArm", reachable.sub(lower.getWorldPosition(new THREE.Vector3())));
    if (preservePalm) {
      hand.quaternion.copy(hand.parent!.getWorldQuaternion(new THREE.Quaternion()).invert().multiply(palm));
      hand.updateWorldMatrix(false, true);
    }
  }
  apply(joints: InterpolatedJoints) {
    for (const b of this.binds.values()) b.bone.quaternion.copy(b.local);
    this.root.updateMatrixWorld(true);
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
      vec(a).sub(srcNeck).multiplyScalar(scale).add(this.neckOrigin);
    for (const side of ["Left", "Right"]) {
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
        const head = this.bone("Head")!.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, .08, .015));
        const delta = target.clone().sub(head), distance = delta.length();
        if (distance > .001 && distance < .13)
          target.addScaledVector(delta.normalize(), Math.min(.03, .13 - distance));
      }
      this.placeArm(side, target, map(elbow));
      const points = ["wrist", "index_1", "middle_1", "pinky_1"].map(
        (n) => joints[prefix + "_" + n],
      );
      const palmBind = this.palms.get(side),
        handBind = this.binds.get(side + "Hand");
      if (points.every(Boolean) && palmBind && handBind) {
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
            .multiply(palmBind.clone().invert())
            .multiply(handBind.world);
          const parent = hand
            .parent!.getWorldQuaternion(new THREE.Quaternion())
            .invert();
          hand.quaternion.copy(parent.multiply(desired));
          hand.updateWorldMatrix(false, true);
        }
      }
      // Calibrated local bend/spread axes: no arbitrary per-phalanx axial twist.
      if (points.every(Boolean) && palmBind) {
        const sourcePalm = basis(...(points.map((p) => vec(p)) as [THREE.Vector3, THREE.Vector3, THREE.Vector3, THREE.Vector3]));
        if (sourcePalm) for (const finger of fingers) for (let i = 1; i <= 3; i++) {
          const bind = this.binds.get(`${side}Hand${finger}${i}`);
          const a = joints[`${prefix}_${finger.toLowerCase()}_${i}`], b = joints[`${prefix}_${finger.toLowerCase()}_${i + 1}`];
          if (!bind || !a || !b) continue;
          const direction = vec(b).sub(vec(a));
          if (direction.lengthSq() < 1e-8) continue;
          direction.normalize().applyQuaternion(sourcePalm.clone().invert());
          const normal = new THREE.Vector3(0, 0, 1).applyQuaternion(palmBind);
          const first = this.binds.get(`${side}Hand${finger}1`)!, second = this.binds.get(`${side}Hand${finger}2`)!;
          // The mesh's slight bind curl defines the anatomical closing direction.
          // Mirrored hands cannot share a hard-coded signed palm-normal rule.
          const curlWorld = first.direction.clone().cross(second.direction);
          if (curlWorld.lengthSq() < 1e-5) curlWorld.copy(bind.direction).cross(normal).multiplyScalar(side === "Left" ? -1 : 1);
          curlWorld.normalize();
          const flexAxis = curlWorld.clone().applyQuaternion(bind.world.clone().invert());
          const spreadAxis = normal.clone().applyQuaternion(bind.world.clone().invert());
          const observedBind = direction.clone().applyQuaternion(palmBind);
          let flex = Math.atan2(curlWorld.dot(bind.direction.clone().cross(observedBind)), bind.direction.dot(observedBind)), spread = 0;
          if (i > 1) {
            const previous = joints[`${prefix}_${finger.toLowerCase()}_${i - 1}`];
            if (!previous) continue;
            const before = vec(a).sub(vec(previous)).normalize().applyQuaternion(sourcePalm.clone().invert());
            const previousBind = this.binds.get(`${side}Hand${finger}${i - 1}`)!;
            flex = before.angleTo(direction) - previousBind.direction.angleTo(bind.direction);
          } else if (finger === "Thumb") {
            // Thumb CMC has a distinct opposition axis; retain bounded 3D swing.
            const observedBind = direction.clone().applyQuaternion(palmBind);
            const swing = new THREE.Quaternion().setFromUnitVectors(bind.direction, observedBind);
            const limit = new THREE.Quaternion().rotateTowards(swing, 1.2);
            const local = bind.world.clone().invert().multiply(limit).multiply(bind.world);
            bind.bone.quaternion.copy(bind.local).multiply(local);
            bind.bone.updateWorldMatrix(false, true);
            continue;
          } else spread = THREE.MathUtils.clamp(-Math.atan2(direction.x, direction.y), -.45, .45);
          flex = THREE.MathUtils.clamp(flex, i === 1 ? -.2 : 0, finger === "Thumb" ? 1.4 : i === 2 ? 1.9 : 1.65);
          bind.bone.quaternion.copy(bind.local)
            .multiply(new THREE.Quaternion().setFromAxisAngle(spreadAxis, spread))
            .multiply(new THREE.Quaternion().setFromAxisAngle(flexAxis, flex));
          bind.bone.updateWorldMatrix(false, true);
        }
      }
    }
  }

  private prepare(motion: MotionFile) {
    const start = performance.now(), names = [...this.binds.keys()];
    const all = motion.frames.map((frame) => {
      this.apply(frame.joints);
      return { time: frame.time_ms, rotations: names.map((n) => this.binds.get(n)!.bone.quaternion.clone()) };
    });
    let rejected = 0;
    const states: Record<string, HandState[]> = {};
    for (const side of ["Left", "Right"]) {
      const indexes = names.flatMap((n, i) => n.startsWith(side + "Hand") ? [i] : []);
      const rest = indexes.map((i) => this.binds.get(names[i])!.local.clone());
      const track = stabilizeRotations(all.map((f, i) => ({ time: f.time, rotations: indexes.map((j) => f.rotations[j]), valid: hasHand(motion.frames[i], side.toLowerCase()) })), rest);
      track.frames.forEach((f, i) => indexes.forEach((j, k) => { all[i].rotations[j] = f.rotations[k]; }));
      states[side] = track.states.map((state, i) => state === "observed" && motion.frames[i].playback_origin && motion.frames[i].playback_origin !== "source" ? "interpolated" : state);
      rejected += track.rejected;
    }
    // Resolve contacts AFTER rotation filtering. Densify to 50 Hz so a collision
    // between source keyframes is also checked; runtime only samples this cache.
    const dense = [], denseStates: Record<string, HandState[]> = { Left: [], Right: [] };
    let correctedFrames = 0, contactFrames = 0, maxCorrection = 0, maxPenetration = 0;
    for (let time = 0; all.length && time <= motion.duration_ms; time = Math.min(motion.duration_ms, time + 20)) {
      const interval = frameInterval(motion.frames, time);
      names.forEach((n, i) => this.binds.get(n)!.bone.quaternion.slerpQuaternions(all[interval.lo].rotations[i], all[interval.next].rotations[i], interval.alpha));
      this.root.updateMatrixWorld(true);
      const joints = interpolateJointPositions(motion, time);
      const original = Object.fromEntries(["Left", "Right"].map((side) => [side, this.bone(side + "Hand")!.getWorldPosition(new THREE.Vector3())]));
      const capsules = (side: string): Capsule[] => fingers.flatMap((finger) => [1, 2, 3].flatMap((i) => {
        const a = this.bone(`${side}Hand${finger}${i}`), b = this.bone(`${side}Hand${finger}${i + 1}`);
        const sa = joints[`${side.toLowerCase()}_${finger.toLowerCase()}_${i}`], sb = joints[`${side.toLowerCase()}_${finger.toLowerCase()}_${i + 1}`];
        return a && b && sa && sb ? [{ a: a.getWorldPosition(new THREE.Vector3()), b: b.getWorldPosition(new THREE.Vector3()), sourceA: vec(sa), sourceB: vec(sb) }] : [];
      }));
      let corrected = false;
      for (let iteration = 0; iteration < 3; iteration++) {
        const left = capsules("Left"), right = capsules("Right");
        if (left.length !== 15 || right.length !== 15) break;
        const hit = handContact(left, right);
        if (iteration === 0 && hit.contact) contactFrames++;
        if (hit.contact || hit.correction.lengthSq() < 1e-10) break;
        for (const side of ["Left", "Right"]) {
          const hand = this.bone(side + "Hand")!, elbow = this.bone(side + "ForeArm")!;
          const target = hand.getWorldPosition(new THREE.Vector3()).addScaledVector(hit.correction, side === "Left" ? 1 : -1);
          const offset = target.clone().sub(original[side]).clampLength(0, .02);
          maxCorrection = Math.max(maxCorrection, offset.length());
          this.placeArm(side, original[side].clone().add(offset), elbow.getWorldPosition(new THREE.Vector3()), true);
        }
        corrected = true;
      }
      const left = capsules("Left"), right = capsules("Right");
      maxPenetration = Math.max(maxPenetration, handContact(left, right).maxPenetration);
      if (corrected) correctedFrames++;
      dense.push({ time_ms: time, rotations: names.map((n) => this.binds.get(n)!.bone.quaternion.clone()) });
      for (const side of ["Left", "Right"]) denseStates[side].push(states[side][interval.alpha < .5 ? interval.lo : interval.next]);
      if (time === motion.duration_ms) break;
    }
    const diagnostics = { preparedMs: performance.now() - start, rejectedSpikes: rejected, correctedFrames, contactFrames, maxCorrection, maxPenetration };
    return { all: dense, names, states: denseStates, diagnostics };
  }

  /** Cached offline derivation; seeking never depends on past render calls. */
  applyMotion(motion: MotionFile, time: number) {
    let track = this.tracks.get(motion);
    if (!track) { track = this.prepare(motion); this.tracks.set(motion, track); }
    this.diagnostics = track.diagnostics;
    if (!track.all.length) return { Left: "untracked", Right: "untracked" };
    const { lo, next, alpha } = frameInterval(track.all, time);
    track.names.forEach((name, i) => this.binds.get(name)!.bone.quaternion.slerpQuaternions(track.all[lo].rotations[i], track.all[next].rotations[i], alpha));
    this.root.updateMatrixWorld(true);
    return Object.fromEntries(["Left", "Right"].map((s) => [s, track!.states[s][alpha < .5 ? lo : next]]));
  }
}
