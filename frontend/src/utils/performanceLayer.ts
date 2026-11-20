import * as THREE from "three";
import { findRigBone, type RigProfile } from "./rigProfiles";

/**
 * "Interpreter performance" layer: game-animation techniques applied AFTER retargeting, every frame, on top of the
 * pose the retargeter just wrote (it resets every bone first, so nothing accumulates). Presentation only: it never
 * changes hand positions by more than a few millimetres, handshapes or the grammatical face.
 *
 * - Arm twist distribution: the hand's roll about the forearm is shared by the forearm twist bones
 *   (MetaHuman lowerarm_twist_01 = 2/3, _02 = 1/3), which removes the "candy-wrapper" wrist.
 * - Springs: slightly underdamped springs on bone rotations (inertia, tiny natural overshoot, residual jitter
 *   removed). They are driven by the MOTION clock: paused or seeking = no simulation (pose = target), so a
 *   frame-stepped export is deterministic.
 * - Moving holds: during held signs, a very small smooth drift of forearms and hands (people are never still).
 * - Hair spring: a ponytail chain (hair_ponytail_01..) swings as a chain of world-space springs, softer towards the
 *   tip; without a chain, the hair cards get one spring bone under the head (pivot near the crown, 6 degrees max).
 */

type ChainLink = { bone: THREE.Object3D; bind: THREE.Quaternion; q: THREE.Quaternion; w: THREE.Vector3; freq: number; zeta: number; maxAngle: number };
type SpringBone = { bone: THREE.Object3D; q: THREE.Quaternion; w: THREE.Vector3; freq: number; zeta: number; init: boolean };
type Twist = { hand: THREE.Object3D; lower: THREE.Object3D; parts: { bone: THREE.Object3D; weight: number; local: THREE.Quaternion }[]; handBind: THREE.Quaternion; axis: THREE.Vector3 };

const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const tmpQ = new THREE.Quaternion(), tmpQ2 = new THREE.Quaternion(), tmpV = new THREE.Vector3(), tmpV2 = new THREE.Vector3();

/** Smooth, cheap 1-D value noise in [-1, 1] (sum of incommensurate sines; deterministic in t). */
export const smoothNoise = (t: number, seed: number) =>
  0.55 * Math.sin(t * 1.31 + seed * 12.9) + 0.3 * Math.sin(t * 2.17 + seed * 4.1) + 0.15 * Math.sin(t * 3.73 + seed * 7.7);

/** Angular velocity integration: q <- exp(w dt) q. */
function integrate(q: THREE.Quaternion, w: THREE.Vector3, dt: number) {
  const angle = w.length() * dt;
  if (angle < 1e-9) return;
  tmpQ.setFromAxisAngle(tmpV.copy(w).normalize(), angle);
  q.premultiply(tmpQ).normalize();
}

/** Rotation vector (axis * angle) of the shortest rotation from `from` to `to`. */
function delta(from: THREE.Quaternion, to: THREE.Quaternion, out: THREE.Vector3) {
  tmpQ2.copy(to).multiply(tmpQ.copy(from).invert());
  if (tmpQ2.w < 0) tmpQ2.set(-tmpQ2.x, -tmpQ2.y, -tmpQ2.z, -tmpQ2.w);
  const s = Math.sqrt(Math.max(0, 1 - tmpQ2.w * tmpQ2.w));
  const angle = 2 * Math.atan2(s, tmpQ2.w);
  return s < 1e-8 ? out.set(0, 0, 0) : out.set(tmpQ2.x / s, tmpQ2.y / s, tmpQ2.z / s).multiplyScalar(angle);
}

export class PerformanceLayer {
  private springs: SpringBone[] = [];
  private twists: Twist[] = [];
  private holdBones: { bone: THREE.Object3D; seed: number; scale: number }[] = [];
  private wrists: THREE.Object3D[] = [];
  private lastWrist: THREE.Vector3[] = [];
  private stillness = 0;
  private hair?: { bone: THREE.Bone; head: THREE.Object3D; pivot: THREE.Vector3; q: THREE.Quaternion; w: THREE.Vector3; prevHead: THREE.Quaternion; init: boolean };
  private chain: ChainLink[] = [];
  private chainInit = false;
  private lastTime = -1;
  /** false disables the whole layer (tests, before/after comparisons). */
  enabled = true;

  constructor(private root: THREE.Object3D, profile: RigProfile) {
    const find = (name: string) => findRigBone(root, profile, name);
    const add = (name: string, freq: number, zeta: number) => {
      const bone = find(name);
      if (bone) this.springs.push({ bone, q: new THREE.Quaternion(), w: new THREE.Vector3(), freq, zeta, init: false });
    };
    for (const n of ["Spine", "Spine1", "Spine2"]) add(n, 3, 0.9);
    add("Neck", 3.5, 0.85);
    add("Head", 4, 0.8);
    for (const side of ["Left", "Right"]) {
      add(side + "Shoulder", 4, 0.85);
      add(side + "Arm", 7, 0.72);
      add(side + "ForeArm", 7.5, 0.72);
      add(side + "Hand", 9, 0.75);
      for (const f of FINGERS) for (let i = 1; i <= 3; i++) add(`${side}Hand${f}${i}`, 13, 0.85);
      const lower = find(side + "ForeArm"), hand = find(side + "Hand");
      if (lower && hand) {
        this.wrists.push(hand);
        this.lastWrist.push(new THREE.Vector3());
        this.holdBones.push({ bone: lower, seed: side === "Left" ? 1 : 2, scale: 0.008 }, { bone: hand, seed: side === "Left" ? 3 : 4, scale: 0.014 });
        // MetaHuman forearm twist bones (children of the forearm); other rigs simply have none.
        const s = side === "Left" ? "l" : "r";
        const parts = [["lowerarm_twist_01_" + s, 2 / 3], ["lowerarm_twist_02_" + s, 1 / 3]]
          .map(([n, weight]) => ({ bone: root.getObjectByName(n as string), weight: weight as number }))
          .filter((p): p is { bone: THREE.Object3D; weight: number } => Boolean(p.bone))
          .map((p) => ({ ...p, local: p.bone.quaternion.clone() }));
        if (parts.length) {
          // forearm axis in the forearm's local frame = direction to the hand joint
          const axis = hand.position.clone().normalize();
          this.twists.push({ hand, lower, parts, handBind: hand.quaternion.clone(), axis });
        }
      }
    }
    // Hair: hair cards skinned to the head joint -> give them their own spring bone at the same place.
    const head = find("Head");
    for (let i = 1; root.getObjectByName(`hair_ponytail_${String(i).padStart(2, "0")}`); i++) {
      const bone = root.getObjectByName(`hair_ponytail_${String(i).padStart(2, "0")}`)!;
      this.chain.push({ bone, bind: bone.quaternion.clone(), q: new THREE.Quaternion(), w: new THREE.Vector3(),
        freq: 3.4 - 0.35 * (i - 1), zeta: 0.32, maxAngle: THREE.MathUtils.degToRad(10 + 3 * (i - 1)) });
    }
    if (head && !this.chain.length) {
      const hairMeshes: THREE.SkinnedMesh[] = [];
      root.traverse((o) => { if (o instanceof THREE.SkinnedMesh && /hair/i.test(o.name + " " + (Array.isArray(o.material) ? "" : o.material.name))) hairMeshes.push(o); });
      if (hairMeshes.length) {
        const bone = new THREE.Bone();
        bone.name = "signaculture_hair_spring";
        head.add(bone);
        bone.updateMatrixWorld(true);
        for (const mesh of hairMeshes) {
          if (!mesh.skeleton.bones.includes(head as THREE.Bone)) continue;
          // Own skeleton copy (the loader may share one Skeleton between hair, face and body), with the head
          // joint replaced by the hair bone: same world transform at bind, so the inverse bind matrices still hold.
          const skeleton = new THREE.Skeleton(mesh.skeleton.bones.map((b) => (b === head ? bone : b)), mesh.skeleton.boneInverses.map((m) => m.clone()));
          mesh.bind(skeleton, mesh.bindMatrix);
        }
        // pivot ~11 cm above the head joint (crown), in head-local coordinates
        const pivot = head.worldToLocal(head.getWorldPosition(new THREE.Vector3()).add(new THREE.Vector3(0, 0.11, 0)));
        this.hair = { bone, head, pivot, q: new THREE.Quaternion(), w: new THREE.Vector3(), prevHead: new THREE.Quaternion(), init: false };
      }
    }
  }

  /** Share the hand's roll about the forearm axis with the forearm twist bones (swing-twist decomposition). */
  private distributeTwist() {
    for (const t of this.twists) {
      const rel = tmpQ.copy(t.handBind).invert().premultiply(tmpQ2.copy(t.hand.quaternion)); // hand rotation relative to bind, in forearm frame
      const p = tmpV.set(rel.x, rel.y, rel.z);
      const proj = t.axis.clone().multiplyScalar(p.dot(t.axis));
      const twist = new THREE.Quaternion(proj.x, proj.y, proj.z, rel.w);
      if (twist.lengthSq() < 1e-12) continue;
      twist.normalize();
      if (twist.w < 0) twist.set(-twist.x, -twist.y, -twist.z, -twist.w);
      const angle = 2 * Math.acos(THREE.MathUtils.clamp(twist.w, -1, 1));
      const sign = Math.sign(tmpV2.set(twist.x, twist.y, twist.z).dot(t.axis)) || 1;
      for (const part of t.parts) {
        part.bone.quaternion.copy(part.local).premultiply(tmpQ.setFromAxisAngle(t.axis, sign * angle * part.weight));
        part.bone.updateMatrixWorld(true);
      }
    }
  }

  /**
   * motionTimeMs: the motion clock (undefined when no motion plays). Springs run only when that clock advances
   * smoothly (0 < dt <= 0.25 s); otherwise the pose is taken as is and the springs are reset to it.
   */
  update(motionTimeMs: number | undefined, signing: boolean) {
    const dt = motionTimeMs === undefined || this.lastTime < 0 ? 0 : (motionTimeMs - this.lastTime) / 1000;
    this.lastTime = motionTimeMs ?? -1;
    const simulate = this.enabled && dt > 0 && dt <= 0.25;
    if (!this.enabled) return;
    // Moving holds: when the wrists barely move, add a tiny smooth drift (additive, below a degree).
    if (signing && motionTimeMs !== undefined) {
      let speed = 0;
      this.wrists.forEach((w, i) => {
        const p = w.getWorldPosition(tmpV);
        if (dt > 0) speed = Math.max(speed, p.distanceTo(this.lastWrist[i]) / dt);
        this.lastWrist[i].copy(p);
      });
      const target = speed < 0.12 ? 1 : 0;
      this.stillness += (target - this.stillness) * Math.min(1, (dt || 0) * 4);
      const t = motionTimeMs / 1000;
      if (this.stillness > 0.01)
        for (const h of this.holdBones) {
          const a = h.scale * this.stillness;
          h.bone.quaternion.multiply(tmpQ.setFromEuler(new THREE.Euler(a * smoothNoise(t, h.seed), a * smoothNoise(t, h.seed + 10), a * 0.6 * smoothNoise(t, h.seed + 20))));
        }
    } else this.stillness = 0;
    // Springs on local rotations (parents first: they are listed spine -> arms -> fingers).
    for (const s of this.springs) {
      if (!simulate || !s.init) {
        s.q.copy(s.bone.quaternion);
        s.w.set(0, 0, 0);
        s.init = true;
        continue;
      }
      const omega = 2 * Math.PI * s.freq;
      const steps = Math.max(1, Math.ceil(dt / (1 / 240)));       // sub-steps keep stiff springs stable
      const h = dt / steps;
      for (let k = 0; k < steps; k++) {
        delta(s.q, s.bone.quaternion, tmpV2);
        s.w.addScaledVector(tmpV2, omega * omega * h).multiplyScalar(Math.max(0, 1 - 2 * s.zeta * omega * h));
        integrate(s.q, s.w, h);
      }
      s.bone.quaternion.copy(s.q);
    }
    this.distributeTwist();
    // Ponytail chain: each link chases its rest orientation under the (already moved) parent, in world space.
    if (this.chain.length) {
      this.root.updateMatrixWorld(true);
      const reset = !simulate || !this.chainInit;
      this.chainInit = true;
      for (const link of this.chain) {
        const parentWorld = link.bone.parent!.getWorldQuaternion(new THREE.Quaternion());
        const target = parentWorld.clone().multiply(link.bind);
        if (reset) {
          link.q.copy(target);
          link.w.set(0, 0, 0);
        } else {
          const omega = 2 * Math.PI * link.freq;
          const steps = Math.max(1, Math.ceil(dt / (1 / 240))), h = dt / steps;
          for (let k = 0; k < steps; k++) {
            delta(link.q, target, tmpV2);
            link.w.addScaledVector(tmpV2, omega * omega * h).multiplyScalar(Math.max(0, 1 - 2 * link.zeta * omega * h));
            integrate(link.q, link.w, h);
          }
        }
        const local = parentWorld.invert().multiply(link.q);
        const off = link.bind.angleTo(local);
        if (off > link.maxAngle) local.copy(link.bind).rotateTowards(local, link.maxAngle);
        link.bone.quaternion.copy(local);
        link.bone.updateMatrixWorld(true);
      }
    }
    // Hair: lags the head's world rotation; local offset = displayed hair rotation relative to the head.
    if (this.hair) {
      const hair = this.hair;
      this.root.updateMatrixWorld(true);
      const headWorld = hair.head.getWorldQuaternion(new THREE.Quaternion());
      if (!simulate || !hair.init) {
        hair.q.copy(headWorld);
        hair.w.set(0, 0, 0);
        hair.init = true;
      } else {
        const omega = 2 * Math.PI * 2.2, zeta = 0.35;           // soft, slightly bouncy
        const steps = Math.max(1, Math.ceil(dt / (1 / 240))), h = dt / steps;
        for (let k = 0; k < steps; k++) {
          delta(hair.q, headWorld, tmpV2);
          hair.w.addScaledVector(tmpV2, omega * omega * h).multiplyScalar(Math.max(0, 1 - 2 * zeta * omega * h));
          integrate(hair.q, hair.w, h);
        }
      }
      // local = head^-1 * displayed, limited to 6 degrees, rotated about the crown pivot
      const local = headWorld.clone().invert().multiply(hair.q);
      local.rotateTowards(new THREE.Quaternion(), Math.max(0, 2 * Math.acos(Math.min(1, Math.abs(local.w))) - 0.105));
      hair.bone.quaternion.copy(local);
      hair.bone.position.copy(hair.pivot).sub(hair.pivot.clone().applyQuaternion(local));
      hair.bone.updateMatrixWorld(true);
    }
  }
}
