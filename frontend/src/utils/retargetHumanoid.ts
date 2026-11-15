import * as THREE from "three";
import type { InterpolatedJoints } from "./motionInterpolator";
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
  const x = index.clone().sub(pinky).normalize(),
    y = middle.clone().sub(wrist).normalize();
  const z = new THREE.Vector3().crossVectors(x, y).normalize();
  if (x.lengthSq() < 0.5 || z.lengthSq() < 0.5) return null;
  y.crossVectors(z, x).normalize();
  return new THREE.Quaternion().setFromRotationMatrix(
    new THREE.Matrix4().makeBasis(x, y, z),
  );
}
/** Direction/IK retargeting only: bind translations and mesh bone lengths never stretch.
 * Every call starts from bind transforms, so seeking is deterministic. */
export class HumanoidRetargeter {
  private binds = new Map<string, Bind>();
  private palms = new Map<string, THREE.Quaternion>();
  private neckOrigin = new THREE.Vector3();
  private shoulderWidth = 0.36;
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
      const shoulder = upper.getWorldPosition(new THREE.Vector3()),
        target = map(wrist);
      // Monocular depth can place a hand inside the torso. A local torso envelope
      // prevents mesh penetration without rewriting the captured motion asset.
      if (
        target.y < this.neckOrigin.y + 0.04 &&
        target.y > this.neckOrigin.y - 0.65 &&
        Math.abs(target.x) < 0.28
      ) {
        const front =
          this.neckOrigin.z +
          0.15 * Math.sqrt(Math.max(0, 1 - (target.x / 0.3) ** 2));
        target.z = Math.max(target.z, front);
      }
      const l1 = lower.position.length(),
        l2 = hand.position.length();
      const axis = target.clone().sub(shoulder);
      const d = THREE.MathUtils.clamp(
        axis.length(),
        Math.abs(l1 - l2) + 0.005,
        l1 + l2 - 0.005,
      );
      axis.normalize();
      const pole = map(elbow).sub(shoulder);
      pole.addScaledVector(axis, -pole.dot(axis));
      if (pole.lengthSq() < 0.00001)
        pole
          .set(side === "Left" ? 1 : -1, -1, -0.3)
          .addScaledVector(axis, -pole.dot(axis));
      pole.normalize();
      const along = (l1 * l1 - l2 * l2 + d * d) / (2 * d),
        height = Math.sqrt(Math.max(0, l1 * l1 - along * along));
      const bend = shoulder
        .clone()
        .addScaledVector(axis, along)
        .addScaledVector(pole, height);
      const reachable = shoulder.clone().addScaledVector(axis, d);
      this.aim(side + "Arm", bend.sub(shoulder));
      this.aim(
        side + "ForeArm",
        reachable.sub(lower.getWorldPosition(new THREE.Vector3())),
      );
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
      for (const finger of fingers)
        for (let i = 1; i <= 3; i++) {
          const a = joints[`${prefix}_${finger.toLowerCase()}_${i}`],
            b = joints[`${prefix}_${finger.toLowerCase()}_${i + 1}`];
          if (a && b) this.aim(`${side}Hand${finger}${i}`, vec(b).sub(vec(a)));
        }
    }
  }
}
