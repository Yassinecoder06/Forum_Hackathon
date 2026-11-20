import * as THREE from "three";

/**
 * MetaHuman faces exported to FBX carry FACIAL_* joints but no ARKit blendshapes, so the
 * Studio's morph-target channels (jawOpen, eyeBlink*, brow*, eyeLook*, mouthSmile*) have
 * no effect on them. This driver maps the same channel values onto a handful of facial
 * joints of the baked LOD3 face:
 *
 * - eyeBlink*      -> FACIAL_[LR]_EyelidUpperA/B rotate down about the eyeball centre,
 *                     lower lids rise slightly;
 * - jawOpen        -> FACIAL_C_Jaw + FACIAL_C_LowerLipRotation rotate about the jaw pivot;
 * - brow*          -> FACIAL_[LR]_ForeheadIn/Mid/Out translate up (raise) or down (frown);
 * - eyeLook*       -> FACIAL_[LR]_Eye yaw/pitch (gaze);
 * - mouthSmile/Frown* -> FACIAL_[LR]_LipCorner translate up-out / down;
 * - mouthPucker / mouthFunnel -> lip corners inward, lips forward (funnel also parts the lips);
 * - mouthStretch*  -> lip corners outward;
 * - eyeWide*       -> upper lids up; eyeSquint* -> lower lids up, lower cheeks up.
 *
 * Every call starts from the bind pose, so nothing accumulates and seeking stays
 * deterministic. Amplitudes are display approximations, not a facial-expression model; they
 * give the broad lip shapes of mouthing (rounded / spread / open), not exact visemes.
 */
type Joint = { bone: THREE.Object3D; local: THREE.Quaternion; position: THREE.Vector3; world: THREE.Quaternion; parent: THREE.Quaternion };

const X = new THREE.Vector3(1, 0, 0);
const Y = new THREE.Vector3(0, 1, 0);

export class MetaHumanFaceDriver {
  private joints = new Map<string, Joint>();
  private rotation = new THREE.Quaternion();
  private temp = new THREE.Quaternion();
  private offset = new THREE.Vector3();

  private constructor(root: THREE.Object3D, names: string[]) {
    root.updateMatrixWorld(true);
    for (const name of names) {
      const bone = root.getObjectByName(name);
      if (!bone) continue;
      this.joints.set(name, {
        bone,
        local: bone.quaternion.clone(),
        position: bone.position.clone(),
        world: bone.getWorldQuaternion(new THREE.Quaternion()),
        parent: bone.parent?.getWorldQuaternion(new THREE.Quaternion()) ?? new THREE.Quaternion(),
      });
    }
  }

  /** Returns a driver when the model has MetaHuman facial joints, otherwise null. */
  static create(root: THREE.Object3D) {
    const names = ["FACIAL_C_Jaw", "FACIAL_C_LowerLipRotation"];
    for (const s of ["L", "R"])
      names.push(`FACIAL_${s}_EyelidUpperA`, `FACIAL_${s}_EyelidUpperB`, `FACIAL_${s}_EyelidLowerA`, `FACIAL_${s}_EyelidLowerB`,
        `FACIAL_${s}_ForeheadIn`, `FACIAL_${s}_ForeheadMid`, `FACIAL_${s}_ForeheadOut`, `FACIAL_${s}_Eye`, `FACIAL_${s}_LipCorner`,
        `FACIAL_${s}_LipUpper`, `FACIAL_${s}_LipLower`, `FACIAL_${s}_CheekLower`);
    names.push("FACIAL_C_LipUpper", "FACIAL_C_LipLower");
    const driver = new MetaHumanFaceDriver(root, names);
    return driver.joints.has("FACIAL_C_Jaw") ? driver : null;
  }

  get jointNames() {
    return [...this.joints.keys()];
  }

  /** Rotate a joint about a world (bind, head-relative) axis through its own pivot. */
  private rotate(name: string, axis: THREE.Vector3, angle: number) {
    const j = this.joints.get(name);
    if (!j || !angle) return;
    // world-space rotation conjugated into the joint's bind frame: local' = local * W^-1 R W
    this.rotation.setFromAxisAngle(axis, angle);
    this.temp.copy(j.world).invert().multiply(this.rotation).multiply(j.world);
    j.bone.quaternion.multiply(this.temp);
  }

  /** Translate a joint by a world-space offset (metres, bind/head-relative axes). */
  private move(name: string, x: number, y: number, z: number) {
    const j = this.joints.get(name);
    if (!j || (!x && !y && !z)) return;
    this.offset.set(x, y, z).applyQuaternion(this.temp.copy(j.parent).invert());
    j.bone.position.add(this.offset);
  }

  apply(weights: Record<string, number>) {
    for (const j of this.joints.values()) {
      j.bone.quaternion.copy(j.local);
      j.bone.position.copy(j.position);
    }
    const w = (name: string) => THREE.MathUtils.clamp(weights[name] ?? 0, 0, 1);
    const jaw = w("jawOpen") * 0.32;
    this.rotate("FACIAL_C_Jaw", X, jaw);
    this.rotate("FACIAL_C_LowerLipRotation", X, jaw);
    for (const [s, side] of [["L", "Left"], ["R", "Right"]] as const) {
      const blink = w(`eyeBlink${side}`);
      this.rotate(`FACIAL_${s}_EyelidUpperA`, X, blink * 0.62);
      this.rotate(`FACIAL_${s}_EyelidUpperB`, X, blink * 0.62);
      this.rotate(`FACIAL_${s}_EyelidLowerA`, X, -blink * 0.12);
      this.rotate(`FACIAL_${s}_EyelidLowerB`, X, -blink * 0.12);
      const sign = s === "L" ? 1 : -1;
      const inner = w("browInnerUp"), outer = w(`browOuterUp${side}`), down = w(`browDown${side}`);
      this.move(`FACIAL_${s}_ForeheadIn`, -sign * down * 0.0015, (inner * 0.006) - down * 0.004, 0);
      this.move(`FACIAL_${s}_ForeheadMid`, 0, (inner * 0.003 + outer * 0.004) - down * 0.003, 0);
      this.move(`FACIAL_${s}_ForeheadOut`, 0, outer * 0.005 - down * 0.0015, 0);
      // gaze: eyeLookOut moves the iris towards the character's own side
      const out = w(`eyeLookOut${side}`), inward = w(`eyeLookIn${side}`);
      const yaw = (out - inward) * 0.35 * sign;
      const pitch = (w(`eyeLookDown${side}`) - w(`eyeLookUp${side}`)) * 0.25;
      this.rotate(`FACIAL_${s}_Eye`, Y, yaw);
      this.rotate(`FACIAL_${s}_Eye`, X, pitch);
      const smile = w(`mouthSmile${side}`), frown = w(`mouthFrown${side}`);
      const pucker = Math.max(w("mouthPucker"), w("mouthFunnel")), stretch = w(`mouthStretch${side}`);
      this.move(`FACIAL_${s}_LipCorner`, sign * (smile * 0.004 + stretch * 0.005 - pucker * 0.006),
        smile * 0.004 - frown * 0.003 - stretch * 0.001, -smile * 0.002 + pucker * 0.004);
      this.move(`FACIAL_${s}_LipUpper`, -sign * pucker * 0.002, w("mouthFunnel") * 0.0015, pucker * 0.004);
      this.move(`FACIAL_${s}_LipLower`, -sign * pucker * 0.002, -w("mouthFunnel") * 0.002, pucker * 0.004);
      const wide = w(`eyeWide${side}`), squint = w(`eyeSquint${side}`);
      this.rotate(`FACIAL_${s}_EyelidUpperA`, X, -wide * 0.22);
      this.rotate(`FACIAL_${s}_EyelidUpperB`, X, -wide * 0.22);
      this.rotate(`FACIAL_${s}_EyelidLowerA`, X, -squint * 0.28);
      this.rotate(`FACIAL_${s}_EyelidLowerB`, X, -squint * 0.28);
      this.move(`FACIAL_${s}_CheekLower`, 0, squint * 0.002, 0);
    }
    const pucker = Math.max(w("mouthPucker"), w("mouthFunnel"));
    this.move("FACIAL_C_LipUpper", 0, w("mouthFunnel") * 0.0015, pucker * 0.004);
    this.move("FACIAL_C_LipLower", 0, -w("mouthFunnel") * 0.002, pucker * 0.004);
    for (const j of this.joints.values()) j.bone.updateMatrixWorld(true);
  }
}
