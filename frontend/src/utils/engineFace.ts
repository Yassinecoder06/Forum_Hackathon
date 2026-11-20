import * as THREE from "three";

/**
 * Face channels measured by the LSF engines (lsf_pose_convert.face_ratios) -> ARKit-named weights.
 *
 * - mouth_width / mouth_corner / eye_open_* are read RELATIVE to the clip's neutral (face_calibration, the
 *   lexicon rest face): narrower lips -> mouthPucker / mouthFunnel, wider -> mouthStretch, raised corners ->
 *   mouthSmile, lowered -> mouthFrown, less open eyes -> eyeSquint / eyeBlink, wide open -> eyeWide.
 * - nm_* are the grammatical non-manual markers (questions, negation) baked in by the backend; they are a
 *   separate layer: emotion never writes them and they win over the measured brows.
 *
 * Thresholds are display calibrations measured on the lexicon's range (5–95 %), not a facial-expression model.
 */
const clamp = (v: number) => THREE.MathUtils.clamp(v, 0, 1);

export type EngineFace = {
  weights: Record<string, number>;
  /** Grammatical brow raise / furrow and head motion (radians), 0 when no marker is active. */
  nonmanual: { browRaise: number; browFurrow: number; headYaw: number; headPitch: number };
};

export function engineFace(face: Record<string, number> | null | undefined, neutral?: Record<string, number>): EngineFace {
  const weights: Record<string, number> = {};
  const nonmanual = { browRaise: 0, browFurrow: 0, headYaw: 0, headPitch: 0 };
  if (!face) return { weights, nonmanual };
  const finite = (k: string) => Number.isFinite(face[k]) && Number.isFinite(neutral?.[k]) && Math.abs(neutral![k]) > 1e-6;
  if (finite("mouth_width")) {
    const rel = face.mouth_width / neutral!.mouth_width;
    weights.mouthPucker = clamp((0.92 - rel) / 0.18) * 0.8;
    const stretch = clamp((rel - 1.06) / 0.16) * 0.7;
    weights.mouthStretchLeft = stretch;
    weights.mouthStretchRight = stretch;
    if (Number.isFinite(face.mouth_open)) weights.mouthFunnel = clamp((0.96 - rel) / 0.15) * clamp(face.mouth_open * 8) * 0.8;
  }
  if (Number.isFinite(face.mouth_corner) && Number.isFinite(neutral?.mouth_corner)) {
    const d = face.mouth_corner - neutral!.mouth_corner;
    weights.mouthSmileLeft = weights.mouthSmileRight = clamp((d - 0.015) / 0.07) * 0.6;
    weights.mouthFrownLeft = weights.mouthFrownRight = clamp((-d - 0.02) / 0.06) * 0.5;
  }
  for (const [key, side] of [["eye_open_left", "Left"], ["eye_open_right", "Right"]] as const) {
    if (!finite(key)) continue;
    const rel = face[key] / neutral![key];
    weights[`eyeSquint${side}`] = clamp((0.75 - rel) / 0.35) * 0.7;
    weights[`eyeBlink${side}`] = clamp((0.4 - rel) / 0.3);
    weights[`eyeWide${side}`] = clamp((rel - 1.8) / 1.2) * 0.5;
  }
  if (Number.isFinite(face.nm_brow_raise)) nonmanual.browRaise = clamp(face.nm_brow_raise);
  if (Number.isFinite(face.nm_brow_furrow)) nonmanual.browFurrow = clamp(face.nm_brow_furrow);
  if (Number.isFinite(face.nm_head_yaw)) nonmanual.headYaw = THREE.MathUtils.clamp(face.nm_head_yaw, -0.4, 0.4);
  if (Number.isFinite(face.nm_head_pitch)) nonmanual.headPitch = THREE.MathUtils.clamp(face.nm_head_pitch, -0.3, 0.3);
  return { weights, nonmanual };
}

/** Rotate a bone by a world-space yaw (about +Y) and pitch (about +X, positive = nod forward). */
export function turnHead(head: THREE.Object3D, yaw: number, pitch: number) {
  if (!head.parent || (!yaw && !pitch)) return;
  const world = head.getWorldQuaternion(new THREE.Quaternion());
  const parent = head.parent.getWorldQuaternion(new THREE.Quaternion());
  const rotation = new THREE.Quaternion().setFromEuler(new THREE.Euler(pitch, yaw, 0, "YXZ"));
  head.quaternion.copy(parent.invert().multiply(rotation.multiply(world)));
  head.updateWorldMatrix(false, true);
}
