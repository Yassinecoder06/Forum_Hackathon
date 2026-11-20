import * as THREE from "three";

/**
 * Skeleton naming profiles. The retargeter and the display layers address bones by
 * canonical (Ready Player Me) names; a profile maps those names to a rig's own joints.
 *
 * - `rpm`: Ready Player Me rigs (ReadyEva, Noa). Identity mapping, unchanged behaviour.
 * - `metahuman`: Epic MetaHuman / UE5 skeleton converted by scripts/metahuman/convert_metahuman.py
 *   (adds `<finger>_04_<side>` tip joints). No ARKit blendshapes: the face is driven
 *   through a few FACIAL_* joints instead (see metahumanFace.ts).
 */
export type RigProfileId = "rpm" | "metahuman";

export type RigProfile = {
  id: RigProfileId;
  /** canonical -> rig joint name; `null` means identity (use the canonical name). */
  bones: Record<string, string> | null;
  /** canonical bone -> canonical child that defines its bind direction. Without it the
   * first child joint is used (the original RPM behaviour). */
  chain: Record<string, string> | null;
  credit: string;
};

const FINGERS = ["Thumb", "Index", "Middle", "Ring", "Pinky"];
const SIDES = [["Left", "l"], ["Right", "r"]] as const;

export const RPM_PROFILE: RigProfile = {
  id: "rpm",
  bones: null,
  chain: null,
  credit: "UPF-GTI / Ready Player Me",
};

function metahumanBones() {
  const bones: Record<string, string> = {
    Hips: "pelvis",
    Spine: "spine_01",
    Spine1: "spine_03",
    Spine2: "spine_04",
    Neck: "neck_01",
    Head: "head",
  };
  for (const [side, s] of SIDES) {
    bones[`${side}Shoulder`] = `clavicle_${s}`;
    bones[`${side}Arm`] = `upperarm_${s}`;
    bones[`${side}ForeArm`] = `lowerarm_${s}`;
    bones[`${side}Hand`] = `hand_${s}`;
    for (const finger of FINGERS)
      for (let i = 1; i <= 4; i++)
        bones[`${side}Hand${finger}${i}`] = `${finger.toLowerCase()}_0${i}_${s}`;
  }
  return bones;
}

function metahumanChain() {
  // MetaHuman joints carry corrective/twist helpers as extra children
  // (upperarm_correctiveRoot_l, pinky_03_bulge_l…), so "first child" is not the limb.
  const chain: Record<string, string> = { Neck: "Head" };
  for (const [side] of SIDES) {
    chain[`${side}Shoulder`] = `${side}Arm`;
    chain[`${side}Arm`] = `${side}ForeArm`;
    chain[`${side}ForeArm`] = `${side}Hand`;
    chain[`${side}Hand`] = `${side}HandMiddle1`;
    for (const finger of FINGERS)
      for (let i = 1; i <= 3; i++) chain[`${side}Hand${finger}${i}`] = `${side}Hand${finger}${i + 1}`;
  }
  return chain;
}

export const METAHUMAN_PROFILE: RigProfile = {
  id: "metahuman",
  bones: metahumanBones(),
  chain: metahumanChain(),
  credit: "Epic Games MetaHuman",
};

/** Detect the profile from the loaded skeleton (no configuration needed). */
export function detectRigProfile(root: THREE.Object3D): RigProfile {
  let metahuman = false;
  root.traverse((o) => {
    if (o instanceof THREE.Bone && o.name === "upperarm_l") metahuman = true;
  });
  return metahuman ? METAHUMAN_PROFILE : RPM_PROFILE;
}

export function rigBoneName(profile: RigProfile, canonical: string) {
  return profile.bones ? profile.bones[canonical] ?? canonical : canonical;
}

export function findRigBone(root: THREE.Object3D, profile: RigProfile, canonical: string) {
  return root.getObjectByName(rigBoneName(profile, canonical));
}
