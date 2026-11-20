# MetaHuman avatar (experimental, local only)

The Studio can render a MetaHuman character instead of Ready Eva / Noa. The character file is **not in Git**: it is converted locally from a MetaHuman FBX export and served from `.runtime/avatar-candidates/MetaHuman.glb`, like the other review avatars.

**Licence status: confirmed by the project owner (2026-11-20)** for use outside Unreal Engine, i.e. this GLB rendered by Three.js in the web Studio. The GLB is still built locally and kept out of Git; whether the raw character file may also be redistributed in this public repository is a separate decision (see [AVATAR_PROVENANCE.md](AVATAR_PROVENANCE.md)). Output is not a validated LSF translation.

## Build the GLB

Requirements: Blender 4.x or 5.x (tested with 5.1.1 in background mode). Input: a MetaHuman Creator FBX export with baked textures:

```
<Export>/<Name>_Body.fbx  <Name>_Face.fbx  <Name>_Outfit.fbx  Hair/*.fbx
<Export>/Textures/Body_Baked/*.png  Face_Baked/*.png  Grooms_Textures/*.png
```

```bash
blender -b --factory-startup --python scripts/metahuman/convert_metahuman.py -- \
    --src /path/to/MetaHuman_Export --out .runtime/avatar-candidates/MetaHuman.glb
SIGNACULTURE_AVATAR=MetaHuman make dev       # MetaHuman is opt-in; the default stays ReadyEva, then Noa
```

What the converter does:

| Step | Detail |
|---|---|
| One skeleton | The body's UE5 skeleton (341 joints). The face FBX's 843 `FACIAL_*` joints are copied into it under their original parents (the rest poses of the shared joints match exactly). Face, outfit and hair are re-skinned to this single armature. |
| Hair | The hair and brow cards are not skinned in the export. Hair cards are bound rigidly to `head`. Their colour comes from the groom atlas: R is coverage, used as alpha with a 0.35 cut-off, double-sided. The brow cards have no atlas UVs, so they are dropped. The brows painted in the LOD5-7 face texture are blended into the LOD3 skin texture instead. |
| Face materials | The FBX material labels of the baked face are shifted (the skin slot is labelled "Teeth"), so the script assigns roles from geometry instead: skin, teeth, left/right eye, upper lashes. The UE-only transparent layers (eye shell, lacrimal fluid) and the untextured lower-lash cards are removed. |
| Eyes | The iris texture is composited into the sclera texture, with the iris limbus at radius 0.17 of the eye UV. |
| PBR | Base colour, normal map and SRMF textures (G = roughness, B = metallic, the glTF layout). Teeth use their own textures. No textures are exported for the outfit, so it uses plain dark matte colours: a dark top keeps the hands readable, as for interpreters. |
| Finger tips | `<finger>_04_<side>` joints are added at the distal end of each finger, measured on the skinned body mesh. The web retargeter needs them for the last phalanx and for the contact capsules. |
| Pruning | Joints with no skin weight and no needed descendant are removed: 1,194 → 508 joints, of which 303 are `FACIAL_*`. |
| Units / axes | Metres, +Z towards the camera in glTF, WebP textures (1024², eyes and teeth 512²). |

Result for the tested export: **6.0 MB GLB**, height 1.775 m, about 59 k vertices (hair cards about 35 k of them), 508 joints.

## Playback in the Studio

The skeleton is detected automatically: a rig with `upperarm_l` uses the **MetaHuman profile** in `frontend/src/utils/rigProfiles.ts`. The retargeter (`retargetHumanoid.ts`) still works on canonical Ready Player Me names (`LeftArm`, `LeftHandIndex1`, …). The profile maps them to UE5 joints (`upperarm_l`, `index_01_l`, …) and gives explicit limb chains, because MetaHuman joints have corrective/twist helpers as extra children (`upperarm_correctiveRoot_l`, `pinky_03_bulge_l`), so "first child" is not the next limb segment. Only mapped joints are driven; correctives keep their bind pose relative to their parent.

The Ready Player Me path is unchanged: replaying a full motion through the original and the new retargeter on an RPM rig gives a maximum quaternion difference of 0. `frontend/tests/motion.test.ts` checks the MetaHuman profile on a synthetic rig with corrective first children. It checks that the hands land in the same place as the RPM rig, that seeking is deterministic, that no bone stretches, and that correctives are never driven.

### Face

The MetaHuman face has **no ARKit blendshapes**, so the Studio's morph-target code finds no morph targets and does nothing (no error). `frontend/src/utils/metahumanFace.ts` maps the same channel values onto `FACIAL_*` joints instead. Every frame starts from the bind pose:

| Channel (ARKit name) | Joints | Effect | Checked on the mesh |
|---|---|---|---|
| `jawOpen` (from captured `mouth_open`) | `FACIAL_C_Jaw`, `FACIAL_C_LowerLipRotation` | rotation about the jaw pivot, up to 0.32 rad | yes: mouth opens, lower teeth and chin follow |
| `eyeBlinkLeft/Right` (idle blinks) | `FACIAL_[LR]_EyelidUpperA/B`, `…LowerA/B` | upper lid rotates about the eyeball centre, lower lid rises slightly | yes: lids close fully |
| `browInnerUp`, `browOuterUp*`, `browDown*` (captured brow ratios) | `FACIAL_[LR]_ForeheadIn/Mid/Out` | up to 6–7 mm up, 4 mm down | subtle; mostly hidden by this character's fringe |
| `eyeLook*` (idle gaze) | `FACIAL_[LR]_Eye` | yaw ±0.35 rad, pitch ±0.25 rad | not inspected closely |
| `mouthSmile*` / `mouthFrown*` (resting smile) | `FACIAL_[LR]_LipCorner` | 4 mm up/out | subtle |

These amplitudes are display approximations, not a facial-expression model. Mouthing, cheek puffs, lip shapes and eye squints are **not** driven, and LSF uses all of them grammatically.

## Known limits

- **Distribution**: web use is licensed (see above), but the GLB is not committed; it is rebuilt locally from the export.
- **Hair** is rigid on the head (no secondary motion). The hair colour is a fixed dark brown, because the groom colour is not in the export. Card sorting uses alpha-test, so strand edges are hard.
- **Clothing** is the exported outfit mesh with flat colours (no textures in the export).
- **Mesh penetration**: the torso envelope and hand-contact solver of the retargeter were tuned on Ready Player Me bodies. With MetaHuman, B2 test sentences showed 6 mm maximum residual hand-to-hand penetration after correction. Hand-to-body contact (hands on the chest/face) is not solved against the actual MetaHuman mesh.
- **Fingers**: the profile uses `<finger>_01..04`. The metacarpals (`index_metacarpal_l`, …) stay at bind, so cupping of the palm is not reproduced.
- **LOD**: the converted face is the baked LOD3 (about 3.7 k vertices) with 303 facial joints. Higher-LOD faces need a new export.
