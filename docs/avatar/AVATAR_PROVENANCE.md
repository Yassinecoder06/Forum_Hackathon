# Local rigged avatar provenance

## FrenchWoman (MetaHuman) — default when installed

The Studio prefers **FrenchWoman**, a MetaHuman character exported from Epic Games' MetaHuman tools as FBX (body, face, outfit, hair and eyebrow cards, baked textures). The backend order is `FrenchWoman`, then `ReadyEva`, then `Noa` (`AVATARS` in `backend/app/main.py`); `SIGNACULTURE_AVATAR` still forces one.

- Source export (not in Git): `.runtime/avatar-sources/FrenchWoman/`
- Built model (not in Git): `.runtime/avatar-candidates/FrenchWoman.glb` (~7 MB)
- Build: `scripts/build-metahuman-avatar.py`, run with Blender's Python module in an isolated tool environment:

```bash
uv venv .runtime/tools/bpy-venv --python 3.11
uv pip install --python .runtime/tools/bpy-venv/bin/python bpy==4.2.0
.runtime/tools/bpy-venv/bin/python scripts/build-metahuman-avatar.py \
    .runtime/avatar-sources/FrenchWoman .runtime/avatar-candidates/FrenchWoman.glb
```

The conversion adapts the asset to the Studio rather than changing the retargeter:

- Body, face, outfit, hair and eyebrows are merged onto one skeleton. Twist, corrective and helper bones are removed and their skin weights merged into the nearest kept bone, so every arm and finger bone has a single bone child (the retargeter derives each bone's direction from its child).
- Core bones are renamed to the Studio names (`Hips`, `Spine1/2`, `Neck`, `Head`, `LeftArm`, `LeftForeArm`, `LeftHand`, `LeftHandIndex1`…); fingertip (`…4`) and `HeadTop_End` bones are added.
- Face material slots are classified by geometry. This export carries material names shifted by one slot (the head skin arrives labelled "Teeth"). The transparent eye shell and tear film layers are dropped.
- Facial blendshapes are **derived approximations**, not MetaHuman RigLogic:
  - `jawOpen` rotates the jaw joint and carries the lower-lip joints with it.
  - `eyeBlinkLeft/Right` move the upper-lid joints onto the lower lid. The lids close about 90%, and a sliver of iris stays visible at full blink.
  - `browInnerUp`, `browOuterUp*` and `browDown*` are smooth displacement fields, because no brow joints are weighted at this LOD.
- Unreal computes hair colour procedurally, so the hair uses a chosen dark brown with opacity from the card atlas coverage channel. The iris is composited into the sclera texture. Outfit colours are chosen (dark top for hand legibility).
- glTF keeps 4 joint influences per vertex; some face vertices had more and are renormalized.

**Rights boundary:** MetaHuman assets are governed by Epic Games' MetaHuman and Unreal Engine licence terms. The converted GLB is kept local and is not bundled in Git. Before publishing it on a public site (where the GLB itself becomes downloadable), confirm that those terms permit that distribution of a converted character. No such confirmation is recorded here.

## Ready Eva and Noa

The studio can render the public **Ready Eva** demo GLB served by UPF-GTI and listed in the Performs application. **Noa** remains installed as the reference rig for the measurement scripts (`frontend/scripts/measure-*.ts`). Both share the same Ready Player Me skeleton and ARKit blendshapes, so retargeting is identical. With no explicit setting, the studio uses FrenchWoman when installed, then ReadyEva, then Noa. The studio and video previews credit the selected rig. Use `SIGNACULTURE_AVATAR=ReadyEva|Noa` to require a particular model. An explicitly selected missing file produces an error.

- Ready Eva: https://resources.gti.upf.edu/3Dcharacters/ReadyEva/ReadyEva.glb
- Ready Eva SHA-256: `b6dc3c8e34df93aad378d24daa38f305c753f45fac147608fc0149218812bbf0`
- Local path: `.runtime/avatar-candidates/ReadyEva.glb`

Noa (reference rig):
- Model: https://resources.gti.upf.edu/3Dcharacters/Noa/Noa.glb
- Performs model listing: https://github.com/upf-gti/performs/blob/main/js/Performs.js
- SHA-256: `610f0b246a3239a0bc8bc8989454ce03c87b7a47fda723bd3a0266e141ca3cbe`
- Model creator/platform credit: UPF-GTI demo / Ready Player Me (Wolf3D mesh and facial blendshapes).
- Local path: `.runtime/avatar-candidates/Noa.glb`. The selected model is served by the local API at `/avatar/model`.

**Rights boundary:** The Performs MIT software licence is not treated as a blanket licence for avatar assets. This model is downloaded for local evaluation and is not bundled in Git. Per-asset redistribution/publication permission is pending. No commercial or unrestricted redistribution right is asserted. The application must retain the preview/review restriction; verify model rights or replace it with an explicitly licensed character before publication. Provider background: https://readyplayerme.hashnode.dev/how-to-create-3d-characters-for-your-unity-game-with-our-avatar-sdk (noncommercial-use guidance, not a per-Noa licence).

Reproduce the local setup explicitly with `python3 scripts/setup-avatar.py`. The API never downloads a character during a request. If neither supported file is installed, the studio displays an error.

Retargeting uses two-bone arm IK, palm orientation from measured wrist/index/middle/pinky landmarks, and finger bone rotations. It preserves the character's bind bone lengths, starts from bind rotations at every sample, and remains deterministic when seeking. Missing finger measurements fall back to the character's neutral finger posture; that posture is a rendering fallback, not reconstructed signer evidence. Facial ratios drive available blendshapes and take priority over experimental audio expression. This renderer is not linguistic validation.

A simple torso envelope with palm clearance limits mesh penetration caused by monocular depth error. This is an approximate display constraint, not a full collision solver or correction to the original captured coordinates.

A display-only "life" layer (`frontend/src/utils/idleLife.ts`) adds breathing, blinks, gaze saccades and a slow head sway on top of the retargeted pose. It never modifies captured motion; head sway is damped while signing because head movement can be grammatical in LSF, and the resting smile is only shown when no sign is playing. The café backdrop (`frontend/src/utils/stageBackdrop.ts`) is procedural and mid-toned to keep hands legible.
