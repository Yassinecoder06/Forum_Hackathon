# Local rigged avatar provenance

The studio renders the public **Ready Eva** demo GLB (default) served by UPF-GTI and listed in the Performs application. **Noa** remains installed as the reference rig for the measurement scripts (`frontend/scripts/measure-*.ts`). Both share the same Ready Player Me skeleton and ARKit blendshapes, so retargeting is identical. With no explicit setting, the studio uses installed ReadyEva first and Noa next. The studio and video previews credit the selected rig. Use `SIGNACULTURE_AVATAR=ReadyEva|Noa` to require a particular model. An explicitly selected missing file produces an error.

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

**MetaHuman (opt-in, `SIGNACULTURE_AVATAR=MetaHuman`)**: converted locally from a MetaHuman Creator FBX export with `scripts/metahuman/convert_metahuman.py` (Blender) to `.runtime/avatar-candidates/MetaHuman.glb`; never downloaded, not committed. Use outside Unreal Engine (web rendering) is licensed, as confirmed by the project owner on 2026-11-20; redistributing the raw GLB in Git is a separate, open decision. It uses the UE5 skeleton (rig profile `metahuman`) and facial joints instead of ARKit blendshapes; see [METAHUMAN.md](METAHUMAN.md).

Reproduce the local setup explicitly with `python3 scripts/setup-avatar.py`. The API never downloads a character during a request. If neither supported file is installed, the studio displays an error.

Retargeting uses two-bone arm IK, palm orientation from measured wrist/index/middle/pinky landmarks, and finger bone rotations. It preserves the character's bind bone lengths, starts from bind rotations at every sample, and remains deterministic when seeking. Missing finger measurements fall back to the character's neutral finger posture; that posture is a rendering fallback, not reconstructed signer evidence. Facial ratios drive available blendshapes and take priority over experimental audio expression. This renderer is not linguistic validation.

A simple torso envelope with palm clearance limits mesh penetration caused by monocular depth error. This is an approximate display constraint, not a full collision solver or correction to the original captured coordinates.

A display-only "life" layer (`frontend/src/utils/idleLife.ts`) adds breathing, blinks, gaze saccades and a slow head sway on top of the retargeted pose. It never modifies captured motion; head sway is damped while signing because head movement can be grammatical in LSF, and the resting smile is only shown when no sign is playing. The café backdrop (`frontend/src/utils/stageBackdrop.ts`) is procedural and mid-toned to keep hands legible.
