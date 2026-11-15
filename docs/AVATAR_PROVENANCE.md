# Local rigged avatar provenance

The renderer now uses the public **Noa** demo GLB served by UPF-GTI and listed in the Performs application, rather than a body built from independent primitive meshes.

- Model: https://resources.gti.upf.edu/3Dcharacters/Noa/Noa.glb
- Performs model listing: https://github.com/upf-gti/performs/blob/main/js/Performs.js
- SHA-256: `610f0b246a3239a0bc8bc8989454ce03c87b7a47fda723bd3a0266e141ca3cbe`
- Model creator/platform credit: UPF-GTI demo / Ready Player Me (Wolf3D mesh and facial blendshapes).
- Local path: `.runtime/avatar-candidates/Noa.glb`; served by the local API at `/avatar/model`.

**Rights boundary:** The Performs MIT software licence is not treated as a blanket licence for avatar assets. This model is downloaded for local evaluation and is not bundled in Git. Per-asset redistribution/publication permission is pending. No commercial or unrestricted redistribution right is asserted. The application must retain the preview/review restriction; verify model rights or replace it with an explicitly licensed character before publication. Provider background: https://readyplayerme.hashnode.dev/how-to-create-3d-characters-for-your-unity-game-with-our-avatar-sdk (noncommercial-use guidance, not a per-Noa licence).

Reproduce the local setup explicitly with `python3 scripts/setup-avatar.py`. The API never downloads a character during a request. Missing models produce a visible error; there is no silent geometric substitute.

Retargeting uses two-bone arm IK, palm orientation from measured wrist/index/middle/pinky landmarks, and finger bone rotations. It preserves the character's bind bone lengths, starts from bind rotations at every sample, and remains deterministic when seeking. Missing finger measurements fall back to the character's neutral finger posture; that posture is a rendering fallback, not reconstructed signer evidence. Facial ratios drive available blendshapes and take priority over experimental audio expression. This renderer is not linguistic validation.

A simple torso envelope with palm clearance limits mesh penetration caused by monocular depth error. This is an approximate display constraint, not a full collision solver or correction to the original captured coordinates.
