# Hand collision correction

The previous solver omitted rendered hands when source tracking was absent, tested only fingers, and disabled the entire correction whenever any source fingers were near contact. That left folded/resting hands and palms intersecting throughout the requested phrase videos.

The replacement includes all displayed finger segments and three palm capsules per hand, including inferred/rest poses. Contact is evaluated per capsule pair. Source-supported touching allows 3 mm of proxy compression, but deeper penetration still receives correction. Coincident segments get a deterministic separating direction. Eight IK projection iterations and a bounded alternative-direction search resolve configurations in which correcting one pair traps another pair. The alternative search moves hands horizontally or vertically rather than backwards into the torso. A torso volume check also prevents collision corrections from pushing fingers/palms backwards through the chest. Finger articulation and palm orientation are preserved; bone lengths remain unchanged.

Corrections are limited to 60 mm per wrist and precomputed at 50 Hz. Captures, gloss selection and review status are unchanged. Larger corrections change signing locations and remain subject to linguistic review.

## Measured result

Replayed the **same 14 captured-motion sequences**, using the real Noa GLB, against baseline commit `e51c664`. Both versions were evaluated by the same geometry diagnostic, including previously omitted untracked hands and palms: 8 mm finger and 22 mm palm capsules sampled every 40 ms.

The mean of the 14 per-phrase penetration means decreased from **16.26 mm to 0.29 mm**, approximately **98.2%**. Worst sampled proxy penetration decreased from **43.99 mm to 18.24 mm**. [All per-phrase results](HAND_COLLISION_RESULTS.json) are retained. These are capsule-proxy measurements, not exhaustive triangle-mesh intersection tests; residual penetration remains in difficult frames. A projection overlap or natural folded-hand contact is not automatically an error.

Meaningful software checks cover deep source-contact penetration, legitimate surface contact, missing tracking, coincident capsules, fixed bone lengths and deterministic seeking. Frontend tests, lint and production build pass. Actual browser checks pass on three real sequences, including front/side screenshots and pixel-identical seeking. Updated recordings of the 14 requested phrases are saved under ignored `artifacts/phrase-review/collision-final/` and compared with the prior `after/` videos in the local gallery at http://127.0.0.1:8766/.

Reproduce the numerical check with the existing genuine local motion inputs:

```bash
AVATAR_MEASURE_ENTRY=scripts/measure-collisions.ts npm --prefix frontend run measure:avatar -- ../artifacts/collision-review/after.json
```

The baseline run used the four utility modules from `e51c664` in an ignored local snapshot, selected via `AVATAR_BASELINE_RETARGETER`. Raw video, model, environments and source-derived screenshots are not committed.
