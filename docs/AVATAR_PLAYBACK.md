# Avatar playback improvements — executed result

Implemented the practical playback changes from [the reviewed plan](AVATAR_IMPROVEMENT_PLAN.md). Raw captures, lexical lookup, `.pose` exports and approval status remain intact. All stabilization is a derived avatar playback track; future video captures also receive improved fallback hand assignment and observation provenance.

## What changed

- Finger rotations use calibrated local bend/spread axes from the actual Noa bind skeleton, distinct thumb opposition, and bounded joint ranges. Near-degenerate palm frames are rejected. Isolated rotation reversals are rejected only when the neighbouring observations agree; small noise is filtered without flattening sustained gestures.
- Hand gaps up to 180 ms can interpolate between observations. Longer gaps fade to rest, with a continuous middle of the gap; an entirely untracked hand stays at rest. The player explicitly flags incomplete tracking. Derived transition/audio samples are distinct from observed source frames.
- Entire rig tracks are cached, including contact corrections sampled at 50 Hz. Runtime quaternion sampling is independent of previous playback, so seeking, pausing and exports agree. Bone translations and lengths do not change.
- Brow measurements are interpolated by timestamp instead of stepped. Six visually inspected opening rest frames provide engineering reference estimates: Dicta-Sign cs380, cs490, cs905, cs535, cs926 and cs988. References require the expected source URL, time zero and lowered wrists; they are **not linguistic neutral-expression approval**. Explicit `face_calibration` overrides them. Other clips use a restrained fallback; a clip-wide median is never treated as neutral. Missing facial observations fade over 120 ms. Captured face information takes precedence over optional voice emotion.
- Hand fallback detections are assigned jointly and uniquely to anatomical wrists, with recent-position continuity, locked Holistic detections, distance gates and abstention for ambiguous assignments. Future captures record `hand_tracking` observation/source fields and capture-method revision `assignment-v2`. Existing captures are not rewritten.
- A contact guard checks finger capsules **after** filtering and between original keyframes. Source-near contact disables repulsion; uncertain separation has a gradual correction weight. Other overlaps can adjust wrists through arm IK by at most 20 mm per hand while preserving palm orientation and finger articulation. A bounded head guard applies only when the source wrist supports non-contact; the existing torso envelope remains.

## Measured local regression

Used the real Noa GLB skeleton and six unchanged Dicta-Sign captures, sampled at 40 ms. Baseline is the exact retargeter at commit `6d2284f`, verified by SHA256. [Complete numerical results](AVATAR_PLAYBACK_RESULTS.json) include original source-video hashes, timing, proxy measurements and per-clip results. No synthetic signing or training data was used; authored software fixtures test edge cases only.

| Source | Finger local rotation change p95 before → after, degrees per 40 ms |
|---|---:|
| BONJOUR cs380 | 56.6 → 12.7 |
| BIBLIOTHEQUE cs490 | 79.8 → 18.7 |
| MERCI cs905 | 58.3 → 19.4 |
| RENCONTRER cs535 | 35.1 → 12.0 |
| ENSEMBLE cs926 | 78.7 → 26.5 |
| LAVER cs988 | 49.8 → 20.6 |

These are engineering rotation diagnostics, not signing-accuracy scores. The old rotations also used a different, unconstrained articulation method, so a smaller change alone does not prove better meaning or handshape. Paused before/after screenshots and source-frame inspection support the review; expert linguistic assessment remains pending.

Across 410 brow samples, the old mapping saturated 178 samples; the new mapping saturates none. This checks the scale defect, not linguistic validity of facial expressions. Across 88 source-separated samples, mean **capsule-proxy** hand penetration fell from **1.441 mm to 0.378 mm** (74%). Proxies use an 8 mm radius and a 35 mm source-contact guard; they are not a triangle-mesh collision measurement.

Collision correction is **not complete**: RENCONTRER's worst source-separated sample rises from 9.19 to 12.68 mm even though its mean falls substantially. Bibliothèque's maximum is nearly unchanged (8.81 → 8.94 mm), while ENSEMBLE's decreases (14.84 → 5.83 mm). Near-contact uncertainty, intended contacts and missing observations limit automatic correction. No claim is made that every collision or twitch is removed. Full skin collision handling and linguistically checked contact windows remain future work.

Node CPU measurements on this machine: preparation approximately 22–60 ms per tested clip; cached sample p95 below 0.1 ms. Browser preparation for longer combined sequences is higher, approximately 40–130 ms. These are rig costs, not a whole-page FPS or GPU benchmark.

## Verification and local evidence

The unified suite passes **74 backend/schema/geometry tests**, frontend motion/avatar checks, lint and build. Added checks cover unique hand assignment, ambiguous crossings, locked detections, brow interpolation/calibration, missing face fade, short/long hand gaps, sustained gestures, extreme finger curls, source immutability, fixed bone lengths and cached seeking.

The actual browser passes pixel-identical paused seeking in Bonjour merci, Bibliothèque and Rencontrer ensemble laver, with front/side screenshots. The Dicta-Sign browser check also passes real WebM export, `.pose`/MP4/GIF downloads and unknown-word rejection. A real Bibliothèque video was recaptured into an ignored review folder to exercise the updated worker; all 29 sampled body frames were tracked and hand observation metadata was emitted. This is a canary, not a full corpus recapture.

The broader browser suite also passed composition/seeking/WebM, real WAV analysis and pause-aware timing, JSON capture/provenance/persistence/lookup, unknowns, mobile layout and offline errors. Its fresh screenshots and recording are retained locally under `artifacts/avatar-review/browser/`; the earlier tracked evidence was preserved.

Local-only evidence: `artifacts/avatar-review/comparison.html` / `comparison.png`, original and `after/` paused screenshots, `verified/` front/side views, browser report, measurements and canary capture. `artifacts/dictasign/avatar.webm` contains the current avatar export. The comparison viewer plays it at half speed. Pending-rights source videos and derived images/videos remain ignored.

```bash
npm --prefix frontend test
npm --prefix frontend run measure:avatar
CHROME_PATH=/path/to/chrome npm --prefix e2e run test:avatar
CHROME_PATH=/path/to/chrome npm --prefix e2e run test:dictasign

# Optional exact baseline comparison; output remains a local artifact.
git show 6d2284f:frontend/src/utils/retargetHumanoid.ts > /tmp/signaculture-avatar-before.ts
AVATAR_BASELINE_RETARGETER=/tmp/signaculture-avatar-before.ts npm --prefix frontend run measure:avatar -- ../artifacts/avatar-review/measurements-before.json --baseline
```

The measurements require the actual local Noa model and six source captures; missing assets cause failure, never a synthetic fallback. The remaining work is focused source/contact review and better observation quality, rather than more language-model training.
