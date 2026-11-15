# Avatar hand and eyebrow improvement plan

Requested scope: inspect the working avatar and propose improvements. This review changes no animation, capture or source data. No new model training is needed for the first fixes.

## Evidence from the running application

On 15 November 2026, paused the real Dicta-Sign sequences **Bonjour merci** (cs380/cs905) and **Bibliothèque** (cs490) at five timestamps each. Screenshots, source plans and diagnostics are local in `artifacts/avatar-review/`; `review.html` and `contact-sheet.png` show all ten frames. `capture.mjs` repeats the browser capture against the running app.

- Bibliothèque at 677/968 ms shows hands crowded around the face, with noticeably different finger configurations by 1258 ms. Bonjour merci at 3006 ms shows hands overlapping in the image. Projection overlap alone does **not** establish physical mesh penetration; replay and 3D measurements must distinguish intended contact from errors. Static screenshots cannot measure twitching.
- All samples show a strongly raised/tensed-looking brow. There is a numerical explanation: captured brow ratios span approximately **0.344–0.452**, while `AvatarCanvas.tsx` maps `(ratio - 0.055) * 3` and clamps to [0,1]. Many values saturate. Facial values are selected with a preceding-frame lookup, rather than interpolated.
- Real hand coverage is incomplete. Bonjour has left-hand landmarks in **0/22** frames and right-hand landmarks in **13/22**. Bibliothèque has left **23/30**, right **21/30**; Merci left **22/28**, right **28/28**. An untracked hand must not be presented as a reliable measured handshape.
- Consecutive observed source finger-segment directions have large angular changes: p95 approximately **148° / 79° / 117°** for Bonjour / Bibliothèque / Merci. These pooled measurements flag inputs worth inspecting; they do not prove all changes are noise or measure rendered joint rotations. They exclude pairs where either segment is missing and include real signing motion.

## What the code currently does

`backend/app/workers/capture.py` combines world-body positions with wrist-relative image hand points and estimates hand scale from shoulder width. Its Hands fallback associates detections to the nearest pose wrist independently, without a joint two-hand assignment or temporal identity check. The conversion stores no individual hand confidence in native joint vectors.

`frontend/src/utils/motionInterpolator.ts` interpolates Cartesian points. If a joint exists at only one endpoint, it uses that endpoint throughout the interval. `retargetHumanoid.ts` resets every bone to bind pose on each call; missing finger chains consequently fall back to bind pose. Each finger bone is aimed independently at the observed segment direction. Palm bases have no temporal continuity handling, anatomical finger limits or calibrated local flexion axes. The current torso envelope adjusts wrist targets, but there is no hand-to-hand or hand-to-face collision test. These are candidate causes to measure, not an assertion that each screenshot has the same cause.

## Ordered implementation

### 1. Establish a small, reproducible regression set

Keep these three real clips and their transition boundaries fixed; add an existing two-handed contact sign and an existing non-contact crossing sign after source-video inspection. Show synchronized source video, raw skeleton, retargeted skeleton and skinned avatar, with front and side cameras. Export short slow-motion recordings as well as stills. Log tracking validity, palm orientation, wrist targets, finger joint rotations and collision-proxy distances per frame.

Acceptance: identify whether each defect originates in extraction, interpolation, retargeting or mesh contact. Include manual review of intended contacts; do not score every close pair as a collision. Preserve original raw landmarks and source checksums.

### 2. Correct the eyebrow mapping first

Replace the fixed 0.055 offset with an explicit neutral calibration per recording/signer where suitable neutral reference frames exist. Use head-pose-corrected face distances, separate left/right controls, calibrated inner/outer raise and downward movement, and interpolate channels by timestamps. A clip median may be useful for diagnosis but must not automatically erase a sustained raised-brow expression. When neutral calibration is unavailable, use a conservative documented fallback rather than maximum expression. Verify which morph names and neutral weights the actual Noa mesh supports.

Acceptance: neutral reference frames stay neutral; measured raising and lowering both remain visible; no persistent saturation caused by scale mismatch; deterministic results when seeking. Voice-derived emotion must not overwrite captured nonmanual signing information. Visual tests do not establish linguistic correctness of eyebrows.

### 3. Stabilize hand tracking and missing observations

Add explicit per-frame hand observation validity and derived confidence metadata while keeping raw data unchanged. Assign two detections jointly to anatomical sides using wrist distance plus temporal continuity; reject implausible reassignment rather than duplicating one detection onto both hands. Build a derived playback track with robust outlier rejection, short bounded gap handling and an explicit low-confidence state for long gaps. Do not silently fill long gaps as tracked data.

Acceptance: synthetic dropout/occlusion software tests never swap sides or reset fingers for one frame; real source comparisons preserve quick articulations and intentional contact. Record any imputed samples separately from observations. Inspect the Bonjour left-hand absence before deciding whether a neutral fallback is appropriate.

### 4. Retarget hands in calibrated local joint space

Calibrate palm frames and each finger's local bend/spread axes to the Noa bind skeleton, including the thumb. Derive stable local rotations, maintain quaternion hemisphere continuity, and apply anatomically plausible limits rather than independently aiming every bone without twist constraints. Filter the derived local rotations with a motion-adaptive, bounded method; precompute them per clip so pause, seeking and export are identical. Keep faster valid motion responsive and avoid smoothing handshape distinctions away.

Acceptance: fixed bone lengths; no abrupt bind-pose snaps across short observation gaps; no palm flip from a nearly degenerate basis; reduced isolated angular spikes against the fixed raw-input baseline. Confirm the improvement against source video before selecting numerical filtering thresholds. Test every finger and both hands, seeking in arbitrary order and at transition boundaries.

### 5. Add contact-aware collision correction

Introduce calibrated palm/finger capsules and head/torso proxies, with a debug overlay. First measure penetration rather than moving everything apart. Preserve intentional hand contact and hand-to-face signs; do not impose a universal minimum hand distance. When source evidence supports non-contact but retargeting penetrates, solve a bounded wrist/elbow/orientation adjustment, then refine finger contact under joint limits. Finger segments must follow any wrist/palm correction coherently. Smooth correction weights across time and recheck transitions for swept collisions between keyframes.

Acceptance: fewer measured unintended penetrations on the reviewed regression set, no new contact-breaking errors, bounded changes to source trajectories and no visible correction jitter. Report maximum and accumulated penetration plus wrist correction magnitude. A generic repulsion force is insufficient for this signing task.

### 6. Verify the complete player and export

Compare before/after at the exact saved timestamps and in slow playback, including neutral rest, single signs, concatenated signs and missing-data clips. Add tests for validity propagation, duplicate detection assignment, local rotation continuity, brow calibration/interpolation and contact preservation. Measure frame-time cost on this machine and keep expensive reconstruction outside the render loop. Repeat the existing browser playback/WebM checks and confirm deterministic paused frames and exports.

Completion requires clear source-comparison evidence of improved hands and brows, preserved intentional contacts and handshapes, passing regression checks, and documented remaining capture limitations. This is animation engineering validation; a fluent France-LSF reviewer is still needed for linguistic approval.

## Recommended first implementation batch

Implement **eyebrow calibration/interpolation**, **explicit missing-hand handling**, and **stable local finger rotations** first. Review those results before enabling collision correction. Keep the raw dataset and lexical lookup unchanged; apply any stabilization as an identified, reversible playback derivation.

Local screenshots derive from pending-rights research-preview assets and are not committed or offered as redistributable data.
