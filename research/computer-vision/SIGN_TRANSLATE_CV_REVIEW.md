# sign/translate CV deep dive and MediaPipe improvement plan

Scope: how [`sign/translate`](https://github.com/sign/translate) at
`928d49ffe223fccedca64162851d4951b9152ecc` uses computer vision on the input
side, what is worth transferring to SignaCulture's MediaPipe capture, and what
should not be copied. Output-side models (pix2pix, cloud pose generation,
translation services) are out of scope.

This is an engineering review, not an accuracy claim. The 100-clip Dicta-Sign
set contains isolated lexical citation forms, not sentences, and has no
hand/landmark ground truth. Measurements below use the cached exports in
`.runtime/captures/`.

## 1. What sign/translate does on the input side

### 1.1 Tracking

- Uses the legacy `@mediapipe/holistic` package in the browser
  (`research/code/src/app/core/services/holistic.service.ts`).
- Holistic options: `upperBodyOnly: false`, `modelComplexity: 1`
  (`research/code/src/app/modules/pose/pose.service.ts:51-56`).
- Sends the full video element to MediaPipe
  (`research/code/src/app/modules/pose/pose.service.ts:76-83`).
- Processes every changed video frame; no stride or target FPS
  (`research/code/src/app/components/video/video.component.ts:94-113`).
- Does not run a second hand detector, hand ROI crop, or missing-hand recovery.
  A frame either has `leftHandLandmarks`/`rightHandLandmarks` or it does not.

### 1.2 Whole-pose normalization

`PoseService.normalizeHolistic` (`pose.service.ts:178-213`):

1. Scale `x` and `z` by image width, `y` by image height.
2. Center on the midpoint of the two shoulders.
3. Divide by the **3D** distance between the two shoulders.
4. Preserve zero coordinates as zero (`l.x === 0 ? 0 : ...`).

This is a translation- and scale-invariant body feature stream. It is not the
rig-space representation that an avatar needs: it discards the hip origin and
re-centers every frame on the shoulders. It also assumes a missing landmark is
encoded as zero, which is ambiguous with a real landmark at the image origin.

### 1.3 Hand canonicalization and handshape features

`HandsService` (`research/code/src/app/modules/sign-writing/hands.service.ts`) and
`PoseNormalizationService` (`research/code/src/app/modules/pose/pose-normalization.service.ts`):

- Palm plane through wrist (0), index MCP (5), pinky MCP (17)
  (`hands.service.ts:71-79`).
- Rotate the hand so that plane normal points to +Z, then rotate in the XY
  plane so the wrist → middle MCP segment (`[0, 9]`) lies on +Y
  (`pose-normalization.service.ts:49-82`).
- Scale that segment to length 200 and subtract the wrist
  (`pose-normalization.service.ts:84-91`).
- Flip X for one hand before feeding the model; the caller passes `isLeft`
  (`pose-normalization.service.ts:64-67`). The inline comment says "right hand",
  but the code flips the left hand. This is another convention that must be
  verified before reuse.
- Feed the resulting 21×3 = 63 values to a 3-layer SimpleRNN/LayerNorm model
  that predicts 261 SignWriting handshape classes
  (`hands.service.ts:46-64`).

They also derive coarse, quantized hand features:

- `plane`: `wall` vs `floor` by comparing `1.5*|Δy|` and `|Δz|` of wrist →
  point 13 (`hands.service.ts:82-93`).
- `rotation`: 8 buckets of 45° from the XY angle of wrist → point 13
  (`hands.service.ts:95-107`).
- `direction`: `me` / `you` / `side` from the palm normal
  (`hands.service.ts:109-143`).

Caveats found in their code:

- `normal()` and the model normalization use point **9** (middle MCP), but
  `plane()` and `rotation()` use point **13** (ring MCP). The comments call
  point 13 `M_CMC`, which is not the middle metacarpal. This inconsistency is
  worth fixing if the method is copied.
- `direction()` has explicit TODOs: no chest-normal subtraction, and the
  `floor` branch is marked "not correct".
- The palm normal from a cross product has an arbitrary sign. Nothing in the
  pipeline stabilizes that sign across frames. A normal flip rotates the
  canonical hand by 180°, which is a large input discontinuity for a per-frame
  classifier.
- The canonicalization is position-based and per-frame. At low resolution,
  MediaPipe hand `z` and bone lengths are noisy; positions inherit that noise.
  Their stateful RNN is presumably doing some temporal smoothing, but the
  feature construction itself is not smoothed.

### 1.4 Face features

`FaceService` (`research/code/src/app/modules/sign-writing/face.service.ts:52-113`):

- Builds a plane through nose (4) and the two inner eye corners (133, 362).
- Uses the nose → point 6 line as the second axis, centers on the nose, and
  canonicalizes all 468 landmarks with the same rotation/scale routine.
- Feeds 468×3 = 1404 values to a model with three heads for eyes, eyebrows and
  mouth (4/3/7 classes).
- Draws the predicted SignWriting symbols at hand-computed 2D face locations.

This is a full face-local frame, not just roll compensation. It handles yaw and
pitch as well as roll, at the cost of relying on face-mesh depth.

### 1.5 Signing-activity detection

`DetectorService` (`research/code/src/app/modules/detector/detector.service.ts`):

- Keeps a 20-frame rolling **mean** shoulder width and divides all landmarks by
  it (`:41-69`).
- Rearranges the pose into a 24-point OpenPose-compatible layout, with only
  nose, neck, shoulders, elbows, wrists, eyes and ears populated; the rest are
  zeros (`:71-110`).
- Treats per-frame 2D landmark displacement multiplied by FPS as an "optical
  flow" proxy (`:116-127`, `:139-153`).
- Feeds a 20-step window to a stateful LSTM(64) and thresholds the 2-class
  softmax at 0.5 (`:129-155`; state wiring in `detector.state.ts`).

There is a real bug at `detector.service.ts:45`:

```ts
const rightHandLandmarks = pose.leftHandLandmarks || new Array(21).fill(EMPTY_LANDMARK);
```

The right hand is a copy of the left hand. Any activity model trained with this
feature has duplicated and missing hand motion. Do not reuse the model as-is.

### 1.6 Avatar animation input

`AnimationService` (`research/code/src/app/modules/animation/animation.service.ts:83-108`):

- Uses the same shoulder-centered normalization on 33 pose + 21 left hand +
  21 right hand = 75 points.
- Feeds an LSTM that outputs 208 quaternions for 52 Mixamo bones.

This confirms that their hand/body normalization is designed for a learned
sequence model, not for direct retargeting of a known skeleton.

### 1.7 Not input CV

- `pix2pix` is a pose-image generator/upscaler used for output rendering, not
  landmark extraction.
- The cloud pose endpoint and SignWriting translation services do not affect
  input tracking.
- The `mediapipe-language-detector` model is a text language detector, not a
  vision model.

## 2. What transfers to SignaCulture

| Technique | sign/translate reference | Our current state | Verdict |
| --- | --- | --- | --- |
| World/3D body coordinates | Holistic world landmarks used indirectly | `capture.py` and `capture.mjs` use `PWLM`/`poseWorld` for body when available | Keep; world body is measurably more stable than image body |
| Shoulder-centered, shoulder-scaled features | `pose.service.ts:178-213` | We use hip origin + shoulder-derived pixel scale | Do not change rig output; optionally add a derived feature stream for models |
| Rolling shoulder scale | 20-frame mean in detector | 7-sample / 300 ms median in both exporters | Keep current; 20-frame mean reduces p95 but adds too much lag |
| Per-frame hand canonicalization | `pose-normalization.service.ts` | Raw finger offsets anchored at the pose wrist | Adopt as **derived handshape features**, not as rig positions |
| Hand plane/rotation buckets | `hands.service.ts:82-107` | Not used | Adopt rotation bucket and a stabilized plane label; skip `direction` for now |
| Face-local frame | `face.service.ts:52-55` | Eye-line roll compensation only | Adopt a face-local 3D frame for geometric features, with 2D fallback |
| Motion-energy / signing activity | `detector.service.ts` | Not used | Adopt the cheap feature for capture QA/segmentation; retrain, don't import the buggy model |
| 24-point OpenPose layout | `detector.service.ts:71-110` | Not used | Only if a model needs that exact input; not needed for the rig |
| Zero-preserving missing landmarks | `pose.service.ts:204-209` | Sparse finger keys; missing hands stay missing | Keep our representation; it is less ambiguous |
| Full-frame every-frame processing | `video.component.ts:94-113` | Offline stride to 30 FPS; webcam capped at 15 FPS | Keep offline 30 FPS; raise the webcam cap after benchmarking |
| MediaPipe handedness labels | Implicit left/right Holistic hands | Nearest-wrist assignment with abstention in `hand_tracking.py` | Keep ours; labels are not trustworthy without mirror handling |

## 3. Validation of the current shoulder-scale change

The uncommitted `feat/docker-studio` work adds `ShoulderScale` to
`backend/app/workers/capture.py:26-40` and the equivalent loop to
`webcam/src/capture.mjs:72-90`.

### 3.1 Test suites

- Full backend suite: **84 passed, 1 skipped** (4.91 s).
- Webcam exporter suite: **8 passed** (60 ms), including the new
  "shoulder scale rejects outliers and expires across tracking gaps" test.

No test regressions.

### 3.2 100-clip scale measurement

Replayed 100 cached Dicta-Sign clips, 3,544 frames, 3,444 adjacent scale pairs.
Metric is relative change in the selected shoulder scale between adjacent
usable frames.

| Configuration | Median | p95 | Max | Lag median | Lag p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Per-frame shoulder | 0.245% | 1.491% | 6.551% | 0% | 0% |
| Current median-7 / 300 ms | **0.177%** | **1.396%** | 6.551% | 0.365% | 2.548% |
| Mean-20 / 700 ms | 0.191% | 0.790% | 2.428% | 0.855% | 4.438% |
| Trimmed mean-20 / 700 ms | 0.177% | 0.858% | 2.895% | 0.834% | 4.617% |

Interpretation:

- The current change is real but modest on this larger set: median adjacent
  scale change falls 28%, p95 falls 6%, max is unchanged.
- The 20-frame mean is closer to sign/translate's detector and cuts p95 by 43%,
  but its lag is 1.7× the current p95 and 2.3× the median. Lag biases hand size
  during real movement, which is worse than a small amount of noise.
- The earlier 6-clip result in `docs/project-log/CONVERSATION_CONTEXT.md` (0.150% → 0.132%
  median, 1.141% → 1.138% p95) is consistent in direction with the 100-clip
  result. The p95 improvement was not significant on six clips; on 100 clips
  it is small but measurable.
- Keep the current default. A better filter (e.g. one-euro with outlier
  rejection) is a possible follow-up, but it must be evaluated against lag, not
  just smoothness.

### 3.3 Why world body coordinates are worth keeping

On the same 100 clips:

| Series | Adjacent-frame change median | p95 |
| --- | ---: | ---: |
| World shoulder distance | 0.251% | 1.542% |
| Image-space 3D shoulder distance | 0.641% | 4.330% |
| World upper-arm length | 0.633% | 3.569% |

The world body path in `capture.py:50-55` and `capture.mjs:54-56` is therefore
the right default. It avoids most of the image-space depth noise. The hand
offsets still use the image-derived `pixel_scale` because Holistic does not
provide world hand landmarks; that is an acceptable approximation, but see
Section 4.5.

## 4. New measurements that shape the plan

### 4.1 Hand coverage in the cached exports

3,544 frames from the 100 clips:

| Hands present | Frames | Share |
| --- | ---: | ---: |
| Both | 2,224 | 62.8% |
| Left only | 94 | 2.7% |
| Right only | 397 | 11.2% |
| Neither | 829 | 23.4% |

At least one hand is missing in 37.2% of frames, and both are missing in 23.4%.
This is the strongest argument for keeping our full-frame fallback and ROI
recovery (`capture.py:132-171`, `webcam/src/detector.js:25-49`). A single-pass
Holistic pipeline like sign/translate's would simply have no hand data in those
frames.

### 4.2 MediaPipe handedness labels are not reliable

Tested the standalone `mp.solutions.hands` model on 6 clips, every 20th frame,
comparing its `Left`/`Right` classification with the nearest pose wrist:

- 9 two-hand samples: **9/9 labels were swapped** relative to the nearest pose
  wrist.
- 8 samples had fewer than two hands detected.

This is the known selfie/mirror convention of MediaPipe handedness. Our
`assign_hands` (`backend/app/workers/hand_tracking.py:46-82`) ignores the label
and solves a one-to-one nearest-wrist assignment with ambiguity abstention.
Keep that design. It is a concrete advantage over any pipeline that trusts
`multi_handedness` or Tasks-vision `handedness` directly.

For the cached Holistic exports, the cache key vs nearest pose wrist mismatch
was only 49/4,939 (0.99%), so Holistic's own left/right slots are mostly
consistent. The separate hand model is the unreliable one.

### 4.3 Palm normal and discrete hand features are noisy at low resolution

4,939 hand observations; median palm size 24.1 px (p25 20.7, p75 27.9) in the
320×240 Dicta-Sign clips. Consecutive-frame comparison, 100 ms max gap:

| Feature | Median change | p95 |
| --- | ---: | ---: |
| 3D palm normal, sign-invariant angle | 5.68° | 58.58° |
| 3D palm normal, 3-frame smoothed | 3.48° | 31.51° |
| 2D in-plane palm orientation, sign-invariant | 0.00° | 0.00° |
| `plane` wall/floor label | 12.4% of pairs change | — |
| 8-way rotation bucket | 21.1% of pairs change | — |

The 2D in-plane palm orientation is stable on this dataset; the 3D normal is
the noisy part. The discrete plane/rotation labels change frequently, which is
partly real hand rotation but also detector noise. A per-frame canonical hand
vector would therefore be noisy. Any adoption needs temporal smoothing and
sign stabilization of the normal.

### 4.4 Hand bone lengths are not a reliable scale reference at this resolution

Within-clip coefficient of variation of bone lengths normalized by palm length
(3D landmarks):

- Thumb: 16.8–21.8% median, up to 42.5% p95.
- Index: 23.6–26.6% median, up to 52.4% p95.
- Middle: 27.0–32.2% median, up to 63.1% p95.
- Ring: 26.8–32.7% median, up to 67.0% p95.
- Pinky: 26.7–29.3% median, up to 59.3% p95.

At 24 px palm size, a 1–2 px landmark error is a large fraction of a finger
bone. This is why sign/translate's "scale the middle metacarpal to 200" step
does not magically produce a stable handshape vector at this resolution. It
also explains why our retargeter's direction-based finger articulation
(`frontend/src/utils/retargetHumanoid.ts:208-248`) is the right choice: it
normalizes bone directions and ignores absolute bone length. Adopting
position-based hand canonicalization directly into the rig would amplify noise.

### 4.5 Current hand scale vs a world-consistent scale

`capture.py:49` converts hand pixels to rig units with `0.4 / shoulder_px`,
i.e. it assumes a 0.40 m shoulder width. A multi-segment least-squares estimate
from world/image body correspondences gives a median ratio of
**1.199** (p5 1.096, p95 1.291). In other words, the current hand scale is about
20% larger than the world body would imply, because MediaPipe's world shoulder
distance in these clips is closer to 0.33 m than 0.40 m.

This does **not** visibly distort finger articulation in our retargeter:
`retargetHumanoid.ts` builds the palm basis from directions, maps the wrist
through the avatar's shoulder-width scale, and derives finger flex/spread from
normalized directions. Absolute hand size cancels out. It would matter only if
a downstream model consumed absolute hand positions in rig units. Do not
change this default without a consumer and a ground-truth check; it is not a
tracking bug today.

## 5. Prioritized implementation plan

### P0 — Keep and measure the current pipeline

1. Keep the world-landmark body path and the 7-sample / 300 ms shoulder
   stabilizer. Both are validated by the full test suites and the 100-clip
   measurement.
2. Keep sparse missing hands and the nearest-wrist hand assignment. Do not
   introduce zero-filled hands or MediaPipe handedness labels.
3. Add the annotation set recommended in `docs/project-log/CONVERSATION_CONTEXT.md`: a small
   number of difficult frames with missing hands, wrong-side assignments and
   bad wrist anchors. Without it, changes to the fallback/ROI defaults are
   unmeasurable.
4. Keep a small metrics harness for every future change:
   hand coverage, wrist-gap distribution, palm-normal stability, plane/rotation
   bucket change rate, and shoulder-scale lag.

### P1 — Webcam temporal tracking and frame pacing

**P1a. Run full-frame hands and face in VIDEO mode.**

`webcam/src/detector.js:12-13` creates `HandLandmarker` and `FaceLandmarker`
with `runningMode:'IMAGE'`, while pose uses `VIDEO` (`:11`). That means hands and
face get no temporal tracking or built-in smoothing. sign/translate's Holistic
runs in video mode and benefits from MediaPipe's temporal filters.

Concrete change:

- Create two instances of each model:
  - `handsVideo` / `faceVideo` in `VIDEO` mode for full-frame calls, using
    `detectForVideo(image, timestamp)`.
  - `handsImage` / `faceImage` in `IMAGE` mode for ROI crops, using `detect()`.
- Pass a strictly increasing timestamp. The worker already receives one from
  `main.js`; use `performance.now()` or `video.currentTime * 1000`.
- Keep the existing association/ROI logic; only the full-frame model instances
  change.
- Close all instances in `close()`.

Expected effect: less hand/face jitter, fewer one-frame dropouts, and possibly
faster full-frame inference because the video pipeline can reuse tracking.
Validate with the fake-camera browser test
(`webcam/scripts/browser-test.mjs`) by comparing hand landmark residual jitter
after removing palm rigid motion, detection coverage, and median inference time
between the current build and the VIDEO-mode build. Do not claim an improvement
without that comparison.

**P1b. Raise the webcam frame cap only after benchmarking.**

`webcam/src/main.js:139` sends at most one frame per 66.7 ms (15 FPS). A 30 FPS
camera therefore loses half its frames before inference. The `busy` flag already
provides backpressure, so the cap can be raised to 30 FPS or replaced with
`requestVideoFrameCallback`.

Caveat: the recorded CPU test had 173 ms median inference (about 5.8 FPS), so
the cap is not the bottleneck on CPU. Raise it for GPU-capable machines, and
keep the adaptive CPU fallback. Measure real inference time before/after.

### P1 — Canonical hand features, not canonical rig positions

Add a pure function (suggested `backend/app/workers/hand_features.py` and a
matching JS helper if needed) that produces a 63-value handshape vector:

1. Build the palm frame from wrist (0), index MCP (5), pinky MCP (17).
2. Stabilize the normal sign against the previous frame (`dot < 0` → flip).
3. Smooth the normal over 3–5 frames (median or one-euro) before building the
   rotation. At 320×240, a 3-frame smooth reduced 3D normal p95 from 58.6° to
   31.5°.
4. Prefer the 2D in-plane frame when palm size is below a threshold (e.g.
   40 px); use 3D when the hand is large or the webcam feed is high resolution.
5. Rotate the wrist → middle MCP segment onto +Y, scale by palm length, center
   on the wrist.
6. Emit both the 63 values and a validity/quality flag (palm size, normal
   residual, smoothing lag).

Use it for handshape classification/retrieval and review UI labels. Do **not**
replace the rig's finger joints with the canonical positions. The retargeter's
direction-based articulation is more robust to the 25–33% bone-length variation
measured above.

If a classifier is added later, train it on our own reviewed data. The
sign/translate hand-shape model is CC BY-NC-SA and was trained on a different
feature convention; importing it would add licensing and domain-shift risk.

### P2 — Face-local geometric normalization

Extend `backend/app/workers/face_features.py:5-23` from an eye-line roll
projection to a face-local frame:

- Eye line: 33 ↔ 263.
- Face vertical: e.g. nose 4 → chin 152, or forehead → chin.
- Normal: cross product of the two.
- Project mouth (13, 14) and brow (105, 159; 334, 386) gaps onto the face plane
  and normalize by inter-ocular distance.

This is the geometric part of sign/translate's `FaceService.normalize`
(`face.service.ts:52-55`) without the 468×3 learned model. It should make
`mouth_open` and brow measurements less sensitive to yaw/pitch, not just roll.

Risks: face-mesh `z` is noisy at low resolution; the current 2D features are
already calibrated into the avatar. Add a new `face_feature_method` version,
compute both for a transition period, and compare on reviewed clips before
changing the avatar's morph inputs.

### P2 — Motion-energy feature for capture QA

Transfer the cheap part of `DetectorService`, not the model:

- Normalize selected upper-body and hand landmarks by a rolling shoulder width.
- Compute per-frame 2D or 3D displacement, multiplied by FPS.
- Smooth over a short window and expose `motion_energy` / `signing_activity` in
  the capture quality object.

Uses: warn when the webcam take has no movement, trim dead frames, segment
training clips, and prioritize review frames. If a classifier is wanted, train
it on our data and fix the `rightHandLandmarks = pose.leftHandLandmarks` bug
before generating features.

### P2 — Landmark validity gating

sign/translate's `isValidLandmark` (`detector.service.ts:112-114`) is a simple
inside-frame check. Our worker currently trusts pose wrists even when they are
outside the frame or low-visibility, and the hand review threshold is
`wrist_disagreement_normalized > 0.12` (`capture.py:180-186`).

Measured wrist gaps for correctly associated hands: median 5.4 px, p95 15.7 px
on 320×240 clips, i.e. about 1.7–4.9% of image width. A 0.12 threshold is 2.4×
the observed p95 and will rarely fire. Consider:

- A tighter review threshold (e.g. 0.05–0.06 normalized), or a threshold
  relative to palm width (e.g. gap > 0.5 × palm width).
- A per-joint `observed`/`visibility` flag for body joints, not just hands.
- Keeping the existing `hand_diagnostics` schema additive so the review UI and
  motion validator do not break.

Validate against annotated difficult frames before changing the default.

### P3 — Do not copy

- Single-pass Holistic with no hand fallback/ROI: 37.2% of cached frames lack at
  least one hand.
- Trusting MediaPipe handedness labels: 9/9 two-hand samples were swapped in our
  standalone test.
- Per-frame position-based hand canonicalization in the rig: bone-length CV is
  25–33% at Dicta-Sign resolution; the retargeter's direction-based approach is
  better.
- The `rightHandLandmarks = pose.leftHandLandmarks` bug and the 24-point
  OpenPose layout with 14 zero points.
- The `direction()` feature as-is: acknowledged TODOs and a wrong floor branch.
- `pix2pix` for input tracking: it is output rendering.

## 6. Limits and open questions

- No hand/landmark ground truth. All numbers are engineering stability and
  coverage metrics, not sign-language accuracy.
- Dicta-Sign clips are 320×240 isolated lexical citation forms. Hand-size and
  depth conclusions may not transfer to 960×540 webcam frames, where 3D hand
  features should be much better.
- The 100-clip cache was produced by an earlier pipeline version. It is valid
  for shoulder-span and hand-landmark stability, but not for attributing
  coverage gains to the current fallback/ROI version.
- The webcam fixture is 15 FPS, so the current browser test cannot expose the
  15 FPS cap. A 30 FPS fixture is needed to measure P1b.
- The 9/9 handedness result is a small sample. It is consistent with the known
  MediaPipe mirror convention, but should be repeated on more clips if it is
  used in a report.

## 7. Recommended next actions

1. Keep the current shoulder/world-body defaults; do not switch to the 20-frame
   mean.
2. Implement P1a (VIDEO-mode full-frame hands/face) and validate with the
   fake-camera browser test.
3. Implement the canonical hand feature function with sign-stabilized,
   temporally smoothed palm frames; keep it separate from the rig.
4. Add the annotation set before touching fallback/ROI defaults or claiming
   accuracy.
5. Evaluate the face-local frame and motion-energy feature as separate,
   versioned additions.
