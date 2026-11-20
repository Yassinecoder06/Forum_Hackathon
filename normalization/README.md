# Anatomy-aware LSF pose normalization

An opt-in PyTorch path for shared-coordinate 3D poses. The existing webcam,
capture import, retargeter, model comparison and trained 2D checkpoints keep
their established inputs.

## Pipeline audit

| Source | Format and axes | Use here |
|---|---|---|
| Backend MediaPipe Holistic and Animics import | 33 pose landmarks in PLM; PWLM is pose-world in the recorded axis convention; 21 wrist-first landmarks per hand in LLM/RLM; FLM facial landmarks. The native importer shoulders-scales image XYZ when world pose is unavailable and anchors image-derived finger offsets at body wrists. | Consume native `signaculture-v1` JSON with the native adapter. Its depth is monocular and mixed-provenance. |
| Webcam MediaPipe Tasks | Body pose world and each hand world have independent origins. Browser export uses wrist anchoring and image-relative finger offsets, and carries frame timing and optional face measurements. | Consume the native export or supply world arrays and independent left/right calibration to `mediapipe_world`. Never concatenate raw world arrays. |
| Review MediaPipe, MMPose DWPose-t, SDPose, Sapiens 0.3B | Image-normalized XY groups. COCO WholeBody and Sapiens Goliath have different source ordering. The review groups them into anatomical hand order and a common upper-body order; Sapiens has 238 face points in the review. | Review coordinates remain 2D. No 3D fit runs on them unless an upstream calibration stage establishes shared XYZ. |
| Review HaMeR | Hand mesh and projected 21 joints, with estimated wrist-relative finger depth. The body still uses image-plane points with assumed depth in the preview. | Hand-local depth alone cannot establish global 3D contact with the torso. |
| SignVerse / transition research | SignVerse preprocessor uses XY plus confidence with recording-median shoulder scale. The native transition manifest uses time-stamped `signaculture-v1` captures, then selects five joints and XY windows; its splits group source videos. Text-to-gloss has separate text supervision. | The native manifest can feed `NativeMotionDataset`. Existing 2D results and text training retain their original preprocessing. |
| Noa avatar | The GLB contains a skin, named upper-body and finger bones including tip nodes, and facial blendshapes. The browser solves arms, constrains finger articulation and applies hand-contact correction. | The Python skeleton reads the actual GLB bind positions. The export supplies position targets to the existing runtime IK, which can apply further constraints. |

## API

```python
from normalization import Config, SignPoseNormalizer, invert
from normalization.adapters import NATIVE_LAYOUT, NativeMotionDataset
from normalization.avatar import noa_skeleton, export_motion
from normalization.retargeting import retarget

normalizer = SignPoseNormalizer(Config(NATIVE_LAYOUT, "rig_3d_xyz"))
# pose [batch, time, joint, xyz], confidence [batch, time, joint], times [batch, time] in seconds
output = normalizer(pose, times=times, confidence=confidence)
recovered = invert(output["torso_local"], output["transforms"])
training_clips = NativeMotionDataset.from_transition_manifest(
    "research/TRANSITION_DATA_MANIFEST.json", "train", normalizer
)
```

For other shared XYZ sources, define a `Layout` with one root, topologically
ordered parents, anatomical shoulders and lower torso, optional palm and
finger chains, and explicit spatial/contact pairs. Confidence defaults to a
finite-value mask. Pass calibrated confidence when the source provides it.
Use masks with all training losses. Dataset samples are unpadded. Batch
same-length clips or pad at collate time and mask padding in downstream
losses. Keep each clip's clock unchanged.

### Transform and outputs

- The scale is the median confident shoulder span of each clip. A
  right-handed frame uses the anatomical right-to-left shoulder axis and
  the lower-torso-to-shoulder axis. A confidence-weighted temporal window
  stabilizes the frame. `torso_local` stores
  `R[t]^T (p[t] - center[t]) / scale`; the clip-reference coordinates use
  the initial valid torso frame. The latter retain torso displacement and
  turns. Both unfiltered outputs invert with the recorded transforms.
  For a moving camera without external calibration, signer motion and
  camera motion cannot be separated.
- The normalizer masks low-confidence and nonfinite points, rejects
  low-confidence isolated spikes with agreeing neighbors and only fills
  short interior gaps. It keeps the input time grid, observed/interpolated
  masks, and finite-difference velocity and acceleration in seconds.
  Discrete masks and median ranks are piecewise differentiable. Optional
  face channels pass through and supplied head rotations transform into
  the reference frame.
- `bone_direction`, `local_rotation`, palm bases, finger bends and
  named signed body-relative vectors retain the information for motion
  and handshape. Local rotations are indexed by the child of an edge and
  use the minimal swing needed to align the rest direction. A pair of
  joints cannot determine axial twist. Palm orientation has its own frame.
- With `skeleton=`, directional reconstruction builds each child at
  `parent + target_length * direction`. The optional offline IK optimizes
  local rotations with fixed-length forward kinematics and weighted
  position, rotation, confidence-gated contact offset, handshape and
  acceleration residuals. The rotation limit term and projection use
  explicitly supplied per-axis bounds in radians. The Noa GLB does not
  specify anatomical limits; the existing browser rig applies its own IK
  and finger constraints. Raw reconstructed joint rotations are not GLB
  animation channels.
- Contact candidates require two observed, confident landmarks. They
  preserve the signed offset instead of forcing points to coincide.
  Configure fingertip-to-chin, mouth, forehead, chest and hand pairs when
  those anchors are available in the shared frame and target skeleton.
  The native layout has fingertip-to-fingertip pairs and hand-to-body
  distances, but its face channels are blendshape measurements or ratios,
  not calibrated 3D face-surface anchors. Its head and neck proxies are
  shown in the figure. An active contact is a geometric candidate, not an
  LSF annotation. IK trades off objectives when contacts are unreachable.

## Reproduce the example

Python dependencies are PyTorch, matplotlib and Pillow. Install pytest for
the tests. Run from the repository root, with Noa set up according to
[avatar provenance](../docs/AVATAR_PROVENANCE.md).

```bash
python -m pytest normalization/tests -q
python -m normalization.demo data/approved_motion_assets/LSF_BONJOUR_COMMONS.json \
  --avatar .runtime/avatar-candidates/Noa.glb \
  --out .runtime/anatomy-normalization/bonjour
```

Use `--device cuda` to run the fit on a GPU. `comparison.gif` plays four
skeleton views at the original sample times and marks the wrist and tips.
`comparison.png` shows the middle sample. `noa-motion.json` can be imported
through **Capturer** in the Studio; it is a research position-target export
for the existing Noa retargeter, with the original source attribution.
Hand evidence is exported through the existing renderer’s observed-hand
metadata. A wrist or finger estimate can inform placement without promoting
a missing hand to a detected one. The renderer uses that evidence to
hold a recent observed pose briefly and fade toward its per-frame neutral
hand when the gap persists.
The options allow running any compatible native clip, without updating
the input recordings.

### Verification

- 17 tests passed, including CPU/CUDA gradient checks, geometric
  invariance and inversion, body-turn retention, fixed lengths, four
  synthetic contact cases, variable time intervals, and gap/outlier masks.
- The opt-in loader normalized all 440 native transition-manifest clips
  across the existing train, validation and test splits, 17,181 frames in
  total. No population statistics were fit. These splits are source-video
  disjoint; signer-disjoint recognition evaluation remains unverified
  because the manifest does not establish signer identity.
- The 32-frame Bonjour example preserved 2,440 ms and face payloads. Mean
  target residual changed from 0.480 to 0.152 shoulder widths after IK.
  The wrist-only residual changed from 0.432 to 0.091 shoulder widths,
  with maximum bone-length deviation below 0.000001 shoulder widths.
  This is fit to extracted landmarks, not hand-pose accuracy. Some of
  the six existing synthetic body-only fixtures had higher unweighted
  position error after fitting the multi-objective loss.
- The backend accepted the native export. Chromium rendered distinct
  poses from the exported clip with the existing Noa renderer.
