# Experimental LSF motion engines (C0, B2)

`POST /lsf/generate` turns a French sentence into a `signaculture-v1` MotionFile played by the Studio avatar ("Générer avec les moteurs LSF expérimentaux"). Two engines from the separate research project are integrated, **inference only**:

| Engine | What it does | Strength | Weakness |
|---|---|---|---|
| `b2` | Splits the text into words and plays the per-word sign clip of each word found in a 942-word lexicon. Signs are joined with learned LSF transitions (an MLP trained on about 24 k real in-between movements), real prosody and a 6 Hz zero-phase low-pass filter. | Real signing clips with natural transitions. | Only lexicon words are signed, in French word order. Other words are skipped and listed as unknown. |
| `c0` | Text-to-pose generator. Frozen multilingual-e5 text states feed a 6-layer transformer that predicts motion tokens of a VQ-VAE tokenizer (time ×2, 2-level residual VQ). Sampling uses classifier-free guidance 3.0, temperature [1.0, 0.6], no top-k and stop threshold 0.2; a hold head adds pauses (×1.5). | Covers any sentence. On automatic proxies it reaches the score of real signing. | Clips about 10 % longer than real; pauses come from a hold head. Whether the signs are real LSF is unknown. |

**No output has been checked by an LSF reader or a deaf signer.** The automatic scores are proxies (a text–motion matching model trained on the same data, and a sign-spotting check). Every generated MotionFile has `synthetic: true` and `generation.translation_validated: false`. It carries a French warning, and the Studio shows that the motion is not a validated translation.

## Model files (private, not in Git)

The engines were trained on LSF-interpreted press briefings. The weights, lexicon clips and any pose data derived from these interpreters are **private**: never commit them, and never put them in a public bucket or image. The API never downloads anything at request time. If a file is missing, `/lsf/generate` returns **503** with the list of missing files, and `/capabilities` reports `lsf_engines: {c0: false, b2: false}`.

Either use one folder (default `.runtime/lsf-models`, git-ignored):

```
$SIGNACULTURE_LSF_MODEL_DIR/
  c0/tok_d2_l2.pt  c0/gen_d2_l2_v3b_hold.pt
  b2/transition_lsf.pt  b2/prosody.json  b2/lexicon_union_corpus_fast.npz
```

or point to each file:

| Variable | File |
|---|---|
| `SIGNACULTURE_LSF_C0_TOKENIZER` | C0 tokenizer checkpoint (`tok_d2_l2.pt`) |
| `SIGNACULTURE_LSF_C0_GENERATOR` | C0 generator checkpoint (`gen_d2_l2_v3b_hold.pt`: generator v2b + hold head; an older checkpoint without a hold head still loads and runs without holds) |
| `SIGNACULTURE_LSF_B2_TRANSITIONS` | B2 transition model (`transition_lsf.pt`) |
| `SIGNACULTURE_LSF_B2_PROSODY` | B2 prosody statistics (`prosody.json`) |
| `SIGNACULTURE_LSF_B2_LEXICON` | B2 lexicon (`lexicon_union_corpus_fast.npz`, 942 words) |
| `SIGNACULTURE_LSF_TEXT_ENCODER` | optional local folder of `intfloat/multilingual-e5-base` |
| `SIGNACULTURE_LSF_DEVICE` | `auto` (default), `cpu` or `cuda` |

C0 also needs the public text encoder `intfloat/multilingual-e5-base`. It is loaded with `local_files_only=True`, so download it **once, explicitly**, outside the API:

```bash
pip install -r backend/requirements-lsf-engines.txt        # torch, transformers, scipy (optional extras)
python -c "from transformers import AutoModel, AutoTokenizer as T; m='intfloat/multilingual-e5-base'; T.from_pretrained(m); AutoModel.from_pretrained(m)"
```

The Docker image does not install PyTorch. There, the endpoint answers 503 unless an image with the extras and a private model mount is built.

Measured on the development PC (12 CPU threads, models loaded from a local disk): the first request loads the models in **about 55 s** on CPU (60 s with CUDA). After that, a C0 sentence takes **2–5 s on CPU**. Generations are serialised, one at a time.

## From engine poses to a MotionFile

`backend/app/services/lsf_pose_convert.py`:

- **Input:** (T, 157, 2|3) at 25 fps, normalised: origin at the shoulder midpoint, unit = shoulder width, y up, +x = signer's left. Indices 0:23 are MediaPipe pose 0..22, 23:44 the left hand, 44:65 the right hand, 65:157 the face (lips, brows, eyes: sorted MediaPipe face-mesh ids).
- **Scale:** shoulder width becomes 0.36 m and the shoulder midpoint sits at 1.45 m, which puts the nose at about 1.6 m like the captured assets.
- **Joints:** `head` (nose), `neck` (shoulder midpoint), `hips`, `left/right_shoulder/elbow/wrist`, and `left/right_<thumb|index|middle|ring|pinky>_1..4` (MediaPipe hand 1..20). Hand landmark 0 replaces the pose wrist when present.
- **Depth:** engine depth is ignored, because C0 outputs z = 0. For each segment (upper arm, forearm, each phalanx) with 3D length L and observed 2D length d, `z_child = z_parent + sqrt(max(L² − d², 0))`: foreshortening always goes towards the camera. L is the 95th percentile of d over the clip. For the arm it is kept within [prior, 1.2 × prior] (upper arm 0.78 and forearm 0.68 shoulder widths), so lexicon clips with odd proportions cannot throw the hands forward. Wrists are kept at least 0.11 m in front of the shoulder plane over the chest and 0.13 m over the face.
- **Face:** `mouth_open`, `brow_left`, `brow_right` use the formula of `backend/app/workers/face_features.py`: same landmarks, projection on the eye-line normal, ×16/9 for the broadcast aspect. `face_calibration` is the B2 lexicon rest pose (brows 0.428 / 0.424); C0 uses the same constants.
- **Frames:** missing points are interpolated over time. A frame whose hand was missing gets `hand_tracking.<side>.observed = false` and no finger joints, so the player falls back to its neutral hand. Output fps is 25 by default (`fps_out` resamples).

Tests: `backend/tests/test_lsf_pose_convert.py` (geometry fixtures, no interpreter data) and `backend/tests/test_lsf_engines_api.py` (503 without files, validation, MotionFile contract with a stub engine, MetaHuman opt-in).

## API

```http
POST /lsf/generate
{"text": "Bonjour, je veux apprendre la langue des signes.", "engine": "b2" | "c0", "seed": 0}
→ 200 MotionFile + "generation": {engine, lemmas, unknown (B2) | tokens, sampling (C0), seconds, translation_validated: false}
→ 422 empty or > 400 characters, unknown engine;  503 model file / dependency missing
```

## Body layer in the player

The Studio retargeter originally drove only the arms (IK to the wrists), the hands and the fingers; spine, clavicles, neck and head stayed at rest. Engine poses move the whole upper body (torso sway, shoulder tilt, head nods and tilts), so for MotionFiles from the engines (`generation` present or `capture_method` starting with `lsf-engine`) `frontend/src/utils/retargetHumanoid.ts` adds a body layer:

- **Torso:** sideways and forward sway of the neck, relative to the clip's median, becomes a roll/pitch spread over `Spine`, `Spine1`, `Spine2` (at most ±14° / ±11°).
- **Shoulders:** each shoulder's height relative to the neck, relative to the median, raises or lowers its clavicle (−4 to +5 cm).
- **Head:** the rotation from the median neck→nose direction to the current one, plus the eye-line tilt (`left_eye` / `right_eye` joints, outer eye corners), split 40 % neck / 60 % head, at most about 26°.
- **Arms:** wrist targets are placed relative to the avatar's current neck, so the arms travel with the body.

Captured clips keep the original arms-and-hands behaviour (tested in `frontend/tests/motion.test.ts`). Body yaw is not estimated, because 2D shoulders do not give its direction.

## Known limits

- **Not a translation.** B2 signs the French words it knows, in French order: no LSF grammar, space, classifiers or role shift. C0's signs are unverified.
- **Depth is a guess.** The always-towards-the-camera rule is wrong when a hand moves behind the body plane or when a palm faces away (fingers then bend the wrong way in depth). Hand-to-hand and hand-to-face contacts come from 2D proximity plus the player's contact solver, not from measured depth.
- **Face:** only mouth opening and brow height reach the avatar. Lip shapes (mouthing), eye gaze and eye aperture from the engines are not used yet.
- **C0:** holds come from a hold head (probability per 0.16 s step that real signing is still there, ×1.5); predicted hold stretches are made still with a smooth move in and out. Clips are about 10 % longer than real signing and handshapes slightly less varied. One sample per request (`seed` makes it reproducible).
- **B2:** words outside the 942-word lexicon are dropped, and some lexicon clips come from isolated-sign recordings with other proportions and framing.
