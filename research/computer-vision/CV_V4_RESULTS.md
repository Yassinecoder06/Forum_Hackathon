# CV v4 — executed quality plan

Implemented and ran the next extraction experiment on the **31 unique Dicta-Sign clips used by the 14 French phrases**. This set differs from the earlier 37-clip experiment, which also included extra regression clips; their aggregate scores should not be compared directly.

## Changes in actual extraction

- Retain native source frames up to 30 fps. Earlier stride selection discarded frames from common 25–30 fps sources. Extra frames are genuine detector observations, not interpolation.
- Confirm cropped-hand recovery at both 384×384 and 512×512 resolutions. Corresponding landmarks must agree in original image pixels relative to palm width; unresolved tiny hands and inconsistent detections are rejected. Agreement is a consistency check, not independent ground truth.
- Keep explicit observation/source/review metadata and existing one-to-one wrist assignment. No missing hand is filled by a pose generator.

`scripts/evaluate_cv_candidates.py` is a resumable, local two-worker evaluator. It preserves original captures, verifies source checksums, saves each candidate extraction and compares **exact matching timestamps**. The gate requires at least 95% of old timestamps to be comparable, no body-observation count decrease, and **zero individual hand-observation losses**, even if other frames improve. It never labels candidates APPROVED or activates them automatically.

## Results

| Measurement | Previous captures | CV v4 candidates |
|---|---:|---:|
| Actual sampled frames | 1,046 | 1,399 |
| Hand observations at shared timestamps | 1,628 | 1,635 |

Eleven hand observations recovered and four lost. **29 clips pass the no-observation-loss gate; two require review:** LIVRE cs96 and PAIN cs103. Source-frame overlays of their lost observations were inspected. Both show overlapping hands; the older detector may itself be estimating occluded points. The gate conservatively holds them for review rather than treating either version as accurate without reference labels.

Complete per-source results, source video SHA256, common-frame counts and gate statuses are in [cv_v4_experiment.json](../../data/source_inventory/cv_v4_experiment.json). Passing this gate does not validate spatial accuracy, handedness, fingertip precision or monocular depth. Existing active motion assets were preserved; fresh captures now use v4. No interpolation was changed.

## Review and verification

`scripts/render_cv_review.py` generated **214 source-image overlays**, paired at exact timestamps, prioritizing recovered/lost observations and adding three fixed positions per clip. The local gallery is `artifacts/cv-review/v4/index.html`. It displays raw hand landmarks, without avatar corrections, smoothing or imputation. Missing hands are left unmarked. The difficult loss cases were visually inspected; this is not a claim that all 214 images have independent expert labels.

The isolated MediaPipe worker actually executed every candidate extraction. **40 backend tests passed**, including cross-scale disagreement rejection, tiny unresolved hands, coordinate restoration, crop boundaries and assignment ambiguity. Training files modified concurrently were left alone.

```bash
python3 scripts/evaluate_cv_candidates.py
.venv-motion/bin/python scripts/render_cv_review.py
.venv/bin/python -m pytest backend/tests -q
```

Candidate files are local under `.runtime/cv-v4/`; review images remain ignored. To test different extractor settings, use a fresh `--output` directory rather than reusing an old extraction cache. Replacing active captures still requires source-overlay review, particularly where the detectors disagree. A labelled reference set is required before claiming measured keypoint accuracy; coverage and cross-scale agreement cannot supply that claim.
