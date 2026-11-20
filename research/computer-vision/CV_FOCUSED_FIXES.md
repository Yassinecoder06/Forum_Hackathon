# Focused fixes for PAIN, LIVRE, TRAVAIL, CHAT and CHERCHER

Implemented and executed a conservative recovery pass on these five existing CV-v4 raw captures. It runs the installed MediaPipe hand tracker over the **actual video frames in forward and reverse order**. A missing observation can be added only when forward/reverse estimates agree on all 21 image landmarks, the body wrist supports its anatomical side and the one-to-one assignment is unambiguous. Duplicate estimates of one physical hand are collapsed before assignment.

The pass does not overwrite existing landmarks, interpolate missing keypoints, retrain weights or invent independent world-hand coordinates. New estimates remain `review_required: true`; agreement between two passes of the same model is not independent ground truth. MediaPipe's image-based landmarks are model estimates ([official documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker?tab=Python)), including when fingers are partially occluded. The experiment uses the already installed 0.10.21 legacy Hands implementation.

## Actual results

| Clip | New hand observations | Raised-visible-wrist missing frames before → after |
|---|---:|---|
| PAIN | 2 left-hand observations, 1560/1640 ms | Left 10 → 8 |
| LIVRE | 0 | Right 1 → 1 |
| TRAVAIL | 1 right-hand observation, 840 ms | Right 1 → 0 |
| CHAT | 0 | Left 4 → 4; right 1 → 1 |
| CHERCHER | 0 | Left 1 → 1 |

The three recovered observations were inspected as overlays on source frames. TRAVAIL's new estimate broadly follows the blurred moving hand. PAIN's hands overlap heavily, so agreement does **not** establish the correctness of its hidden fingertips. Both cases remain review candidates. Existing observations, body/face arrays and source timestamps are unchanged, checked by exact comparison against v4.

Remaining gaps did not produce estimates consistent enough for acceptance. LIVRE, CHAT and CHERCHER are not declared fixed. CHAT/CHERCHER now have an explicit v4-versus-focused-pass raw comparison and direct source review, but this cannot recreate the missing historical raw baseline or supply independently labelled keypoint accuracy. These remaining cases need better source visibility or independent reference labels; filling them just to improve a coverage number would be misleading.

Additional cropped temporal passes were tested but added no recoveries on this set. The final tool retains the simpler full-frame forward/reverse pass. This is a targeted engineering repair, not a general accuracy benchmark.

## Working artifacts

The five corrected **candidate** raw files and native motion files are under `.runtime/cv-v5-focus/<source-sha256>/`. Native poses are reconstructed with the existing capture projection and preserved last-frame hold; sign interpolation was not changed. Original v4 and active library assets are preserved. No quality or linguistic approval status was promoted.

The running source-video comparison gallery is **http://127.0.0.1:8769/**. All five before/after diagnostic videos decode successfully, and source-frame images of each recovery are local under `artifacts/cv-review/focus-fixed/`. The [textual results](../../data/source_inventory/cv_focused_fixes.json) retain source hashes, recovered timestamps, rejected hypotheses and unresolved flags. Images, raw points, model files and environments remain uncommitted.

```bash
.venv-motion/bin/python scripts/refine_cv_focus.py
.venv-motion/bin/python scripts/render_cv_focus.py
.venv/bin/python -m pytest backend/tests -q
```

**41 backend tests pass**, including a new regression check that duplicate crop detections cannot be treated as two different hands and that disagreeing hypotheses are rejected. Real extraction ran on all five clips; source video SHA256 checks and exact preservation of prior observations passed. No further model was downloaded, and concurrent training work was untouched.
