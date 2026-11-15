# Actual keypoint extraction audit

Audited **all 1,025 existing Dicta-Sign raw captures**, independently of avatar smoothing and collision correction. The [machine-readable audit](../data/source_inventory/cv_quality_audit.json) records per-hand observation coverage, longest missing runs, wrist disagreement with the body detector and wrist-relative fingertip jumps. These are triage signals, not labelled keypoint accuracy measurements.

The audit found **347 clips below 80% left-hand coverage and 342 below 80% right-hand coverage**; 97 have no left-hand detections. Some hands are outside the video or intentionally inactive. Large normalized hand/body wrist disagreements occurred in 92 left-hand and 51 right-hand frames. Such disagreement can come from either detector and needs source inspection; it does not prove which detector is wrong. No blanket “accurate CV” or linguistic approval is justified by the old coverage gate.

## Executed improvements

Missing hands now receive an additional static-image detection pass on square, enlarged wrist regions. Square crops preserve aspect ratio, are bounded to the image, and map x/y/z back to the original image coordinate system. A separate detector prevents crop coordinates from contaminating the full-frame tracker. Pose wrist visibility, one-to-one anatomical assignment, duplicate-candidate suppression and a tighter wrist-distance gate limit ambiguous recovery. Raw missing points remain missing if no detection is accepted. No temporal interpolation was changed.

Extraction metadata now records detector source, body/hand wrist disagreement and `review_required` flags. ROI recoveries are always marked for review; their mere presence is not treated as accurate or expert-approved annotation. Capture method is versioned `roi-v3`. Existing source captures and the active library are preserved; new candidate extractions remain local for review, not silently installed as approved replacements.

Re-extracted **37 Dicta-Sign clips** from the 14 requested phrase scenarios, plus the existing six-clip regression set where needed. Compared exactly matching raw timestamps: 1,974 → 1,983 observed hands across 2,490 hand-frame opportunities. **10 observations recovered, one lost**, a modest net gain of nine. The per-hand results retain regressions. Commons clips without equivalent raw Dicta-Sign captures were not included in this paired comparison.

Actual source-frame overlays were generated for recovered observations and visually inspected, including edge-on hands near the face and blurred, overlapping hands. Their broad placement follows visible hands, but several blurred fingertip positions remain uncertain. Visual inspection of ten candidates cannot establish dataset-wide keypoint precision or reliable depth. Monocular z remains approximate; no independent world-hand coordinates are claimed.

## Verification and evidence

All **39 backend tests** passed. New tests verify square crop boundaries and exact original-pixel coordinate restoration without mutation of detector output. The real extraction worker ran on all paired clips; these are actual MediaPipe outputs, not mocked detector results.

Local source-derived evidence is ignored under `artifacts/cv-review/`: `overlays.html`, `overlays.png`, individual source-frame overlays, candidate raw captures and motion files, logs and `recovery.json`. The audit summary is committed; video, images and raw keypoints are not redistributed.

```bash
python3 scripts/audit_capture_cv.py --output artifacts/cv-review/corpus-before.json
.venv/bin/python -m pytest backend/tests/test_hand_tracking.py -q
```

Remaining limits: no independent manually annotated keypoint reference set; no measured pixel error against such labels; no reliable reconstruction of fully occluded fingers; incomplete hands in low-resolution source footage. Improved observation coverage should not be confused with validated handshape accuracy. Further gains must be assessed against source frames and independent keypoint labels rather than avatar appearance alone.
