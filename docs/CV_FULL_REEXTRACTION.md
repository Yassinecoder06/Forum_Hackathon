# Full cached-video CV v4 re-extraction

Requested operation: re-extract observed keypoints from every locally cached real-source video, rather than train model weights. `scripts/reextract_all_cv.py` inventories source provenance and registered non-synthetic Commons captures, verifies available recorded source checksums, adds the attributed curated Bonjour example and groups exact duplicates by SHA256. Nine unprovenanced uploads were excluded.

The inventory contains **1,472 source entries, 1,355 unique video files**. All aliases remain recorded so deduplication does not discard concept/source provenance. The run uses four local CPU workers, the existing isolated MediaPipe environment and cached videos. No new video retrieval or model training is involved.

Candidates are written under ignored `.runtime/cv-v4-all/captures/<source-sha256>/`. Each contains actual raw landmarks, native motion, execution log and a comparison result. Originals, active library assets, source licences and pending review statuses are preserved. A worker-code fingerprint identifies the extractor; completed matching results can be resumed. Each failed extraction is retried once, then recorded as a failure rather than substituted.

Where a previous raw capture exists, the evaluator compares exact common timestamps and checks individual lost hand/body observations. At least 95% of old timestamps must be comparable. Passing yields `no_observation_loss`, an engineering triage flag—not keypoint accuracy or linguistic approval. Missing baselines and regressions are explicitly held for review. Commons captures with only retargeted native motion cannot supply a raw keypoint accuracy reference and are not treated as validated baselines.

Progress is atomically checkpointed to `.runtime/cv-v4-all/status.json`; inventory and final per-file results are alongside it.

## Completed run

All **1,355 unique videos completed**, covering **1,472 source entries**, with **zero processing failures**. The candidates contain **61,817 actual sampled frames**. Independent raw-file checks passed for every completed video/frame: monotonic timestamps, expected body/face/hand landmark counts, finite coordinates, observation-flag consistency and absent independent world-hand fields. The saved integrity report contains **zero structural errors**. These checks do not establish anatomical or linguistic accuracy.

Of the 1,472 source-entry comparisons, **886 have no observation loss**, **139 require regression/comparability review**, and **447 have no older raw keypoint baseline**. The latter entries are not processing failures. Common-timestamp comparisons recorded 570 recovered and 315 lost hand observations, weighted by source aliases; exact duplicate recordings may contribute to more than one source comparison. These figures describe detector availability, not correctness of the points.

Among the 1,355 unique candidates, 530 have left-hand coverage below 80% and 549 have right-hand coverage below 80%. This population includes Commons videos and differs from the earlier Dicta-only audit; the counts cannot be compared directly. Inactive/out-of-frame hands, occlusions and detector misses need source review.

The [complete textual report](../data/source_inventory/cv_v4_all_report.json) preserves per-source checksums, aliases, gates and failures. Runtime was Python 3.11.14, MediaPipe 0.10.21 and OpenCV 4.11.0.86; one worker-code fingerprint identifies all outputs. The recorded elapsed seconds refer to the final resumed invocation. The initial tool-bound runner stopped after 128 files; a detached runner resumed checkpoints and completed the remaining work without lost outputs.

No active motion assets were replaced, no review status was promoted, and no model weights were trained. Raw landmarks and source-derived motions remain ignored local research artifacts under `.runtime/cv-v4-all/`. The no-observation-loss set is available for source-overlay review; regressions and absent-baseline cases remain explicitly held.

```bash
python3 scripts/reextract_all_cv.py --workers 4
# Same invocation resumes completed matching work; originals are never overwritten.
python3 scripts/verify_cv_batch.py
python3 scripts/summarize_cv_batch.py
```

These data remain local research previews. Detection coverage and repeated-scale agreement cannot establish true fingertip precision under occlusion; source overlays and independent labelled references are still needed for accuracy claims.
