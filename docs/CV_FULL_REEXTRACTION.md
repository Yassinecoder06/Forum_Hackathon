# Full cached-video CV v4 re-extraction

Requested operation: re-extract observed keypoints from every locally cached real-source video, rather than train model weights. `scripts/reextract_all_cv.py` inventories source provenance and registered non-synthetic Commons captures, verifies available recorded source checksums, adds the attributed curated Bonjour example and groups exact duplicates by SHA256. Nine unprovenanced uploads were excluded.

The inventory contains **1,472 source entries, 1,355 unique video files**. All aliases remain recorded so deduplication does not discard concept/source provenance. The run uses four local CPU workers, the existing isolated MediaPipe environment and cached videos. No new video retrieval or model training is involved.

Candidates are written under ignored `.runtime/cv-v4-all/captures/<source-sha256>/`. Each contains actual raw landmarks, native motion, execution log and a comparison result. Originals, active library assets, source licences and pending review statuses are preserved. A worker-code fingerprint identifies the extractor; completed matching results can be resumed. Each failed extraction is retried once, then recorded as a failure rather than substituted.

Where a previous raw capture exists, the evaluator compares exact common timestamps and checks individual lost hand/body observations. At least 95% of old timestamps must be comparable. Passing yields `no_observation_loss`, an engineering triage flag—not keypoint accuracy or linguistic approval. Missing baselines and regressions are explicitly held for review. Commons captures with only retargeted native motion cannot supply a raw keypoint accuracy reference and are not treated as validated baselines.

Progress is atomically checkpointed to `.runtime/cv-v4-all/status.json`; inventory and final per-file results are alongside it. The final measured summary will be recorded after completion.

```bash
python3 scripts/reextract_all_cv.py --workers 4
# Same invocation resumes completed matching work; originals are never overwritten.
```

These data remain local research previews. Detection coverage and repeated-scale agreement cannot establish true fingertip precision under occlusion; source overlays and independent labelled references are still needed for accuracy claims.
