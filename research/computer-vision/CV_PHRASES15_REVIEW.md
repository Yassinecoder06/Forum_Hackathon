# CV review of the 15 exact dictionary sequences

All requested dictionary concepts resolve exactly, with no missing/ambiguous entries: **35 unique source clips** across the 15 sequences. Phrase 13 uses **TRAVAIL / Dicta-Sign cs1032**, as requested, rather than the earlier TRAVAILLER clip. Phrase 15 includes **ABANDONNER / cs726**.

The local gallery at **http://127.0.0.1:8768/** provides 15 actual source-video diagnostic sequences. Each shows older raw CV on the left and the full-corpus CV v4 observations on the right. Green denotes the left hand, cyan the right; yellow body anchors and selected pink face anchors are also shown. Absent detections remain unmarked. Where the older extractor did not sample a timestamp, the left panel explicitly says so. There is no avatar retargeting, inferred hand filling or sign interpolation in these diagnostic videos.

Display videos use 30 fps, repeating source diagnostic images as necessary for playback; numerical counts always use native detector observations, not presentation duplicates. The videos concatenate dictionary citations for review and are not validated signed sentences. Source URLs and hashes are retained in the [per-phrase report](../../data/source_inventory/cv_phrases15_review.json) and original corpus inventories; original attribution and pending-rights status remain in force.

## Findings

Of the 35 sources, **30 pass the no-observation-loss gate**, **3 require regression/comparability review** (PAIN, LIVRE, TRAVAIL), and **2 have no older raw baseline** (CHAT, CHERCHER). Passing the gate is not a measurement of keypoint accuracy.

The following diagnostic flags identify missing hand landmarks while the body detector reports a visible wrist raised above its elbow. The heuristic uses body visibility ≥0.75 and an image-y separation >0.01. It is a review aid: the wrist estimate can be wrong, and fingers can still be occluded despite a visible wrist. No independent point labels were created.

| Phrase | Raised-visible-wrist absence flags | Other review context |
|---|---|---|
| 1 | None | BOIRE's left hand is outside the crop in inspected signing frames; active right hand is tracked there. |
| 2 | MANGER: right 2 frames; PAIN: left 10 | PAIN is the main follow-up priority; overlapping hands in inspected source frames. |
| 3 | FRERE: right 2 | AIMER's inactive left hand is outside the crop in inspected frames. |
| 4 | DEMAIN: right 1 | Inspect brief loss; absence flag alone does not prove incorrect signing capture. |
| 5 | None | No absence heuristic alert; not an accuracy approval. |
| 6 | None | No absence heuristic alert; not an accuracy approval. |
| 7 | LIVRE: right 1 | Held for source comparison under the regression gate. |
| 8 | CHAT: left 4/right 1; DORMIR: right 1 | CHAT has no older raw baseline. DORMIR's missing inactive left hand is distinct from an active-hand miss. |
| 9 | None | No absence heuristic alert; not an accuracy approval. |
| 10 | None | Exact LANGUE and SIGNE source clips inspected; lexical sense and LSF grammar are not validated by CV. |
| 11 | CHERCHER: left 1 | No older raw baseline; current observations remain review candidates. |
| 12 | DEMAIN: right 1 | Same source as phrase 4. |
| 13 | TRAVAIL: right 1; AUJOURD HUI: left 1/right 1 | TRAVAIL is a different source from the previous TRAVAILLER example and requires regression/comparability review. |
| 14 | None | CAFÉ has incomplete whole-clip tracking, without this raised-wrist absence flag. |
| 15 | None | ABANDONNER has 53% whole-clip coverage per hand; both are tracked in the inspected middle frame. |

Whole-clip coverage includes initial/rest/end frames and should not automatically be interpreted as accuracy during the gesture. For example, BOIRE/AIMER have no left-hand observations, but the source samples show that hand below/outside the crop. Conversely, visible overlapping hands in PAIN need closer review even though overall coverage is fairly high. Finger placement under motion blur and occlusion remains uncertain.

Quantitative diagnostics cover every selected clip. Visual inspection sampled BOIRE, AIMER, TRAVAIL, ABANDONNER, LANGUE, SIGNE and PAIN; it is not a claim of independent expert review of every frame. Full per-hand coverage, longest missing interval, ROI review counts and source pointers are recorded for all 15 sequences.

## Executed checks

Generated **35 word-level H.264 diagnostics and 15 complete concatenated diagnostics**. ffprobe decoded every video; each phrase's decoded frame count matches the sum of its constituent word videos. All word videos have the same 30 fps display rate and dimensions. The real browser loaded all 15 video elements and successfully played a 960×400 diagnostic. Evidence and the browser screenshot are local under `artifacts/cv-review/phrases15/` and remain ignored.

```bash
.venv-motion/bin/python scripts/review_cv_phrases.py
```

The runner reads the active dictionary only to identify exact source entries, then uses the staged **CV v4 full-corpus outputs** for the review. It does not replace the active avatar assets or alter interpolation.
