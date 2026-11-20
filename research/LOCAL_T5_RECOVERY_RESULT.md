# Local T5 recovery experiment — 16 November 2026

Completed the requested technical recovery plan and monitored both runs to completion. These are exploratory, unreviewed lexical LSFB experiments. No official test data was opened and no application deployment was changed.

## Repairs and checks

- Added `training/lsfb_cont/scoring.py`; malformed predictions retain recoverable operations and invalid spans. A separate raw-token WER is reported. New `train.py` runs use this scorer; the old scorer remains explicitly named `legacy_score` for historical reproduction.
- Rescoring the old 679 predictions yields operation WER 219.44%, raw-token WER 143.08%, exact 2.95%, invalid 23.86%. These are different metrics, not new model performance.
- Removed 327 exact-time duplicate lexical operations in the diagnostic preparation. Remaining ambiguous overlapping same-lemma rows were excluded, not guessed. Original candidates and hand-tier snapshots remain intact.
- Added regression coverage for quoted separators, fingerspelling, malformed repetitions, empty output and simultaneous versus sequential repeated events. **15 tests passed**.
- Initial FP16 diagnostic failed with nonfinite loss; restart in BF16 completed. This does not establish the cause of the older completed model's failure.

## Small learning gate

Pinned pretrained `google-t5/t5-base`, rank-16 q/v LoRA, BF16, seed42, batch8, learning rate5e-4. Compact target is a JSON list of lexical author lemmas; source is current French only. No thousands of atomic tokens were added.

At step200, 29/32 training examples were exact (90.625%), with zero malformed outputs and zero generation-cap hits. Validation was 5/32 (15.625%). Gate passed in about20 seconds. This checks learnability, not translation validity.

Artifacts: `.runtime/lsfb-recovery-v2/RESULT.json`, `DATA.json`, `predictions.json`, `adapter/`, and `RUN_MANIFEST.json`.

## Bounded exploratory run

Built a top128 operation vocabulary from training candidates only, requiring at least five occurrences. Retained short targets (1–6 lexical events), current French of at most25 words, no fingerspelling, ambiguous overlapping same-lemma events, boundary exclusions, ellipsis flags or bracketed notes. Nonlexical meaning is still omitted: this remains a lexical projection.

Resulting pool: **361 training pairs and51 validation pairs**, with no shared conversations. Zero human approvals. The51 validation rows are only7.5% of the previous679; the new task and scores must not be presented as directly comparable to the previous full validation benchmark. Gate and bounded validation overlap by design; all these are development measurements, never fresh test scores.

Warm-started from the gate adapter. BF16, q/v LoRA rank16, batch8, LR1e-4, 1200 optimizer steps. Checkpoints scored every100 steps; best selection requires zero invalid validation outputs, then lowest operation WER. Training duration approximately250 seconds. GPU process memory observed around1.2GB, no OOM.

| Same51-example validation split | Exact match | Operation WER | Invalid |
|---|---:|---:|---:|
| Training-only word TF-IDF retrieval | 15.69% (8/51) | 112.63% | 0% |
| Gate adapter before broader training | 13.73% (7/51) | 100.00% | 0% |
| Selected checkpoint, step1100 | **27.45% (14/51)** | **89.47%** | **0%** |
| Final checkpoint, step1200 | 25.49% (13/51) | 137.89% | 1.96% |

Selected checkpoint has zero validation cap hits. Its full training-set exact match is only31.86%, so learnability on32 examples has not solved broader fit. Validation contains only95 reference operations; this is a small, short-utterance evaluation. Six training inputs map to multiple different targets (theoretical deterministic exact ceiling97.51%); ambiguity remains. No claim of semantic intelligibility is made.

**Reload verification:** reloaded the saved best adapter in a fresh process and reproduced every one of the51 saved validation predictions exactly.

Selected artifact: `.runtime/lsfb-bounded-v1/best_adapter/`; use the pinned base revision from `RUN_MANIFEST.json`. The `adapter/` directory is the final step1200 adapter and must not be confused with `best_adapter/`.

Evidence: `.runtime/lsfb-bounded-v1/BEST.json`, `RESULT.json`, `history.json`, `best_predictions.json`, `DATA.json`, `RETRIEVAL_BASELINE.json`, `RELOAD_VERIFICATION.json`.

## Next stage

Technical recovery succeeded: T5 can memorize this representation and a bounded model beats the word-retrieval baseline. Generalization remains weak, with unstable generation on some checkpoints. Keep LSFB language IDs distinct from the application's France-LSF motions. Source-video/pose review and a language-consistent bank of complete meaning-bearing phrases remain necessary before a reliable translator or broad-data training claim. No invented human approvals or automatic promotion of these rows was performed.

Do not scale this configuration to all conversational rows on the strength of14 exact short examples. Review the saved failures, curate self-contained pairs and protected evaluation, and compare semantic phrase retrieval with the bounded planner. No state-of-the-art claim is supported.
