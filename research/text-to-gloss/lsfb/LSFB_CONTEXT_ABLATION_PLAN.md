# Matched context ablation

Context arm completed in .runtime/lsfb-narrow-context-run-v1; current-only arm trains in .runtime/lsfb-narrow-current-run-v1. Same145train/23development rows, frozen35label vocabulary, labels, seed84, pinned T5-base, rank16attention/feedforward LoRA, BF16, dropout.05, batch2/accumulation4,25epochs/475steps, warmup/decay LR2e-4, max384source/192target tokens. Original task prefix and current text retained; only preceding dialogue removed. Each arm chooses its own lowest validation-WER checkpoint. Same decoding and scoring.

Primary diagnostic: lexical WER and exact on the same23development rows, with six multi-sign cases reported separately. Compare current-only and context-inclusive retrieval, preserve failed outputs, no vocabulary expansion. Dataset and prior results have been inspected; this is not a fresh heldout test. No qualified local review exists. A matched seed cannot eliminate differences in padding/dropout draws caused by input length. No significance claim based on this tiny sample.

Commands:

```bash
.runtime/t5-local-venv/bin/python -m training.lsfb_cont.train_narrow_context --data .runtime/lsfb-narrow-context-data-v1 --output .runtime/lsfb-narrow-current-run-v1 --epochs 25 --input-mode current
.runtime/t5-local-venv/bin/python -m training.lsfb_cont.compare_narrow
```

Run names must be new to reproduce without overwriting. Both selected adapters require pinned pretrained T5-base; all motion/pose references remain Belgian LSFB. Further progress needs reviewed multi-sign supervision and a genuinely fresh conversation holdout. The experiment cannot itself produce human review.

Review inventory:1,869technically eligible training multi-sign rows staged privately with original lexical events, timing,context and pose references. Only49are entirely within the frozen35labels;all49are already trained. Additional23quarantined train rows use these labels:18have fragment/editorial flags,1uncertain boundary,1simultaneous lexical activity,4French-overlap flags (reasons overlap). No quarantined rows were admitted; shared-validation phrases must stay excluded. Rechecking/repairing source alignment requires qualified review, not automatic approval. Private queues:research/private/lsfb-multisign-review-queue.jsonl and lsfb-quarantined-multisign-review.jsonl.
