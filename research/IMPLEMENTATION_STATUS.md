> Current evidence update: the lead has now read the supplied seven-page `research_paper.pdf`. Earlier PDF-unavailable / not-read notes below are historical. See [verified review](PAPER_B_VERIFIED_REVIEW.md).

# Implementation status — completed 2026-11-15

Branch `feat/research-training`; sole engineer codex2. All project writes in training/, research/, data/catalog/, data/text_to_gloss/. Initial measured milestone committed as `8405a87`; expanded milestone `078cafb` was merged by lead; this follow-up records the requested adaptation investigation. No weights or environments committed. No motion/audio/UI/backend/frontend changes.

Minimal, expanded and longer validation-selected training completed. None of the resulting models is a useful translator.

| Run | Train / val / test | Recording groups | Training / total seconds | Peak allocated / reserved MiB | Test exact match | Test WER | Test BLEU |
|---|---|---|---|---|---|---|---|
| T5-small LoRA CPU smoke,2 epochs |39/28/39|1/1/2|8.2657 /27.0761|CPU: not applicable|0%|207.2816%|0|
| mT5-small LoRA GPU,8 epochs |61/28/39|3/1/2|28.5574 /33.2956|1904.10 /2196|0%|100%|0|
| mT5 rank-16 attention/FF LoRA, selected epoch 11; stopped epoch 21 |61/28/39|3/1/2|113.9165 /246.7427|1961.35 /2250|0%|517.9612%|0.042209|

WER can exceed100% due to insertions. BLEU uses0–100 scale. T5 retrieval baseline testWER111.1650%, lexical196.6019%; expanded retrieval120.8738%, lexical193.2039%. mT5 emits mainly `<extra_id_0>` or `<extra_id_0>.`; its100%WER reflects near-total failure, not a useful translation improvement. No test-driven tuning was performed. Per-epoch losses and EVERY validation/test prediction from all three completed training/evaluation runs are saved in research/runs/. First mT5 attempt was stopped before evaluation to use native slow SentencePiece; explicit aborted-run record retained.

Data:108 original UD STK segments →106 derived free-translation research pairs. Added22 new upstream SUD utterances, excluding five overlapping documents. v002 total128 pairs. ZERO local linguistic approvals; strict approved gate remains unchanged. Source manual annotation != local linguistic approval. Exact manifest hashes, raw annotation lineage, source revisions and quality flags retained. v001 validation/test bytes preserved in v002. All split/near-duplicate/schema/provenance tests pass. Cross-corpus signer identity unknown; recording-disjoint, not independently verified signer-disjoint. Tiny single-family corpus: engineering proof only.

Catalog:11 audited sources,238 raw annotation records (108 UD+130 SUD, including overlap retained as raw provenance),128 derived research pairs,447 Commons lexical concepts/video references,3 completed training/evaluation runs plus one train-only diagnostic. Separate reviews table has no invented human approvals. Sources with rights/access blocks are metadata-only. 447 lexical videos are NOT sentence-training examples. Lead-provided Commons inventory consumed locally; no requery or duplicate motion extraction. Bonjour metadata includes exact licensed source/download URL and credit; lead reports successful real capture/render.

Hardware: RTX4060Laptop8188MiB, torch2.7.1+cu128, BF16 supported. Initial CUDA download timeout resolved; isolated Python3.11training/.venv-train and CPUtraining/.venv-smoke. Models downloaded at exact HF revisions. requirements lock files retained.

Inference CLI implemented and actually executed on CPU and CUDA: `training/infer.py --text ... --model ... --device cpu|cuda`. JSON includes candidate gloss tokens, exact base revision/run/checkpointSHA256/datasetSHA256, pending review, false translation validation/publication flags. Checkpoint hashes are verified on load. Absolute base paths in local adapter configs are documented for lead integration. Existing controlled lookup should remain the product behavior; these checkpoints are failed research candidates.

Read full767-line research brief and531-line original plan. Paper B PDF absent and official access blocked403/401; NOT read. Crossref metadata verified; supplied brief's architecture/counts/scores remain secondary claims. Dicta-Sign lexicon permissions unresolved. Rosetta/Dicta-Sign-v2/MEDIAPI impose institutional download and ORTOLANG-only redistribution conditions. No content downloaded under those terms, no fake importer, no AZee-to-gloss flattening. Additional Matignon/PGloss/parlr/Swiss-resource boundaries recorded in SOURCE_AUDIT.md and sources.csv.

Not completed or claimed: benchmark-quality alignment, qualified local linguistic review, useful arbitrary-French translation, intelligibility study, SOTA, Paper B reproduction, t5-base comparison (optional, not run), approved model-to-motion mapping. Lead owns actual motion/audio/UI, application integration and publication review.


Adaptation follow-up: all 61 training targets round-trip exactly with zero unknown/sentinel tokens. French inputs also have zero unknowns; padding/EOS/decoder shift and model vocabulary bounds are correct. Four TRAIN-only examples reduced teacher-forced loss 36.1055→0.1103 after 200 updates, reaching 2/4 exact greedy outputs. This confirms substantial optimization but not complete memorization.

One longer mT5 run was configured in advance in MT5_LONG_RUN_PLAN.json: rank 16 q/k/v/o/wi_0/wi_1/wo, alpha 32, lr 0.001, BF16, batch 1 / accumulation 4, 5% warmup, up to 100 epochs, validation-loss patience 10. Counted 2,555,904 trainable parameters BEFORE adapter selection/freezing. Fixed the last partial accumulation group's divisor and logged token-weighted validation loss plus the historical mean-per-example loss. No hyperparameter sweep or test-based changes. Validation selected epoch 11 (token loss 3.2943 / mean-per-example 3.2148), then stopped after ten non-improving epochs at 21. This is substantially below the short run's mean-per-example 13.9221, under a different adapter/training configuration; it is not a controlled epoch-only ablation.

Final test was decoded once after selection. Sentinel output disappeared (0/39), but repeated/malformed gloss strings replaced it. Test model WER 517.96% and BLEU 0.0422 are worse than retrieval WER 120.87% / BLEU 0.3010. No useful generalization or translation accuracy is claimed. Same frozen test had been used in historical experiments, so this is a reused engineering test, not fresh confirmation. All earlier run files and all v002 data hashes are verified unchanged. Twelve tests pass. No other panes or schedules were inspected, controlled or created during this follow-up.
