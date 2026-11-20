# ACL-inspired Belgian LSFB training run

Session: `signaculture-acl2023-lsfb`, Tesla T4. Runner: `research/code/training/acl2023_lsfb/train_torch.py`. PyTorch adaptation, not an exact Marian reproduction and not T5 fine-tuning.

Architecture: shared 2,000-piece train-only SentencePiece BPE vocabulary, sinusoidal positions, dimension512,8 heads, feedforward2048,1 encoder and2 decoder layers, tied input/output embeddings. Current French→reversible lexical gloss sequence. No France-LSF examples. No auxiliary or synthetic examples in this baseline.

Gate:32 distinct train-only French inputs, no dropout or label smoothing, up to800 updates, >=95% exact training decoding required. Full run reinitializes weights after passing; gate examples are not a validation set.

Full run:2,146train/385development rows; batch32; Adam(.9,.98), maxLR3e-4,200-step warmup then inverse square-root decay; dropout.2, smoothing.2; up to40epochs. Full-development greedy decoding every two epochs, cap96subtokens, lowest lexical WER selects weights. Stop after8 non-improving evaluations. Save gate results, history, predictions, tokenizer and best weights to Drive. Report unknown labels, cap hits, one/multi-sign results, exact, WER, BLEU, chrF, TER. This adaptation differs from Marian in optimizer schedule, decoding, stopping and selection. No claim of exact reproduction.

Checks passed on Colab: causal decoder mask, source padding invariance, scoring empty/unknown outputs without dropping their errors. Reported Torch2.14.1+cu130; sacrebleu2.5.1.

Runtime paths:
- `/content/acl2023-run-v1.log`
- `/content/acl2023-run-v1/`
- `MyDrive/SignaCulture/acl2023-lsfb-v1/run-v1/`

Status: launch submitted; monitoring for process confirmation. No successful training result claimed yet.

## First measured run and selection repair

The32-pair overfit gate passed at50updates with100% exact match (24/24multi-sign cases), zero cap hits and zero unknown labels. This demonstrates train-set learnability, not generalization.

Runv1 completed18epochs but its WER-selected epoch2 checkpoint had224/385 unfinished predictions. Repeated punctuation inside a single gloss was counted as one lexical substitution, making WER-only selection misleading. The checkpoint is retained as failure evidence, not promoted. Its nominal99.12%WER is not a usable translation result.

Runv2 starts afresh with the same architecture, data, seed and regularization. Checkpoint eligibility now requires zero generation-cap hits. The maximum is60epochs, early stopping waits until at least20epochs and12non-improving evaluations, and the final weights are saved separately from selected weights. All evaluations remain in history; rejected generations are never deleted from metrics. Runtime `/content/acl2023-run-v2`, Drive `run-v2/` under the same experiment folder. The identical gate again passed100% at50updates.

## Completed corrected run

Runv2 completed50epochs in294.6seconds and early-stopped. Selected epoch26 after fresh checkpoint reload:

| Measure | Result |
|---|---:|
| Full development exact |14/385 (3.64%)|
| Multi-sign exact |0/328|
| Single-sign exact |14/57 (24.56%)|
| Lexical WER |104.48%|
| BLEU (none tokenizer) |0.2581|
| chrF |15.2151|
| TER |104.2356|
| Generation cap hits |0/385|
| Unknown predicted labels |109/1,656 predicted operations|

The final epoch training sample reached23/32exact (71.88%) and9.74%WER; the validation gap is substantial. This is evidence of poor generalization under this setup, not proof that all Transformer/transfer-learning approaches fail. No auxiliary translation or back-translation experiment has yet been run.

Compared on the same385rows, the prior T5 had240.60%WER and15/385exact; current-French retrieval had113.92%WER and9/385exact. The new checkpoint reduces WER versus those systems but remains worse on WER than an empty-output diagnostic (100%), and it gets no multi-sign sequences exactly right. It is not ready for application integration. Exact-match metrics alone also miss partial correctness, so WER and predictions remain included.

All training reference targets fit within56subtokens including EOS; all validation targets fit within68, below the96generation cap. Reference truncation therefore does not explain this result.9training current-French strings have multiple target sequences, affecting25rows (casefold+strip grouping); the current-only baseline removes dialogue context, which remains an untested ablation here.

The model, tokenizer, configuration, source runner, training log, history, gate and validation predictions were copied to `MyDrive/SignaCulture/acl2023-lsfb-v1/run-v2`.16files were SHA256-verified, with a backup receipt stored alongside them. The selected checkpoint is52,460,173bytes. Both selected and final weights are retained in Drive. Local results archive download is in progress; archive SHA256 `e230a8d38ce4103ad1468d7e22c825c59bd24d9e5cb73ab519236f2c6bdab33e`.
