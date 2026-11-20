# Why all Colab exact-match scores are zero

The supplied `/home/jmal/Downloads/research_paper.pdf` was reread in full, including methodology §§3.2–3.4 and results Table 3. The existing trials are **not a reproduction** of this paper. Moving between mT5, T5-base and FLAN-T5-base did not repair the experimental prerequisites. Data coverage and decoded outputs should have been audited before those additional model runs.

## Verified differences from the paper

| Item | Supplied paper | Executed Colab trials |
|---|---|---|
| Target language | France LSF | Belgian LSFB |
| Paired corpus | 2,500 manually verified French/gloss sentence pairs | 300 automatic overlap candidates, 252 after length filtering, **111 actually used for training** |
| Lexicon | 319 isolated signs; tokenizer extended with 319 glosses | No task-specific gloss tokens added; 407 distinct whitespace-delimited target forms in training and 815 across splits |
| Supervision | Manually verified sentence pairs | Unreviewed timestamp overlap, flattened `both_hands` annotation events |
| Optimizer | AdamW, LR 2e-5, batch 4, weight decay .01 | Adafactor, LR 5e-5, per-device batch 1 with accumulation 4 |
| Schedule | 20 epochs | mT5: 100 steps (~3.58 epochs); T5/FLAN: 300 steps (~10.72 epochs) |
| Canonical correction | 47 expert rules, RapidFuzz threshold 85%, rule reordering | No matching expert dictionary, fuzzy correction or reordering in the evaluated models |
| Generalization protocol | Paper describes a single-signer corpus; no equivalent signer-disjoint split is detailed | Train: four signers; validation: S022; test: unseen signer S041 |

The paper's 2,500 figure describes its corpus; it does not sufficiently specify the exact train/validation/test counts. Its results cannot be treated as the expected score on our different language, raw labels and held-out signer.

## Measured coverage problems

The 111 training pairs contain only **884 whitespace-delimited target tokens** but **407 distinct forms**; **257 forms occur once**. The validation set has 12 examples / 75 target tokens: **45/75 (60%) are absent from training**, and 11/12 sentences contain an unseen form. The test set has 129 examples / 1,156 target tokens: **596/1,156 (51.56%) are absent from training**, and **119/129 sentences contain an unseen form**.

Subword models can theoretically compose unseen gloss strings, so unseen forms are not a mathematical impossibility for these models. The measurements show extremely thin task-specific supervision and substantial target-vocabulary shift. They also show why adding more model parameters is not evidence that the task has been learned.

The group-disjoint split prevents signer leakage, but the small candidate cap and unequal group sizes leave only 111 training examples and 129 test examples. It is a demanding internal generalization check, not a matched reproduction of the paper. Raw targets include pointing labels, name signs, fingerspelling, descriptive/gesture and variant notation rather than a reviewed closed set of avatar-library gloss IDs. Do not automatically strip morphology or map Belgian annotations to France-LSF signs to inflate a metric.

Temporal overlap is weak supervision. A short gloss reference beside a longer French translation is not, by itself, proof of bad annotation: sign languages differ in segmentation and context. The actual message alignment remains unverified and requires source-context / competent linguistic review.

## The zero is independently reproduced

`audit_data_and_predictions.py` recomputes Levenshtein edits with RapidFuzz, independently of the training script's distance implementation. It reproduces every saved WER and the **0/129** exact scores. Unicode/case/whitespace normalization also leaves **0/129** for each model. All runs use identical French/reference pairs.

| Run | Reference tokens | Predicted tokens | Substitutions | Deletions | Insertions | WER |
|---|---:|---:|---:|---:|---:|---:|
| mT5 100 steps | 1,156 | 201 | 160 | 996 | 41 | 103.55% |
| T5-base 300 steps | 1,156 | 3,357 | 1,033 | 61 | 2,262 | 290.31% |
| FLAN-T5-base 300 steps | 1,156 | 3,005 | 1,022 | 83 | 1,932 | 262.72% |

Every mT5 test output contains a pretraining marker such as `<extra_id_0>`. T5 and FLAN repeat short fragments, producing many insertion errors. A T5 example repeats `pt:pro1 texte facile` instead of recovering the annotated sequence. These are observable generation failures; a lower teacher-forced loss did not establish successful free generation.

The tokenizer inspection found **zero added task tokens** in all three final tokenizers. All 111 training, 12 validation and 129 test targets end in EOS. mT5 roundtrips all 129 reference strings exactly. T5/FLAN roundtrip 128/129; one annotation contains a backtick that becomes unknown. That isolated representation defect cannot explain all 129 failures. Absence of the paper's tokenizer extension is a recipe difference and a learning concern, not evidence that all labels are unencodable.

## Correct interpretation of the paper

Table 3 reports **21.9% raw T5 exact match**, improving to **30.0% after correction**. Its **100% animability** claim concerns canonical gloss lookup, not 100% translation accuracy. The GAN acts after gloss prediction to generate motion transitions; adding it cannot fix text-to-gloss exact match. Emotion or avatar rendering also cannot repair these text predictions.

The paper does not fully disclose the split construction, exact 47-rule dictionary or all pair-building details. Those omissions limit exact reproduction; they do not excuse our failed models or establish that the authors' reported results are false.

## Next experimental gate

1. Pick and preserve the target language. For the current France-LSF avatar, obtain aligned French/France-LSF sentence targets and an approved mapping to the available canonical sign lexicon. Retain the LSFB experiment as separately labeled research.
2. Audit pair alignment and target conventions. Preserve raw annotations and morphology; define reviewed canonicalization rather than applying blanket string cleanup.
3. Add task gloss tokens for an independently defined canonical vocabulary and resize the model embeddings. Do not build that vocabulary from held-out target answers or claim the authors' 47 expert rules have been reconstructed.
4. Require a learning sanity check: overfit 8–16 reviewed real pairs and inspect generated outputs / EOS stopping. This is a memorization diagnostic, not validation or a publishable translation score. Passing file/checkpoint tests alone was insufficient.
5. Train on a sufficiently covered, frozen paired corpus using a documented recipe; the paper's T5-base/AdamW configuration is a reproducible starting point, not a guarantee. Track generated validation WER, repetition and target coverage alongside loss. Preserve a new untouched final test after this exploratory reused-test investigation.
6. Add and evaluate justified canonical correction separately. Report raw versus corrected translation and separate lookup coverage from semantic accuracy.

No additional model training was launched during this audit, and previous checkpoints/scores were not overwritten. Machine-readable evidence is [COLAB_ZERO_AUDIT.json](COLAB_ZERO_AUDIT.json). Reproduce the local data/scoring audit with:

```bash
.worktrees/wt-codex2/training/.venv-train/bin/python training/colab/audit_data_and_predictions.py
```

Primary sources: the supplied paper (PDF pp. 3–5, §§3.2–3.4 / Table 3); the [LSFB-CONT card](https://huggingface.co/datasets/ppoitier/lsfb-cont); actual frozen splits, final tokenizers and saved predictions from these runs.
