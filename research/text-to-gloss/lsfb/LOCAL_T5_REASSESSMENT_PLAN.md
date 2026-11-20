# Local T5 and LSFB-CONT reassessment

**Status: assessment and run plan only. No T5 weights were loaded and no training steps were run.**

## Readiness decision

| Area | Assessment | Decision |
|---|---|---|
| GPU | RTX 4060 Laptop, 8 GB VRAM; CUDA is available | Suitable for a small T5-base LoRA run |
| System memory | 16 GB RAM; roughly 7.5 GB available during the check; 4 GB swap | Keep batches small and use gradient checkpointing; avoid full fine-tuning initially |
| Storage | About 47 GB free at assessment time | Enough for one base checkpoint, adapter, and a few checkpoints; monitor before a full run |
| T5-base checkpoint | Pinned `google-t5/t5-base`, 222.9M parameters, 891,646,390-byte weights | Model is suitable; first download from this PC's Hub/CDN route was very slow and was stopped before loading |
| Prepared data | 3,866 exploratory train rows, 679 exploratory validation rows | Technically prepared, not linguistically approved |
| Translation-quality training | Zero reviewed train rows and zero reviewed validation rows | **Not ready for a quality claim or broad training run** |

The PC is not the main limitation. The key limits are the quality and amount of reviewed supervision, and a slow model download route. The cancelled download did not use GPU memory. The temporary training environment and draft gate were removed after the request changed to assessment-only; recreate the pinned environment when training is authorized.

## Proposed model and task

Keep the new task separate from the earlier failed atomic-gloss run:

1. **Input:** the current French translation plus up to four completed French dialogue turns from the preceding 60 seconds. Turn roles are relative (`Moi` and `Autre`). Do not include glosses, future turns, or unfinished dialogue.
2. **Target:** a readable sequence of lexical operations, for example `SIGN "bonjour"` or `FINGERSPELL "fk" "fs:fk(ok)"`. Keep the author-provided lemma and the original fingerspelling hint. Use the pinned T5 tokenizer as-is; add no thousands of output tokens.
3. **Model:** pinned `google-t5/t5-base` with LoRA adapters on attention `q` and `v` projections for the first local experiment. Use FP16, gradient checkpointing, batch size 1, and gradient accumulation. This tests the context-conditioned lexical projection and is much lighter than full-parameter AdamW training.
4. **Output checks:** parse generated operations; report invalid-plan rate, operation-sequence WER, exact plan match, predicted/reference operation counts, and repetition. Do not evaluate only token loss.
5. **Scope boundary:** this predicts a lexical projection for Belgian LSFB. It does not produce full LSFB grammar, hand timing, nonmanual features, a France-LSF translation, or an avatar-motion sequence. The avatar and motion lookup need their own language-consistent mapping and evaluation.

The prior atomic-token T5 run is a negative baseline, not an initialization checkpoint: it added 6,743 output tokens, failed to fit its training examples, and produced severe insertion/repetition collapse. The local 100-step mT5 POC is also not a clean pretrained starting point; its held-out exact match was 0%. Start from the pinned pretrained checkpoint for this experiment.

## Data state and interpretation

The prepared package is `.runtime/lsfb-cont-task-v3/`, built from local `/home/jmal/Dataset/lsfb-cont` at source revision `64b0c112f59d3710367fb736ef1808f37ff15e28`.

- 5,354 candidate French segments were found in seven verified annotation shards.
- 3,866 rows are technically eligible for exploratory training and 679 for validation.
- 299 training rows were quarantined because their conversation occurs in validation; 809 rows have one or more technical exclusions.
- 43 review examples with source-video excerpts were prepared; **0 train and 0 validation examples have linguistic approval**.
- These are weakly time-aligned conversational translations. Temporal overlap and tokenizer checks do not certify that a French turn is the right translation target or has enough discourse context.
- The lexical projection intentionally leaves pointing, depictions, buoys, palm-up events, simultaneous hand tiers, and timing outside the model target. Those remain in the source sidecars; this output is not a full signing plan.
- The new preparation excluded official test shard 0. Keep it closed until model choices and validation criteria are frozen. The older experiment's test results have already been inspected and must not be reused to tune this approach.

Dataset preparation evidence and limitations are recorded in [`LSFB_CONT_PREPARATION_RESULT.md`](LSFB_CONT_PREPARATION_RESULT.md), [`LSFB_FULL_DATASET_ASSESSMENT.md`](LSFB_FULL_DATASET_ASSESSMENT.md), and [`LSFB_T5_FAILURE_DIAGNOSIS.json`](LSFB_T5_FAILURE_DIAGNOSIS.json). The data package README documents how to import provenance-bound human review decisions.

## Run sequence when authorized

### 1. Review before scaling

Have a reviewer fluent in French and LSFB approve a stratified initial set (target 200–500 examples) for meaning, source alignment, dialogue context, and target representation. Include short and long segments, fingerspelling, simultaneous events, and boundary cases. Keep decisions tied to the row hashes using `research/code/training/lsfb_cont/release.py`. Until both train and validation rows are accepted, any training result is an engineering diagnostic only.

Use only accepted rows for the quality experiment. If only a small subset is reviewed, state the limited domain and vocabulary coverage; do not silently promote all 3,866 exploratory rows to ground truth.

### 2. Freeze the experiment inputs

- Verify the prepared data manifest and all row hashes.
- Preserve the conversation-disjoint train/validation split and assert there is no conversation overlap.
- Freeze the model ID, revision, tokenizer, review release, seed, token limits, generation settings, and evaluation code in a run manifest.
- Keep the official test shard closed. Do not select a checkpoint from test results.
- Download T5-base once, verify its pinned revision and file digest, and retain it in the local HF cache. The previous attempt was interrupted because the current Hub/CDN route delivered weights slowly; it did not indicate GPU or data failure.

### 3. Small local learning gate

Implement a reproducible LoRA trainer for the reviewed data, then fit a tiny reviewed subset before scaling:

- 32 reviewed train pairs and 32 conversation-disjoint reviewed validation pairs;
- T5-base LoRA, rank 8, `q`/`v` projections, FP16, gradient checkpointing, batch 1, accumulation 4;
- about 100–300 optimizer steps, recording loss and peak VRAM;
- compare base-model and post-training greedy outputs on the same rows;
- require finite loss, clear training-set fit (target at least 80% exact match on the 32-row memorization sample), syntactically valid plans, and no OOM before proceeding.

This gate checks that the implementation can learn the representation. It is not a translation benchmark. If it cannot fit the tiny training sample, debug the tokenizer, labels, loss masking, and generation before scaling.

### 4. Reviewed-data training and model selection

Train only on the reviewed release, with LoRA first. Select checkpoints using generated validation operation-WER plus exact match and plan validity. Compare against training-only retrieval baselines. Report validation performance by target length, fingerspelling, context use, and operation frequency; report repetition and predicted/reference length ratios. Include a no-context ablation to measure whether dialogue history helps. Keep all source rows and review decisions immutable.

Scale to full-parameter fine-tuning only if LoRA demonstrably fits the reviewed task and validation improves. With 8 GB VRAM, full T5-base AdamW is a higher OOM risk and has no current evidence of being necessary.

### 5. Final evaluation and downstream handoff

Freeze the selected model and decoding protocol before opening any still-unseen official test data. Report test metrics once, with the exact test provenance and any overlap with previously inspected evaluation sets. Evaluate text-to-lexical-plan quality separately from motion coverage, signing grammar, facial expression, and avatar animation. Do not claim a working translator or state-of-the-art system from a tiny memorization gate.

## Go/no-go summary

- **Go:** environment setup, checkpoint staging, reviewed-data gate, and a small LoRA experiment on this PC.
- **No-go until review:** broad training on all exploratory rows as if they were verified French-to-LSFB pairs, or making translation-quality claims from them.
- **No-go for the old path:** resume the 6,743-token atomic-gloss experiment or use its failed checkpoint as the new model's starting point.
- **First action after “go”:** create the pinned Python 3.11/CUDA environment, download and checksum T5-base once, then implement/run the reviewed 32/32 learning gate. Do not open the official test shard during that work.
