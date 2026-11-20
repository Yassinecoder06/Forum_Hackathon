# Colab model comparison — 15 November 2026

All runs used actual full-model fine-tuning on a Colab Tesla T4. **Drive is mounted and all three final models, tokenizers, metrics and predictions are saved.** The mT5 100-step run was copied to Drive with SHA256 verification; T5-base and FLAN-T5-base trained directly into Drive. Each new final model matches its validation-selected checkpoint by SHA256.

| Model | Steps | Gloss WER | Exact match |
|---|---:|---:|---:|
| mT5-small | 100 | 103.55% | 0% |
| T5-base | 300 | 290.31% | 0% |
| FLAN-T5-base | 300 | 262.72% | 0% |
| Training-only retrieval | — | 114.36% | 0% |

The earlier 30-step mT5 smoke run reached 100% WER and 0% exact match. None of these results establishes usable translation. The larger T5-family trials generated many extra/repeated tokens; WER can exceed 100% because insertions count as errors. Lower teacher-forced validation loss did not establish good autoregressive generation.

All models were evaluated on **the identical 129 French/reference pairs** from signer S041. The training-only retrieval baseline also matches exactly across reports. Data: **111 train / 12 validation / 129 test**, signer-disjoint, weakly time-aligned candidates from official LSFB-CONT training shard 000002. Outputs are **Belgian LSFB**, not France LSF. No expert approvals were invented. The test was reused during exploratory model investigation; these are not official benchmark or independent confirmation results. mT5 versus T5 also has unequal training budgets. T5 and FLAN used the same 300-step budget, 128-token limit and four-beam decoding.

T5 validation loss: 3.64 → 2.83; FLAN final validation loss: 2.86. T5 training took 352.6 seconds; FLAN training took 406.8 seconds, excluding model download and decoding. Serial decoding was replaced by batch-eight inference with the same checkpoint, references, four beams and output limit; T5's batched evaluation took 59.2 seconds. Notebooks now use batching and preserve completed runs rather than retraining them when rerun.

Drive paths under `MyDrive/SignaCulture/mt5-lsfb-poc/`:

- `mt5-small-steps-100/final/`
- `t5-base-steps-300/final/`
- `flan-t5-base-steps-300/final/`

Each run includes checkpoints, `metrics.json`, `predictions.json`, `history.json`, `run_config.json`, `COMPLETED.json` and `DRIVE_VERIFIED.json`. Full model hashes, pinned revisions, split fingerprints and decoding details are in [validation.json](validation.json). Private local metadata/prediction backups are in `.runtime/colab-results/`. The mT5 final model also has a verified local weight backup.

The [T5 companion notebook](SignaCulture_T5_Base_LSFB_Drive.ipynb) supports both `google-t5/t5-base` and `google/flan-t5-base` through `BASE_MODEL`. The existing [mT5 notebook](SignaCulture_mT5_LSFB_Drive.ipynb) remains separate. Five local extraction/split/notebook checks pass, alongside the real GPU runs and complete held-out evaluations.

These models remain research artifacts and are not enabled in the France-LSF avatar lookup. Improving annotation alignment and training coverage is the next evidence-based investigation; changing the base model did not yield an acceptable translator in these trials.

Detailed paper, target-coverage, tokenizer and independent-scoring audit: [COLAB_ZERO_EXACT_DIAGNOSIS.md](../../../text-to-gloss/lsfb/COLAB_ZERO_EXACT_DIAGNOSIS.md).
