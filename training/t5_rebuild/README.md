# Atomic-unit LSFB T5 rebuild

This is the separate Belgian LSFB experiment requested in the Forum workspace. It uses
the LSFB-ISOL training inventory for output units and LSFB-CONT French/annotation pairs
for sentence-level training. It is not connected to the France-LSF avatar or motion
lookup.

## Recorded progress

- The 11-pair memorization gate passed at 120 steps. This is a pipeline check, not a
  translation or generalization result.
- The frozen rebuild dataset contains 4,396 training pairs and 714 validation pairs,
  with disjoint recordings and signers. Its training-derived inventory covers 93.5% of
  validation reference units. The official test shard was kept closed during selection.
- The earlier T5-base run had no recovered completion marker. A fresh T4 run
  (`signaculture-t5-atomic`) completed 10,900 steps with AdamW at `2e-5`, a 20-epoch
  maximum, early stopping, validation-WER checkpoint selection, atomic units, and the
  locked 128-token generation cap. Best checkpoint: step 7,630.
- Validation at the selected checkpoint: constrained WER 3.498 (349.8%) and exact match
  2.55%. On the previously unopened official test shard, 435 eligible examples were
  evaluated: constrained WER 3.924 (392.4%) and exact match 2.76%. Training-only TF-IDF
  retrieval had WER 3.009 (300.9%) and exact match 1.15%. Raw model decoding had 46.2%
  invalid sequences. This experiment does not establish a usable translator.
- Test evaluation used no test-vocabulary additions, had no signer overlap with train,
  and kept the five overlength candidates in the exclusion report. Linguistic approval
  remains pending.
- Data, run metadata, and a compact final report are local under
  `.runtime/colab-results/t5-lsfb-atomic/`. The selected 912 MB model and full prediction
  files remain on the Colab runtime: Colab's file-transfer endpoint returned HTTP 500,
  and Drive was not mounted. The final model SHA-256 is
  `5044e1966ec14066869809032a303afe7d21716368f1f24454c8de60b73bb48d`.

## Resume and finish

The training and locked test evaluation are complete on the Colab runtime. To move the
model and full predictions into a durable local/Drive backup, reconnect to
`signaculture-t5-atomic`, restore Drive authorization or repair Colab file transfer, and
copy `/content/t5_rebuild/full-run/final/` plus its result JSON files. Do not select
checkpoints or tune settings from the reported test results.

If Drive has no complete checkpoint, recreate the runtime and frozen data from the
pinned sources using `prepare.py`, `collect_annotations.py`, `isol_vocab.py`, and
`build_dataset.py`; then rerun the training script in a new output directory. Preserve
the official test shard as unopened until validation selection is complete.

Earlier, smaller exploratory runs are recorded in [`../colab/COMPARISON.md`](../colab/COMPARISON.md).
They used a different split and do not substitute for the unfinished 4,396-pair run.
