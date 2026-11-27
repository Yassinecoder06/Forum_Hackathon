# Belgian LSFB-only extraction and training

The active run uses only `/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont-usable-v1`. No France-LSF application assets, dictionary clips, or annotation IDs are used by the scanner or trainer. Language in every research/code/training/evaluation row is asserted to be LSFB.

## Full local scan

Source: `/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont`, pinned revision `64b0c112f59d3710367fb736ef1808f37ff15e28`.

All24 listed source files were size-checked and SHA256 hashed; all available pinned LFS digests matched. This includes20 pose TARs and the31.5GB video TAR. Video content was hashed, not decoded frame by frame. Numpy pose arrays were read with pickle disabled and inspected for dimensions, finiteness and zero frames. There are1,619 unique recordings and16,190 indexed body-part arrays. The recorded arrays and all extracted crops are finite with no wholly zero frames. These checks establish technical integrity, not tracking accuracy or linguistic correctness. Official test labels remain unopened; test file integrity/pose structure was inspected only.

An extraction-stage bug concerning cached source-role metadata was fixed after the full integrity pass. Extraction resumed from the completed verified inventory. Source role and conversation metadata were reconstructed from verified shard IDs. Scan and extraction logs are `.runtime/lsfb-full-scan.log` and `.runtime/lsfb-full-extraction.log`.

## Separate extracted package

Folder: `/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont-usable-v1/`

- `train.jsonl`:2,146 technically eligible, unreviewed French-context→LSFB lexical pairs.
- `val.jsonl`:385 pairs, with no shared conversations, signers or normalized current French phrases with training.
- `quarantine.jsonl`:2,823 candidates with explicit overlapping exclusion reasons.
- `poses/*.npz`:2,531 corresponding actual Belgian LSFB pose crops, containing body, hands, lips, eye/iris and brow components.
- `inventory.json`, `pose_index.json`, `SCAN_REPORT.json`:source hashes, offsets, technical statistics and package hashes.
- `README.md`:scope and interpretation.

The extractor considers every5,354 source French candidate segment from the seven train/validation annotated shards. It removes478 exact-time duplicate lexical operations across all candidates. This count differs from327 because the earlier experiment counted only its narrower eligible rows. Ambiguous overlapping lexical events, fingerspelling needing its own task, uncertain boundaries, fragments/editorial notes, excessive lengths and tokenizer incompatibilities are quarantined. No automatic semantic corrections or linguistic approvals were invented. Pointing, depiction and other nonlexical information remain in raw-event sidecars; model targets are explicitly lexical projections.

Training contains88 conversations/40 signers; validation contains29 conversations/5 signers. Training has277 single-sign and1,869 multi-sign examples. Validation has57 single-sign and328 multi-sign examples. The1,756 training label types do not cover307 of2,054 validation operations; the final report must distinguish all-validation and seen-vocabulary results. Pose-only recordings without train/validation annotations (1,179 unique records) are inventoried rather than fabricated into French-label training pairs.

## Active training protocol

Runner:`research/code/training/lsfb_cont/train_extracted.py`.

Pinned pretrained T5-base; fresh rank16 adapters on q/k/v/o/wi/wo; BF16; gradient checkpointing; batch2 with accumulation4; three shuffled epochs,807 optimizer steps; peak LR2e-4 with warmup and decay. Source contains current French and completed preceding French dialogue. Source/target budgets384/192 tokens with no silent truncation. Targets are readable JSON lists of original LSFB lexical lemmas.

A fixed96-row validation subset selects checkpoints using raw serialized-token WER, which never drops malformed predictions. Operation WER, exact match, invalid syntax, termination-cap hits and single/multiple-sign scores are tracked. After training the selected adapter is reloaded and scored against all385 validation rows; seen-vocabulary results are additionally reported. No official test labels are used for selection or scoring. This is a larger/different evaluation from the prior51-row bounded experiment; its scores must not be directly ranked against those historical scores.

Run folder:`.runtime/lsfb-extracted-train-v1/`; progress is `progress.json`, checkpoints `best_adapter/`, and completion evidence `RESULT.json` and `validation_predictions.json`.

## Additional accessible Belgian resource

GeoQuery-LSFB was found at https://zenodo.org/records/19220833 and https://gitlab.unamur.be/beehaif/GeoQuery-LSFB . Source README, license and both JSON releases were downloaded into `/home/jmal/Projects/Active/Forum_Hackathon/datasets/geoquery-lsfb-source-v1/`, with exact upstream commit and file hashes in `PROVENANCE.json`. No1.6GB recording ZIP was downloaded.

The source describes250 human-recorded/annotated geographic questions and a4,519-row augmented release. A separate candidate package at `/home/jmal/Projects/Active/Forum_Hackathon/datasets/geoquery-lsfb-usable-v1/` contains203 original training rows,47 original validation rows grouped by template type, and3,081 synthetic training candidates excluding held-out types and French strings in the original set. This does not establish full semantic independence between templates; further alignment checks are necessary before benchmarking. Human provenance refers to the authors' release, not local linguistic approval.

These graphs preserve hand roles, sign-occurrence IDs, HamNoSys and temporal relations. They are not silently mixed with LSFB-CONT author lemmas. A graph-target/tokenizer compatibility check is required before any training on that new task. The current local run uses LSFB-CONT only.

## Colab

No active Colab SSH tool, running tunnel, or configured SSH host was found. The user was asked for a current SSH connection command while local work continued. No claim is made that a Colab session was opened or that any checkpoints were uploaded.

## Completed result

Training completed807 optimizer steps and all385 validation examples in765.8 seconds (12.8 minutes). The selected epoch3 adapter was reloaded before full validation. Exact match15/385 (3.90%), operation WER240.60%, raw serialized-token WER162.35%, invalid JSON132/385 (34.29%). Multi-sign exact4/328 (1.22%); single-sign exact11/57 (19.30%). Seen-vocabulary subset12/207 exact (5.80%). The current-French TF-IDF retrieval baseline scores113.92% operation WER on the same385 examples, so this model does not beat that baseline on operation WER. These are diagnostic development measurements, not an untouched test benchmark.

Additional direct verification compared every extracted crop with the original annotated source:25,310 body-part comparisons across2,531 segments, zero mismatches. Evidence:`PAIRED_POSE_VERIFICATION.json` in the dataset folder. All21 targeted tests passed.

A local data-and-model backup is available at `/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont-usable-v1/training-run-v1/`, including the selected LoRA adapter, tokenizer, exact configuration, metrics, predictions and file hashes. Its adapter config pins the same pretrained base revision used throughout the run. The base model remains in the verified local HF cache. No Colab connection was available and no France-LSF assets were used. The trained model remains unsuitable for reliable translation; corpus file integrity and correct pose extraction have not resolved lexical supervision ambiguity, sparse label coverage or generation failure.
