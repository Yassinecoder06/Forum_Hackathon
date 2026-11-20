# Source-annotated research pipeline

All output is pending local linguistic review. Neither completed model is a useful translator. The default training mode requires human-approved records and rejects this research corpus. The explicit research mode additionally checks frozen export hashes and exact source-derived catalog records; relabelling flags does not authorize arbitrary data.

Rebuild data (Python standard library; public licensed text only):

```bash
python training/catalog.py
python training/import_sud.py
python training/import_commons.py --inventory data/catalog/commons_lexical_metadata.json
python training/register_sources.py
```

`catalog.py` freezes v001 and refuses changed exports. `import_sud.py` freezes v002 while preserving v001 val/test bytes; overlapping source documents are catalogued as raw provenance and excluded from added training rows. Commons importer makes no network requests and imports lexical/video references only. Rosetta/Dicta-Sign-v2 have no content importer because institutional rights and actual row alignment are unverified. Schema is created by catalog.py. Tests: `training/.venv-train/bin/python -m pytest training/test_catalog.py -q`.

Environments and downloaded/trained weights are ignored. Python3.11.14. Exact installed versions in requirements-smoke.lock.txt (CPU) and requirements-train.lock.txt (CUDA). Install CUDA PyTorch separately from official index before the remaining pinned packages; `torch==2.7.1+cu128` uses `https://download.pytorch.org/whl/cu128`. HF CLI downloaded pinned revisions without authentication or uploads:

```bash
hf download google-t5/t5-small config.json generation_config.json model.safetensors tokenizer.json tokenizer_config.json spiece.model --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 --local-dir training/weights/t5-small
hf download google/mt5-small config.json generation_config.json pytorch_model.bin spiece.model tokenizer_config.json special_tokens_map.json --revision 73fb5dbe4756edadc8fbe8c769b0a109493acf7a --local-dir training/weights/mt5-small
```

Executed experiments (use a new run name when reproducing; existing outputs deliberately cannot be overwritten):

```bash
training/.venv-smoke/bin/python training/experiment.py --mode source-annotated-research --base google-t5/t5-small --base-path training/weights/t5-small --revision df1b051c49625cf57a3d0d8d3863ed4d13564fe4 --run t5-small-cpu-smoke --lora --epochs 2 --device cpu
training/.venv-train/bin/python training/experiment.py --data data/text_to_gloss/v002 --mode source-annotated-research --base google/mt5-small --base-path training/weights/mt5-small --revision 73fb5dbe4756edadc8fbe8c769b0a109493acf7a --run mt5-small-lora-expanded-slow --lora --epochs 8 --device cuda --lr 0.0002
```

Native slow SentencePiece is used for mT5 to preserve byte fallback; T5 smoke uses its fast tokenizer. No examples are truncated: over512tokens fails. Fixed greedy decoding max128newtokens; checkpoint selected only by validation loss. Baselines use only training data. WER aggregates Levenshtein edit counts over all reference tokens. BLEU is sacrebleu case-sensitive tokenization:none on0–100 scale; subword loss is separate. OOV is vs the training gloss vocabulary; approved motion-map OOV is undefined and explicitly null.

Inference (both CPU and CUDA actually checked for mT5):

```bash
training/.venv-train/bin/python training/infer.py --text 'Bonjour à tous' --model training/checkpoints/mt5-small-lora-expanded-slow --device cpu
```

Returns JSON candidate `gloss_sequence` tokens plus exact base revision, run, dataset digest and checkpoint hashes; verifies checkpoint hashes before loading. `translation_validated` and `render_allowed_for_publication` are false. Current mT5 often emits `<extra_id_0>`: this is an observed model failure, not a sign ID. No silent substitution or motion mapping. LoRA configuration references this worktree's absolute base-model path; keep that local base directory or review its adapter config when moving checkpoints. Lead owns API integration and should retain existing controlled retrieval behavior.

The requested adaptation follow-up is documented in `research/ADAPTATION_INVESTIGATION.md`, with settings locked in `research/MT5_LONG_RUN_PLAN.json`. It remains a failed translation experiment: lower teacher-forced loss and no sentinels, but repetitive/malformed final output and WER517.96%. Historical runs used the script at commit078cafb; the current script corrects partial gradient accumulation and adds token-weighted validation selection/early stopping, so use that historical script revision for exact old-run reproduction.

The one longer run was executed as follows (a reproduction must use a new run name):

```bash
training/.venv-train/bin/python training/experiment.py --data data/text_to_gloss/v002 --mode source-annotated-research --base google/mt5-small --base-path training/weights/mt5-small --revision 73fb5dbe4756edadc8fbe8c769b0a109493acf7a --run mt5-small-lora-long-validation-selected --lora --epochs 100 --device cuda --lr 0.001 --rank 16 --alpha 32 --targets q,k,v,o,wi_0,wi_1,wo --dropout 0.05 --accumulation 4 --patience 10 --min-delta 0 --warmup-ratio 0.05 --weight-decay 0 --validation-loss token
```

`diagnose_adaptation.py` audits only train and runs its four-example optimization diagnostic; it reads neither validation nor test. It writes to a new diagnostic output directory and intentionally refuses overwrite. New protocol tests are in `training/test_adaptation.py`; run them together with test_catalog.py. Parameter count is saved before selected-adapter freeze; train/val only are parsed during training, followed by a selection-complete record and one final test pass. Existing run artifacts and data hashes are checked against the locked plan. No decoding change was made after final metrics were seen.

## Colab + Google Drive: Belgian LSFB mini experiment

The self-contained [Colab notebook](colab/SignaCulture_mT5_LSFB_Drive.ipynb) trains `google/mt5-small` on real, weakly aligned LSFB-CONT annotations and saves resumable checkpoints, final model/tokenizer and results to Drive. Use [the instructions and plan corrections](colab/README.md). This is a separate Belgian LSFB research experiment, not a France-LSF motion-model update.

The [T5-base companion](colab/SignaCulture_T5_Base_LSFB_Drive.ipynb) uses the same frozen Drive splits for a separate 300-step comparison.

## Current LSFB-CONT task preparation

The [context-conditioned lexical planning pipeline](lsfb_cont/README.md) prepares
the complete local train/validation annotations with source-video review,
conversation quarantine, explicit fingerspelling, existing T5 subwords and
provenance-bound review imports. Its [preparation result](../research/LSFB_CONT_PREPARATION_RESULT.md)
records the generated data and checks. These candidate pairs are exploratory
until linguistically reviewed; the earlier Colab experiments above remain
historical baselines.

Narrow Belgian LSFB seed reliability experiment: [frozen protocol](../reports/LSFB_SEED_REPLICATION_PLAN.md). The seed is configurable with `--seed`; the original default is 84. Runs use existing local data and model weights. Development results remain diagnostic, with no qualified linguistic approval or official-test claim. Additional dataset leads and empty-target safeguards: [supervision audit](../reports/ADDITIONAL_SUPERVISION_AUDIT.md).
