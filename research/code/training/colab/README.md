# Colab mT5 / LSFB proof of concept

Upload **SignaCulture_mT5_LSFB_Drive.ipynb** through https://colab.research.google.com/ → File → Upload notebook. Select a T4 GPU and run the cells in order. The notebook is self-contained: it writes its helper script, downloads one exact pinned training shard, inspects weak candidate labels, filters overlength examples, makes group-disjoint splits and trains `google/mt5-small`.

Mount your Drive when prompted. All checkpoints (including optimizer state), final model/tokenizer, frozen splits, predictions and metrics save under:

`MyDrive/SignaCulture/mt5-lsfb-poc/`

Default `TRAIN_STEPS=30` is a smoke experiment. Inspect its loss and predictions, then change to `100` for a separate run directory. `ALLOW_WEAK_LABELS=True` is explicit experimental supervision, not expert approval; set it false to require approved rows. Checkpoints save at 15/30 or 50/100 steps. Rerunning training resumes the latest checkpoint, and mismatched data/configuration is rejected. Use a new run folder for changed configurations. Allow several GB of Drive space. A disconnection before the first checkpoint loses those initial steps.

Corrections to the supplied plan:

- Official shard 000000 is the **test** split. This notebook uses official training shard **000002** (1,146,209,792 bytes), pinned dataset revision `64b0c112f59d3710367fb736ef1808f37ff15e28`.
- Output is **Belgian LSFB**, not France LSF. It is not wired to the Dicta-Sign France-LSF motion lookup.
- No FP16: use Adafactor and BF16 when PyTorch reports support, otherwise FP32. The actual T4 runtime selected BF16 under PyTorch 2.14.1; this can be emulated on T4, rather than native hardware BF16. Full-model mT5 is not guaranteed to fit every runtime; this Colab training has not been benchmarked here.
- Raw JSON schemas were inspected in the real downloaded shard. Extraction yielded 300 unique weak candidates from 21 recordings / six signers, with no source video or pose arrays loaded into training. Candidate text/glosses retain corpus labels and are not fabricated. Of 300 candidates, 252 fit the 96-token limit: signer-disjoint split 111 train / 12 validation / 129 test.
- Overlength inputs/targets are retained separately before splitting; targets are never silently truncated. Split proportions can be uneven because signers stay intact. This is an internal holdout within a training shard, not an official benchmark result.
- Checkpoints go to Drive during training, rather than only copying the final model after the run.

Local validation: five extraction/group/leakage/metric tests pass. A two-step tiny random T5 software fixture exercised actual Transformers Trainer, Adafactor, validation selection, optimizer checkpoints, final model/tokenizer and metrics. This is a plumbing test, **not trained mT5 performance**. The notebook code cells compile. Subsequent SSH execution completed actual 30-step and 100-step full mT5 runs on a Colab Tesla T4 without OOM. Validation loss reached 16.16 and 13.31 respectively; held-out exact match stayed 0%, with WER 100% and 103.55%. These models remain unusable as general translators. The smoke model was backed up locally; the 100-step final model and tokenizer are backed up at `.runtime/colab-results/poc-100/final/`, with archive and SHA256 verification. Google Drive subsequently mounted successfully. The mT5 100-step run, including checkpoints and final weights, was copied to `MyDrive/SignaCulture/mt5-lsfb-poc/mt5-small-steps-100/`; its model SHA256 was verified against the local backup. See validation.json.

The dataset is CC BY-NC-SA 4.0. Keep this experiment private while reviewing derivative-sharing requirements. Original plan: `/home/jmal/Downloads/SignaCulture_mT5_LSFB_PoC_Plan.md`. Sources: https://huggingface.co/datasets/ppoitier/lsfb-cont and https://huggingface.co/docs/transformers/model_doc/mt5.

Regenerate embedded helper after editing `lsfb_poc.py`:

```bash
python3 research/code/training/colab/build_notebook.py
.venv/bin/python -m pytest research/code/training/colab/test_lsfb_poc.py -q
```

## T5-base comparison

[SignaCulture_T5_Base_LSFB_Drive.ipynb](SignaCulture_T5_Base_LSFB_Drive.ipynb) reuses the existing frozen Drive splits. The SSH comparison uses `google-t5/t5-base` revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`, 300 steps, validation every 50 steps, and a 128-token limit because T5 tokenizes these same examples differently (maximum observed: train 112, validation 73, test 117). No labels are truncated or re-split. The mT5 100-step and T5 300-step configurations have different budgets and tokenizers; their validation losses are not directly comparable. Compare held-out WER/exact match as exploratory results, with the reused-test limitation stated explicitly.

The companion also supports `BASE_MODEL='google/flan-t5-base'`, pinned to revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`. It selects a separate Drive run folder. [FLAN-T5-base](https://huggingface.co/google/flan-t5-base) is instruction-fine-tuned; that is a reason to test it, not evidence of LSFB competence. T5-base completed with 0% exact match and 290.31% gloss WER; its lower teacher-forced validation loss did not translate into good autoregressive outputs. Serial decoding was replaced by batch-eight inference with unchanged beam search and limits. `evaluate_saved.py RUN SPLITS --batch-size 8` evaluates an already saved checkpoint and preserves partial prediction progress.

Final results: [completed Colab comparison](COMPARISON.md). T5-base and FLAN-T5-base both completed; all models are in Drive with verified hashes. Their measured accuracy remains insufficient.
