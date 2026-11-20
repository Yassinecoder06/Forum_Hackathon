# Active France-LSF model comparison

Data:39training/28development UD-FSL-STK-FORM examples verified against original annotations. No SUD/UD convention mixing, LSFB labels, SignVerse pseudo-glosses or restricted annotation content. Original test39examples withheld in these runs. French targets are free translations of published gloss annotations; local LSF semantic approval pending. This is explicitly a small feasibility study, not a defensible general translator benchmark.

Models: local mT5-small and T5-base (different sizes AND pretraining/tokenizers; not a pure size ablation). mT5-base requested through Colab after public ungated license/size checks to enable a within-family comparison. All launched runs use identical39/28splits, seed71, BF16, rank16LoRA across attention/feedforward modules, LR3e-4, effective batch8, gradient checkpointing, up to40epochs and validation-token-loss early stopping patience6. Selected adapters are reloaded before full-development greedy scoring. No targets silently truncated; target UNK rejects training. Scripts/configs/history/weights remain local under research/private/. No production model promotion implied.

Reproduce:

```bash
.runtime/t5-local-venv/bin/python -m training.fsl_size_comparison --model .worktrees/wt-codex2/training/weights/mt5-small --name fsl-mt5-small-reproduction
.runtime/t5-local-venv/bin/python -m training.fsl_size_comparison --model /home/jmal/.cache/huggingface/hub/models--google-t5--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1 --name fsl-t5-base-reproduction
```

Progress logs:research/text-to-gloss/fsl/logs/fsl_mt5_small_training.log and reports/fsl_t5_base_training.log. Per-run history.json, BEST.json, RESULT.json and predictions.json provide completion evidence. optimizer-last.pt records optimizer/RNG diagnostics but automatic resume is not implemented in this comparison runner; use research/code/src/text_to_gloss/train_mt5.py for HF checkpoint resume after provisioning dependencies. Do not interpret diagnostic optimizer state alone as a resumable complete training checkpoint.

## Measured update

mT5-small completed40epochs/200updates. Selected epoch40 validation loss3.5671;28development predictions:0exact,594.85%WER,0BLEU,4.907chrF,60%unknown prediction tokens,28/28generation cap hits. This failed feasibility run is retained, not promoted.

T5-base first attempts failed the target-UNK gate, before optimization. Training-only character audit found `~ Â È Ê`; these four characters were added, embeddings resized and trained/saved alongside LoRA. Two validation examples retain unseen tokenizer symbols; those are counted in CONFIG.json and raw reference scoring retains the full annotations. No validation-derived vocabulary expansion occurs. Active successful run is `fsl-t5-base-comparison-v3`; v1/v2 are failed preflight artifacts. New embeddings mean this is additionally a tokenizer-adaptation comparison, not a pure size study.

T5-base v3 completed with selected epoch 16: 0/28 exact, WER 235.05%, BLEU 0.2480, chrF 14.4424, 94.16% unknown prediction tokens and zero cap hits. Saved checkpoint and prediction counts were rechecked against the original 39/28 manifest hashes.

mT5-base transfer is stopped incomplete, not running: local partial 1,747,976,192 bytes against expected 2,329,735,129. The original remote runtime is absent and the latest `colab sessions` check also reports no active sessions. No mT5-base training result exists. Its Apache 2.0/ungated metadata and pinned revision were checked in the earlier Colab acquisition audit; no new download was performed in this status update.

Authoritative current artifacts: private RESULT.json / CONFIG.json / predictions.json under each completed run, research/text-to-gloss/fsl/MT5_BASE_TRANSFER_STATUS.json and reports/CURRENT_RESEARCH_STATUS.json. Additional model size alone does not resolve the zero-review, tiny-data supervision limitation.
