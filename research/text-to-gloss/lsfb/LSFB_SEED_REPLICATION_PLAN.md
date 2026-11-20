# Narrow LSFB seed replication — frozen protocol

Registered before launching seeds 85 and 86, 2026-11-17. This is a reliability diagnostic of the existing current-text T5-base experiment, not a new independent benchmark.

Use the checksum-verified 145 training / 23 development examples, 35 training-selected labels and original conversation split in `.runtime/lsfb-narrow-context-data-v1`. Preserve all original hyperparameters: 25 epochs, 475 updates, rank-16 attention/feedforward LoRA, BF16, learning rate 2e-4 with the original schedule, batch 2 / accumulation 4, dropout .05, input limit 384, target/generation limit 192. Train from the same pinned local pretrained base. Only the initialization/training RNG seed changes (84 original, 85 and 86 replications). Evaluation subsets and lowest-development-WER checkpoint selection remain identical.

Compare each run with current-text retrieval on exactly the same 23 examples. Report exact sequence match, operation WER, multi-sign performance, checkpoint epoch and runtime; summarize all three seeds without selecting a winner for deployment. This development set has already influenced earlier work. No p-value or independent-test claim, no qualified reviewer approval and no France-LSF mixing. Keep failed runs and malformed/empty predictions in the denominator.

```bash
.runtime/t5-local-venv/bin/python -m training.lsfb_cont.train_narrow_context --data .runtime/lsfb-narrow-context-data-v1 --output .runtime/lsfb-narrow-current-seed85-v1 --epochs 25 --input-mode current --seed 85
.runtime/t5-local-venv/bin/python -m training.lsfb_cont.train_narrow_context --data .runtime/lsfb-narrow-context-data-v1 --output .runtime/lsfb-narrow-current-seed86-v1 --epochs 25 --input-mode current --seed 86
```

Both runs use existing local weights and data. No new downloads or held-out official-test access are needed. This script saves selected adapters but does not resume optimizer states; an interrupted run must use a new output directory and restart. The later graph trainer's resume functionality does not apply here.
