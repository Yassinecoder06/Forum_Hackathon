# Narrow Belgian LSFB: three-seed reliability experiment

Completed 2026-11-17 on the local RTX 4060 Laptop GPU using existing local data and T5-base weights. Two new 25-epoch / 475-update LoRA runs replicate seed 84 with seeds 85 and 86. All run manifests match exactly except the seed. Selected adapters were reloaded before final decoding. No official test set was opened.

| Method | Exact / 23 | Operation WER | Multi-sign exact / 6 | Multi-sign WER | Selected epoch |
|---|---:|---:|---:|---:|---:|
| Current-text retrieval | 7 | 63.33% | 0 | 61.54% | — |
| T5-base seed 84 | 12 | 46.67% | 0 | 69.23% | 11 |
| T5-base seed 85 | 10 | 53.33% | 0 | 69.23% | 12 |
| T5-base seed 86 | 11 | 50.00% | 0 | 69.23% | 20 |

Across seeds: mean exact 47.83% (sample SD 4.35 percentage points), mean WER 50.00% (sample SD 3.33 points). These quantify optimization variation on the same examples, not uncertainty across independent populations. All three seeds have lower aggregate development WER than retrieval; none improves exact multi-sign performance, and all have worse multi-sign WER than retrieval. All final generations terminate without empty outputs or generation-cap hits.

## What this establishes

The aggregate gain over retrieval repeats across three seeds on this particular inspected development set. It comes entirely from single-sign exact matches. This does not establish general sentence translation, semantic correctness or independent test superiority. No failed model is promoted to avatar inference.

Five of six multi-sign development target sequences are absent from training. Only one has all adjacent label pairs seen in training, although each individual label occurs at least three times. This is consistent with sparse combination coverage but does not isolate a causal explanation; French/annotation alignment and linguistic adequacy still need qualified review. Private per-example evidence: `research/private/lsfb-seed-replication/multisign_support.json`.

## Execution and evidence

145 training / 23 development examples; 35 train-selected labels; original conversation/signer separation. Each run selects its own lowest development-WER checkpoint using the existing protocol. Two new runs took 332.07 and 327.59 seconds respectively. They partially overlapped and runtime is not a controlled hardware comparison. Approximately 1.5 GB combined GPU memory was observed during concurrent execution; this is not a measured peak-memory benchmark.

Eight targeted tests passed (metadata admission, original split/scoring checks and aggregation consistency). Python compilation and git whitespace checks passed. Downloaded metadata copies matched all three provenance digests. No new datasets or models were downloaded for training. The separate metadata audit used Colab; no empty-gloss rows were admitted.

Run commands and settings: [frozen protocol](LSFB_SEED_REPLICATION_PLAN.md). Machine-readable results: `research/text-to-gloss/lsfb/LSFB_SEED_REPLICATION_RESULT.json`; support counts: `research/text-to-gloss/lsfb/LSFB_MULTISIGN_SUPPORT.json`; test output: `research/text-to-gloss/lsfb/logs/lsfb_seed_replication_tests.txt`.

```bash
.runtime/t5-local-venv/bin/python -m training.lsfb_cont.summarize_seed_replication --runs .runtime/lsfb-narrow-current-run-v1 .runtime/lsfb-narrow-current-seed85-v1 .runtime/lsfb-narrow-current-seed86-v1
```

Both new adapters, tokenizer files, manifests, histories and raw predictions have hash-verified backups under `/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-narrow-context-v1/seed-replication/`. Raw source text and per-example evidence stay private. The existing trainer saves selected checkpoints but cannot resume optimizer state mid-run; interrupted reproductions must restart in a new output directory.

## Next best step and limits

The 23 development examples are repeatedly inspected and too small for a robust benchmark. Three seeds do not increase their effective sample size. Targets are lexical projections with pending review, not complete sign-language sentences or nonmanual grammar. No qualified LSFB review is available.

Before another general-purpose T5 run, obtain review of the queued Belgian multi-sign training examples and source alignment; then freeze an adequate conversation-held-out evaluation with repeated constructions and labels represented in training. Define the admission and scoring rules before inspecting its predictions. Official test labels remain unopened. Do not expand vocabulary, combine sign languages, or deploy this as a sentence translator on the strength of this diagnostic.
