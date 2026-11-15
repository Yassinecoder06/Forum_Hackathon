# Implementation status

First measured milestone completed 2026-11-15. Branch `feat/research-training`, sole engineer codex2, writes confined to training/, research/, data/catalog/, data/text_to_gloss/.

- Read entire research brief (767 lines) and original plan (531 lines). Paper B PDF absent; official access blocked (403 PDF / 401 API). NOT read; supplied paper claims not independently verified.
- STK public CC BY-SA 4.0: 108 source segments → 106 derived SOURCE-ANNOTATED RESEARCH pairs. Local linguistic approvals: ZERO. Strict approval gate preserved.
- Frozen train/val/test counts: 39 / 28 / 39. Recording component counts: 1 / 1 / 2. Two LS-Colin extracts share one recording and stay together. Single-source, cross-corpus signer identity unknown; no benchmark-generalization claim.
- SQLite catalog with original annotations, derivations, pair lineage, reviews, lexical/video references, model runs. Five meaningful catalog tests passed.
- Actual T5-small LoRA CPU smoke: 2 epochs, 8.2657 s training, 27.0761 s total; trainable LoRA parameters 294,912; validation loss 4.5381 → 4.3329. Test exact match 0%, WER 207.2816%, BLEU 0; retrieval WER 111.1650%, lexical WER 196.6019%. Model is WORSE than retrieval. All held-out predictions in research/runs/t5-small-cpu-smoke/.
- CPU inference CLI actually executed. Output is candidate-only, pending review, explicit false translation/publication flags. Current smoke mostly copies input; unsuitable as a translator.
- Lead's 447-entry Commons inventory imported locally, no service requery. 447 lexical concepts/video references, ZERO sentence pairs. Bonjour metadata handoff recorded; lead reports successful real capture/render. Motion/audio/UI remain lead-owned.
- CUDA install still downloading pinned torch 2.7.1 + cu128. Initial current-selector download timed out; CPU smoke ran independently in training/.venv-smoke. mT5 base weights downloaded; GPU experiment remains pending at this milestone.
- Next: inspect/import genuinely new licensed SUD textual utterances without duplicate UD inclusion, preserve frozen test, run actual mT5-small LoRA, report same baselines and measured GPU memory/runtime. Dicta-Sign-LSF-v2/Rosetta blocked by institutional download/redistribution restrictions, not bypassed.
