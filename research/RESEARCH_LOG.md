# Research log

2026-11-15: Read complete user research brief and original plan before implementation. Verified branch/worktree and RTX 4060 Laptop 8188 MiB / driver 580.178.04. No access to lead's audio/motion environments or other worktrees.

Pinned UD STK revision d1a6c6487eeddf67a14a5e61116b7cc12a0d1f6a. Downloaded public textual CoNLL-U, README and license. Inspected five actual annotation examples. `Parallel:no` and repeated `text_fr` establish that this is free translation research, not independently validated parallel training. Kept all FORM brackets/pointers/case, original metadata and MISC columns in catalog. Merged only consecutive same-document/signer equal-normalized-French units; 108 →106. Source's official parsing-test file is repurposed as an explicitly local engineering experiment, not its official translation benchmark.

Created deterministic recording/signer/duplicate union groups before model results: 39 train /28 validation /39 test. No test-driven tuning. Maintained strict approved-mode default; research requires explicit CLI option and published-source annotation/rights checks. No APPROVED labels assigned.

Primary ORTOLANG metadata for Rosetta and Dicta-Sign-v2 imposes public university/research membership and ORTOLANG-only redistribution; content not downloaded. File listings are metadata, not inspected annotation contents. Paper B DOI resolves via Crossref; official PDF 403 and full API 401. Stopped retries; no PDF reading claim.

CUDA current unpinned wheel attempt timed out; restarted official cu128 torch2.7.1 with extended download timeout. Created separate Python3.11 CPU smoke env using cached torch2.15.0+cpu. Five catalog tests passed. Actual two-epoch T5-small LoRA training/evaluation completed: 8.27s training /27.08s total, 0 test exact match, WER2.0728, BLEU0. Retrieval WER1.1117. Full metrics, losses and all validation/test predictions saved. No improvement claim. CLI inference executed with pending-review JSON.

Imported lead-provided Commons inventory (447 compatible lexical clips) from local JSON, no repeated service calls. Earlier independent metadata inventory hit HTTP429 and was stopped; remote importer was replaced by local-only importer. Bonjour is motion-source metadata only; no new extraction by this engineer.
