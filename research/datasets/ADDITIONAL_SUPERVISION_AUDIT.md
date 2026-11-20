# Additional supervision audit — 2026-11-17

This bounded audit did not establish a new downloadable set of verified French-to-gloss sentence pairs. It does not prove that no suitable dataset exists. Languages and representation types remain separate.

## GloSL LSFB metadata: larger row count, no gloss supervision

The public [pedroodb/glosl-lsfb release](https://huggingface.co/datasets/pedroodb/glosl-lsfb), revision `754fc076b3239c04dbaf8c63d0c8b6df92d0de5a`, describes a conversion of LSFB-CONT, already present locally. It is Belgian LSFB (`sfb`), not France-LSF. The actual CSV has 5,436 nonempty French text entries and **zero nonempty gloss entries**. Every `gloss_type` is empty. Counts: 4,476 train, 155 dev, 805 test. Merely having a `gloss` column does not make a parallel gloss dataset.

Only the README (2,065 bytes), annotations CSV (1,486,171 bytes) and card (3,164 bytes) were downloaded through Colab, plus two papers. No video/pose bundles or model weights were downloaded. Metadata copies are private and hashes are recorded in `data/audit/private/new-source-audit/PROVENANCE.json`; public aggregate output is `data/audit/glosl_lsfb_metadata_audit.json`.

The card declares CC BY-NC-SA 4.0 and public access. These are the third-party distributor's statements, not an independent confirmation of every original recording's redistribution permission. The card's coverage note (5,389 built clips and 291 missing pose/video clips) does not reconcile directly with the 5,436 CSV entries; no assets were fetched to resolve that discrepancy. Pose confidence may be synthesized as 1 when not present in source; coordinates are not calibrated 3D. No rows from this copy were admitted to training or an independent test set.

## Other source leads

- **Rosetta-LSF (France-LSF):** The public Hugging Face card for [`Langue-des-signes-francaise/rosetta-lsf`](https://huggingface.co/datasets/Langue-des-signes-francaise/rosetta-lsf) was inspected at revision `40fb65ba8badfac178094886a666419304c92183` without downloading corpus files. Its card identifies French and `fsl`, states CC BY-NC 4.0, and describes an approximately 3-hour corpus with 194 French headline-to-LSF videos, a French-to-video correspondence CSV, `Annotations.csv`, and separate AZee files. The card explicitly says an ORTOLANG login is required to download the data. The public Hub repository itself contains only `.gitattributes` and `README.md`, reports zero bytes of dataset storage, and is not gated; therefore it supplies metadata but no usable training assets. The user's/organization's ORTOLANG eligibility and the conditions for use of the underlying videos are not established. No access-controlled content was requested or downloaded. This is a promising follow-up source only if ordinary institutional access is available and terms are confirmed; AZee must remain a separate target type unless a documented conversion is reviewed.
- [Matignon-LSF paper](https://www.sign-lang.uni-hamburg.de/lrec/pub/24024.pdf): describes substantial France-LSF broadcast material. Its 2024 description explicitly distinguishes subtitles aligned to speech from alignment to signing, which remained future work. It does not establish gold sentence-gloss supervision. Current archive contents and access rights remain unverified; no recordings downloaded.
- [STK LSF motion-capture paper](https://www.sign-lang.uni-hamburg.de/lrec/pub/24031.pdf): describes France-LSF grammatical material designed with a native Deaf signer and body/hand/face/gaze capture. The 2024 conclusion says linguistic annotation remained to be carried out. A currently available public downloadable annotated release and its terms were not verified. Do not conflate this motion corpus with the local UD STK Autogramm treebank.
- [40brèves resource catalog](https://www.sign-lang.uni-hamburg.de/lrec/data/40breves.html) and [concordancer paper](https://aclanthology.org/2020.lrec-1.744/): French/video correspondences and AZee-related resources do not automatically provide flat gloss sequences. Current archive payload and access remain unverified. No login-protected demonstration or access-restricted files were accessed; AZee was not flattened into gloss.
- Swiss LSF-CH resources remain excluded from France-LSF and Belgian LSFB experiments. Motion-only or isolated-label resources cannot substitute silently for sentence supervision.

## Applied safeguard and next experiment

`research/code/src/data/audit_parallel_csv.py` dynamically inspects CSV columns, counts nonempty targets, preserves an explicit source language and rejects conflicting per-record language labels. Even nonempty target strings do not automatically receive gold/reviewed status. Tests cover empty targets, unreviewed nonempty targets and language conflicts.

The next executed learning experiment uses only existing local Belgian narrow-task data and weights: frozen-protocol current-text T5-base seed replications (85, 86), compared with original seed 84 and retrieval. See `LSFB_SEED_REPLICATION_PLAN.md`. No empty-gloss data, synthetic uppercase subtitles, new language mixture or enlarged unreviewed vocabulary is used.

Reproduce the metadata scan:

```bash
python3 -m src.data.audit_parallel_csv data/audit/private/new-source-audit/annotations.csv --sign-language sfb --output data/audit/glosl_lsfb_metadata_audit.json
.runtime/t5-local-venv/bin/python -m pytest research/code/tests/test_parallel_csv_audit.py research/code/training/lsfb_cont/test_narrow_context.py -q
```

The input CSV is a private local audit copy, not included in a public redistribution. Use the pinned public upstream revision and inspect its terms if reproducing on another machine.

## France-LSF source follow-up

- **PGloss-LSF** is the strongest representation match identified here. The
  authors report approximately 3,000 sentence pairs in the expanded corpus,
  about two hours of signing, and annotations that include spatial and
  nonmanual features. Their evaluation of the expanded set was ongoing.
  The paper supplies examples and a description. A downloadable dataset,
  license for the pairs, and recording-disjoint split were not verified.
  Request a small authorized sample with original recording identifiers and
  keep PGloss targets separate from SUD FORM and AZee.
  [Primary paper, sections 4.1–4.2](https://www.sign-lang.uni-hamburg.de/lrec/pub/26046.pdf).
- **CREAGEST** is a genuine France-LSF corpus, including adult dialogues.
  The corpus catalog lists ELAN, approximately an hour of annotation, and
  registered access for a subset of video. French translation availability
  is unspecified. The catalog lists CC BY-NC-ND 3.0, and the ORTOLANG
  deposit page timed out in this check. Neither an aligned training schema
  nor terms for the proposed derivatives were verified.
  [Corpus record](https://www.sign-lang.uni-hamburg.de/lr/compendium/corpus/creagest.html),
  [project paper](https://aclanthology.org/L10-1245/).
- **DEGELS2012** documents an approximate written French translation of an
  LSF excerpt for a segmentation challenge. This is a candidate for studying
  temporal segmentation rather than a verified French-to-gloss training
  release. No downloadable sentence pairs with clear reuse terms were
  established from the paper. [Challenge paper](https://aclanthology.org/W12-1201.pdf).
- **TUB collection** distinguishes the subtitle-based Matignon and Mediapi
  sources from annotated Dicta-Sign, and describes Rosetta's AZee targets.
  This supports keeping these tracks separate. No new permitted Dicta-Sign
  download was found in this check.
  [Collection paper, Table 1 and section 3.1](https://www-live.dfki.de/fileadmin/user_upload/import/16091_ivaadjunct25-13.pdf).

No new examples were admitted. The best next acquisition is a small
PGloss-LSF sample with explicit reuse terms or ordinary institutional
access to the original Dicta-Sign-LSF-v2 annotations. Check translation and
gloss tier alignment, recording IDs and repeated constructions before
planning a fresh held-out training run.
