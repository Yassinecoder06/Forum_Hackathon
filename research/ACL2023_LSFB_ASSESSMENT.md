# ACL 2023 text-to-gloss assessment and Belgian experiment

## Decision

Worth testing as a controlled change, not evidence of an LSFB solution. The prior T5 run failed both serialization and translation: 34.29% malformed JSON, 240.60% operation WER, and only 4/328 multi-sign exact matches. Removing JSON addresses only the serialization burden; it cannot repair alignment ambiguity or missing lexical coverage.

## What the references establish

[Zhu et al., ACL 2023](https://aclanthology.org/2023.acl-long.700.pdf) study low-resource neural translation, not a GAN or pose model. They use Marian Transformers, preprocessing, synthetic augmentation, transfer learning and auxiliary spoken-language translation. Their small baseline uses one encoder and two decoder layers. On broad-domain DGS, baseline test BLEU is 3.13; multilingual plus back-translation reaches 5.78. Augmentation alone can reduce performance. The paper also uses signer evaluation: automatic metrics alone do not establish usable signing.

The [companion repository](https://github.com/DFKI-SignLanguage/text-to-gloss-sign-language-translation) contains experiment scripts and German/DGS corpora, not a ready French-to-LSFB checkpoint. Checked commit: `8f6157cdff80530da3756dd7faa058d57f302703`. Its scripts assume site-specific paths and multiple GPUs; they cannot run unchanged on Colab. Source code is GPL-3.0; corpus terms are separate.

## Our data compatibility

Source: `/home/jmal/Dataset/lsfb-cont-usable-v1`. Prepared: `/home/jmal/Dataset/lsfb-acl2023-v1`.

- 2,146 training / 385 development pairs, including 1,869 / 328 multi-sign rows.
- 1,756 training lexical labels; 601 occur once. 307/2,054 development operations are unseen labels.
- Conversation, signer and normalized current-French separation is asserted. Official test labels remain unopened.
- All rows remain linguistically unreviewed. These are lexical projections, not complete LSFB grammar or simultaneous articulator representations.
- `*.fr`: current French with task tag; `*.context.fr`: preceding-context alternative; `*.gloss`: reversible space-delimited labels; metadata retains original targets and pose paths. Percentage escaping protects spaces and Unicode within labels; decode before linguistic review. Do not interpret escaped substrings as independent signs.
- Pose files are not needed by text-to-gloss training. The original verified pose dataset remains local.
- GeoQuery-LSFB stays separate: graph annotations, domain and synthetic templates differ. Do not merge its occurrence IDs with CONT lemmas.
- No France-LSF, DGS or ASL examples enter this training package. Upstream corpora are reference material only.

## Experiment order and acceptance

1. Fit a train-only 2k joint SentencePiece vocabulary and a small Marian 1-encoder/2-decoder baseline. Start with a 32-pair overfit check. Near-perfect training decoding establishes implementation learnability, not generalization.
2. Train current-French baseline; compare a context input ablation separately. No suffix stripping: CONT already supplies author lemmas, and German annotation rules do not transfer automatically.
3. Add original-plus-French-lemmatized source variants, keeping every gloss unchanged. This requires a verified French lemmatizer; do not substitute German tooling or treat lemmatization as translation.
4. Consider task-tagged French-to-English auxiliary data with a controlled sampling ratio. Such ordinary spoken-language examples are not foreign sign-language labels. Audit and deduplicate auxiliary sources against development before use. No auxiliary corpus has yet been downloaded/prepared for this run.
5. Only attempt back-translation after a reverse LSFB-to-French model passes inspection; do not recycle the failed T5 predictions as gold labels.

Keep the same 385 development rows, retrieval baseline and decoded lexical-token WER. Add BLEU (tokenize=none), chrF, TER, exact match, unknown-label rate, termination rate and multi-sign metrics. Score all references, including OOV labels, with a separately labelled seen-vocabulary subset. Fit tokenizers on training only. Development has already been repeatedly inspected: it is not an untouched test benchmark. Require better WER than the current-French retrieval baseline (113.92%) and improved multi-sign behavior before promotion. Human LSFB review remains required.

## Session and persistence

Created `signaculture-acl2023-lsfb`; verified SSH and Tesla T4 15,360 MiB. This corrects the previous report's incomplete Colab discovery: the local `colab` CLI can create sessions even without SSH config entries.

Google Drive mounting requires fresh browser consent. Until mount and checksum verification succeed, the Colab copy is temporary and the local package is the durable backup. No model training or improvement is claimed by this preparation report.

Prepared runner: `training/acl2023_lsfb/train_marian.sh`; syntax checked, not executed. It deliberately changes batch size, warmup, stopping and selection to form an LSFB pilot, so it is not an exact reproduction. Marian is not installed in the new runtime yet; training must follow dependency setup and the overfit gate, not blindly execute this full-run template.

Uploaded and SHA256-verified `/content/acl2023-lsfb-bundle.zip`; extracted under `/content/signaculture-acl2023-lsfb`. Parallel-file hashes are also checked remotely. Bundle includes the paper, pinned reference script/license, local assessment and exported data; it excludes German signed training corpora. Local bundle SHA256: `dcb7872685b0c26a159eff555051551bf8927edd821c11748a4caefa494dc8c5`.

Drive persistence helper: `/content/save_acl2023_drive.py` (local source `training/acl2023_lsfb/save_drive.py`), destination `MyDrive/SignaCulture/acl2023-lsfb-v1`. It refuses an absent Drive mount and verifies each copied file. Authorization remains pending; no Drive backup claimed.

## Drive backup confirmed

Google Drive mounted successfully after renewed authorization. All 17 package files were copied and SHA256-verified at `MyDrive/SignaCulture/acl2023-lsfb-v1`. Verification receipt: `BACKUP_VERIFICATION.json` in Drive, local copy `.runtime/acl2023-drive-verification.json`. This supersedes the pending-authorization notes above. Training has not started.
