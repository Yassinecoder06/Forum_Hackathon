# LSFB-CONT: context-conditioned lexical planning

This preparation targets **French text plus completed preceding dialogue → lexical
LSFB operations**, using the local `ppoitier/lsfb-cont` corpus. It is a separate
task from reconstructing all signing features or driving the France-LSF avatar.

The prepared local dataset is currently:

`/home/jmal/Projects/Active/Forum_Hackathon/.runtime/lsfb-cont-task-v3/`

Its source is `/home/jmal/Dataset/lsfb-cont`, pinned to revision
`64b0c112f59d3710367fb736ef1808f37ff15e28`. The seven train/validation TARs
were SHA-256 verified against their local HF tree metadata. The official test
TAR, pose-only shards and 30 GB video archive are not copied into the training
package. The existing official test was used by the previous experiment and is
not a fresh development holdout.

## Prepared artifacts

| Artifact | Purpose |
|---|---|
| `candidates.jsonl` | All 5,354 nonempty, valid-duration French translation segments, including technical exclusions and quarantine reasons |
| `exploratory/train.jsonl` | 3,866 technically eligible, **unreviewed** lexical planning candidates |
| `exploratory/val.jsonl` | 679 technically eligible, **unreviewed** validation candidates |
| `sources/` | Thin source snapshots retaining all three hand tiers and translations |
| `review_sample.jsonl` | 43 inspectable cases spanning all shards, common/difficult features and known contextual examples |
| `review/index.html` | Local review interface with indexed source-video excerpts and exportable decisions |
| `review_queue.csv` | Complete review queue with stable row IDs and content hashes |
| `train_vocabulary.json` | Counts from eligible training targets only; no model token additions |
| `manifest.json`, `readiness.json` | Source hashes, task policy, split policy, exclusions and review readiness |
| `lsfb_cont_exploratory_colab.zip` | Compact text/tokenizer package for Colab preflight; explicitly exploratory |

299 training-source rows are quarantined because their conversation occurs in
the official validation source. This preserves both conversation and signer
separation for eligible target rows. Context may include the held-out partner's
French turns from that same conversation; those partner rows are excluded from
training. Previous dialogue contains no glosses or future translations.

## Representation

Model input uses the current French sentence plus up to four completed preceding
turns within 60 seconds, with relative `Moi` / `Autre` roles. Context clocks are
kept separate for exact recording-session prefixes. Broad conversation grouping
is used for split isolation. An ongoing/overlapping turn is not inserted as
completed context. If the T5 source budget is exceeded, the oldest context turns
are removed and recorded. The current sentence and target are never truncated.

Model target uses quoted operations, for example:

```text
SIGN "bonjour" ; SIGN "merci" ; FINGERSPELL "fk" "fs:fk(ok)"
```

This uses existing T5 subwords. The structured equivalent is retained as
`target_operations` in each candidate. JSON braces are not used in the model
target because T5's tokenizer encodes them as unknowns. Every eligible target
has zero unknown token IDs and an exact semantic tokenizer round trip.

Lexical signs use the author-provided lemma; a missing lemma is an exclusion,
not an inferred correction. Fingerspelling retains the raw annotation including
spelling hints; no letters are guessed. Genuine repetitions are retained.
Raw variations, specifiers, pointing, depictions, buoys and palm-up events remain
in source sidecars. Left/right tiers, simultaneous events and events rejected at
translation boundaries are included for review. This lexical projection alone
cannot specify full signing grammar, hand timing or a unique avatar motion.

Technical limits are 80 current French words, 32 lexical operations, 30 seconds,
512 source tokens and 512 target tokens. They define a tractable initial task;
excluded segments remain in the candidate corpus with reasons. They are not
classified as linguistically bad.

## Rebuild from local data

Use a new output directory. The command refuses to overwrite a frozen build.
No datasets or model weights are uploaded by these commands.

```bash
uv run --no-project --with tokenizers==0.22.2 python training/lsfb_cont/prepare.py \
  --root /home/jmal/Dataset/lsfb-cont \
  --tokenizer .runtime/lsfb-cont-tokenizer/tokenizer.json \
  --output .runtime/lsfb-cont-task-next
```

The tokenizer assets were downloaded with:

```bash
hf download google-t5/t5-base spiece.model tokenizer_config.json tokenizer.json config.json \
  --revision a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1 \
  --local-dir .runtime/lsfb-cont-tokenizer
```

## Review with source video

Open the generated `review/index.html`, or serve its directory locally:

```bash
python3 -m http.server 8092 --bind 127.0.0.1 \
  --directory .runtime/lsfb-cont-task-v3/review
```

Each excerpt includes eight seconds before the target and three seconds after
it, where available. This does not guarantee that all relevant conversation is
visible; the review must mark insufficient context when necessary. The page
shows the exact budgeted model input as well as the original annotations.

The reviewer supplies an identity, records source alignment/meaning/context/
representation checks, edits the proposed operation array where appropriate,
and downloads decisions. Decisions are held in page memory until downloaded;
download before closing the page. Automatic flags never approve a row.

Generate a different review pack from any candidate JSONL with:

```bash
python3 training/lsfb_cont/review.py \
  --root /home/jmal/Dataset/lsfb-cont \
  --candidates .runtime/lsfb-cont-task-v3/review_sample.jsonl \
  --output .runtime/lsfb-cont-task-v3/another-review --limit 50
```

Only individual recordings are temporarily read by verified TAR index offsets;
the whole archive is never unpacked. TAR headers and source-video duration are
checked. Generated clip failures are reported rather than hidden.

## Import decisions and release reviewed data

```bash
uv run --no-project --with tokenizers==0.22.2 python training/lsfb_cont/release.py \
  --dataset .runtime/lsfb-cont-task-v3 \
  --decisions /path/to/lsfb-review-decisions.jsonl \
  --output .runtime/lsfb-cont-reviewed-v1
```

The importer verifies frozen data hashes, task identity, row content hashes,
reviewer/time/check fields, corrected operation schemas, tokenizer fidelity and
split isolation. It rejects stale, duplicated, quarantined or technically
excluded acceptances. Correcting source alignment or limits requires a new
prepared build; review cannot silently override split quarantine. Reviewed
training **and** validation examples are required to create a release.

No linguistic approvals were fabricated during preparation. Approved counts
currently remain zero. That is a data-quality state, not a software error or an
extra authorization request.

## Colab handoff

Open `LSFB_CONT_Colab_Preflight.ipynb` in Colab and upload the prepared ZIP. It
backs up the ZIP to Drive and checks all file hashes, every eligible source/
target token length, target round trips and conversation separation. It does
not automatically launch another large training run.

Create a package from a future reviewed release with:

```bash
python3 training/lsfb_cont/bundle.py \
  --dataset .runtime/lsfb-cont-reviewed-v1 \
  --mode reviewed --output .runtime/lsfb-cont-reviewed-v1-colab.zip
```

Training uses the pinned T5-base checkpoint and its existing tokenizer, loading
`input_text` / `target_text`. First run a small learning gate and inspect generated
training examples, teacher-forced token accuracy, repetitions, invalid plans and
validation behavior versus retrieval. A short memorization gate is not a
translation benchmark. Scale only after the representation is actually learned;
evaluate lexical projections separately from full-signing references. The prior
raw annotation WER is not directly comparable to this different target task.

## Validation

```bash
uv run --no-project --with pytest --with tokenizers==0.22.2 python -m pytest \
  training/lsfb_cont/test_prepare.py -q
```

Tests cover source-preserving event selection, explicit fingerspelling, quoted
plan parsing, completed-context boundaries, conversation quarantine, review
provenance, corrected targets and the real pinned T5 tokenizer. Further end-to-end
checks are recorded in `research/LSFB_CONT_PREPARATION_RESULT.md`.

Source documentation: https://huggingface.co/datasets/ppoitier/lsfb-cont

Source-derived artifacts retain the corpus's CC-BY-NC-SA-4.0 attribution and are
local research outputs. No public redistribution was performed.
