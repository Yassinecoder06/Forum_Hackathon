# Local LSFB-CONT task preparation result

Prepared 15 November 2026 from `/home/jmal/Dataset/lsfb-cont`.

The current prepared dataset is `.runtime/lsfb-cont-task-v3/`. Implementation and
reproduction instructions: [`research/code/training/lsfb_cont/README.md`](../../code/training/lsfb_cont/README.md).

## Task and data

**French text + completed preceding dialogue → lexical LSFB operations**, using
readable T5 subwords with explicit fingerspelling operations. This is a lexical
projection, not full signing grammar or an avatar motion representation.

| Measure | Result |
|---|---:|
| Verified source train/validation TARs | 7 |
| Source recording records | 440 |
| Nonempty, valid-duration French segments retained as candidates | 5,354 |
| Technically eligible exploratory training rows | 3,866 |
| Technically eligible exploratory validation rows | 679 |
| Training-source rows quarantined for shared validation conversation | 299 |
| Rows with one or more technical exclusion reasons | 809 |
| Eligible training lexical lemma types | 2,588 |
| Added model tokens | 0 |
| Review sample cases / successful video excerpts | 43 / 43 |
| Linguistically approved training / validation rows | 0 / 0 |

“Eligible” means technical/source/tokenizer checks passed. It does not establish
semantic correctness. Every row remains pending linguistic review. No new T5
training was launched during this preparation task.

The model input uses relative participant roles and up to four preceding completed
French turns within 60 seconds. It excludes future/ongoing translations and all
gloss context. Old context is removed, with a recorded flag, when the source
would exceed 512 T5 tokens. This occurred for 76 eligible rows. Current French
text and target labels are never silently truncated. Eligible source/target
length maxima are 512/313 tokens respectively.

The previous `Voilà la différence.` example now includes preceding discussion
of Belgian/French signs for `pipi`. This repairs the missing context in its input;
it does not automatically approve the example. Some simultaneous conversation
turns cannot be included without future information, and must still be reviewed
for whether the available input is sufficient.

## Source and split checks

All seven annotated source TARs were hashed and matched to local Hugging Face
tree metadata pinned to revision `64b0c112f59d3710367fb736ef1808f37ff15e28`.
All three hand tiers, author lemmas, raw glosses, specifiers, variations,
translations, source IDs and times are retained in thin source snapshots.
Boundary-excluded events and separate hand tiers accompany each candidate.

Official validation target rows remain validation rows. Training conversations
present anywhere in the validation source are quarantined, so eligible targets
have no shared conversation IDs or signers. Exact final model inputs shared with
validation are excluded from training. The official test TAR was not opened by
this preparation pipeline. The prior experiment's test remains previously used;
no fresh test-set performance claim is made.

## Technical exclusions

Counts overlap; each source row is preserved with its reasons.

| Reason | Rows |
|---|---:|
| Conversation present in validation | 299 |
| More than 32 lexical operations | 278 |
| No lexical operations | 236 |
| No aligned sign events | 157 |
| More than 80 current French words | 125 |
| Longer than 30 seconds | 86 |
| More than 512 target tokens | 42 |
| Current input still exceeds 512 tokens after context removal | 6 |
| Target has unknown T5 tokens | 6 |
| Target fails semantic tokenizer round trip | 6 |
| Lexical event missing author lemma | 4 |
| Duplicate training input/target | 4 |

Nonlexical features, repetition and simultaneous hands are review flags, not
automatic errors or rejection criteria. All original data remains available.
The lexical projection deliberately omits nonlexical events from the **model
target only**; it does not erase them from the candidate records.

## Review and Colab handoff

The local review page is served at `http://127.0.0.1:8092/` while the local server
is running. Its file is `.runtime/lsfb-cont-task-v3/review/index.html`.
It shows exact budgeted model input, source-video excerpts, raw events and hand
tiers, editable lexical operations and four explicit review checks. Excerpts
are read from verified TAR index offsets, not a full video extraction. Decisions
are exportable and tied to immutable row content hashes.

`release.py` imports source-bound decisions, validates corrected target schemas
and tokenization, and requires accepted train and validation rows. No automatic
decisions or invented linguistic approvals were added.

The 1.3 MB exploratory Colab ZIP contains text splits, the pinned tokenizer,
source/task manifest and executable preflight checks. The companion notebook is
`research/code/training/lsfb_cont/LSFB_CONT_Colab_Preflight.ipynb`; it backs up the uploaded ZIP
to Drive and runs the checks. It intentionally does not start a large run.
The previous Colab session `signaculture-t5-atomic` was no longer available at
handoff, so no claim is made that the package is currently uploaded there.

## Verification performed

- Eleven new preparation/review tests and five existing scorer/preparation tests
  passed together: **16 passed**.
- Tests exercise explicit fingerspelling, plan parsing, exact-repeat handling,
  boundary retention, completed-context boundaries, conversation quarantine,
  review hashes/checks/corrections, T5 tokenizer fidelity and indexed TAR reads.
- Generated review-page JavaScript passed `node --check`.
- All 43 video excerpts generated successfully with source duration checks.
- The exported Colab package was unpacked locally and its own preflight executed:
  all package hashes, every eligible row's token limits and target round trip,
  and conversation isolation passed.
- No source files or previous model checkpoints were changed.

Next, review source meaning/context and run a small learning gate on the selected
representation before scaling training on Colab. The final quality of this
lexical task must be evaluated separately from the former full-event target.
