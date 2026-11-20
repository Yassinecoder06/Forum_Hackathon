# LSFB dataset and T5 failure assessment

Assessed 15 November 2026 against the frozen data, original annotation snapshots,
selected checkpoint outputs, and a new training-fit probe. No data was filtered,
no model was trained, and no checkpoint or decoder was tuned during this audit.

## Finding

The failed system has **two independent problems**: the sentence supervision is
not approved for the intended task, and the large atomic-output T5 experiment
has not learned even a sample of its own training examples adequately. Blaming
all errors on the original LSFB corpus, or on hardware, is not supported.

The source corpus is still valuable. The unsafe assumption was that temporally
overlapping annotations automatically form clean, context-free French-to-gloss
training pairs. The second unsafe assumption was that adding thousands of output
tokens provided translation knowledge or usable motion coverage.

## What was actually audited

- Every frozen training pair (4,396) and validation pair (714), before the fixed
  model length filter. The model used 4,360 and 706 respectively.
- All seven thin source shards: official CONT training shards 2–7 and validation
  shard 1. They contain 440 recording records in total.
- Original left/right/both-hand annotation tiers for all 63 records of local
  original shard 2, not only the reduced snapshots.
- All 435 official test predictions in each of the constrained and raw modes.
  Test data was used to diagnose the completed experiment, not to tune a new one.
- A new, read-only generation and teacher-forcing probe on 32 training and 32
  validation examples. Each split was sampled with seed 42: eight examples each
  with 1–4, 5–12, 13–30 and 31–96 target units, using the existing source limit.
  These are diagnostic samples, not new aggregate benchmark results.

The audit tool is `training/t5_rebuild/audit_dataset.py`. Detailed machine-readable
results and the 5,110-row review queue are in `.runtime/lsfb-full-audit/results/`.
The raw backups and full recovered predictions are in its sibling `source/`.

## 1. Download corruption and accidental row mixing are not established

All three frozen dataset file hashes match their manifest. Across all 5,110 rows:

- Zero rows lack the exact source translation text and interval.
- Zero stored signer IDs differ from the source record.
- Zero selected events are absent from their source recording.
- Zero canonical target sequences differ from the current normalizer output.

Source event timestamps have positive duration. The frame/ms consistency check
found no disagreement above 40 ms under the observed 50 fps convention. This is
an internal consistency check, not proof of alignment to the correct signer or
video. There are 102 empty raw gloss events; the pair builder excludes them.

On original shard 2, `both_hands` contains 22,417 distinct events, all found in the
union of left/right tiers. It contains no exact duplicate events. There are 28
union events absent from `both_hands`; their significance needs tier review.
It is therefore inaccurate to claim that we simply duplicated every two-handed
sign in our parser. Separate-hand tiers still matter for simultaneous signing.

## 2. Temporal matching is not semantic or discourse validation

`prepare.candidates` takes every event for which at least 80% of that event's
duration overlaps a French translation interval. It does not verify meaning,
speaker-turn correspondence, task type, or whether neighboring conversation is
needed to interpret that interval.

Concrete cases verified against the source snapshots:

- `CLSFBI1603A_S037_B:129292:129762`: French **“D’accord.”**, target **3 4 AGE**.
  The other participant in this conversation has a neighboring translation
  **“C’était il y a longtemps, j’étais petite ! Je devais avoir trois ou quatre
  ans.”** The correspondence is conversationally suggestive, but does not prove
  swapped tiers: echoing the other person's signs is also possible. Source
  video/original annotation tier review is required. Our context-free pair is
  not approved as a French-to-gloss translation example.
- `CLSFBI0211A_S004_B:265166:270426`: **“Voilà la différence.”** accompanies
  repeated **FRANCE / PIPI** signs. The immediately preceding source translation
  explicitly compares signs for “pipi” in Belgium and France. This is a
  context-dependent sign demonstration, not evidence that those signs mean
  the French phrase in isolation. Do not replace its target with an invented
  DIFFÉRENCE gloss.

Consequently, these examples do not justify declaring the original corpus
random or corrupt. They show why a natural signed conversation, its French
interpretation and its sign-by-sign annotations cannot be assumed to constitute
a context-free translation instruction dataset without additional preparation.

Among the six training source shards, 377 recordings contain 4,631 French
translation segments. Only 190 recordings have any French translation; 187 do
not. The builder finds no qualifying events for 144 translation segments. In
shard 6, 110 such segments concentrate in `CLSFBI3102A_S063_B`, requiring a
record-specific completeness/alignment investigation. They must not be silently
treated as verified empty translations.

326 training source intervals contain at least one intersecting event rejected
by the 80% threshold. This is a boundary sensitivity issue to review; it does not
establish that changing the threshold fixes semantic alignment. Tiny boundary
crossings are common and are not themselves errors.

## 3. The target mixes lexical translation with contextual performance

The 4,396 training rows contain 63,323 target events:

| Event type | Count |
|---|---:|
| Lexical | 50,165 |
| Pointing | 8,162 |
| Depictive | 2,916 |
| Palm-up | 1,783 |
| Buoy | 297 |

3,145 rows contain at least one nonlexical event. These events are legitimate
parts of signing; calling them noise or blindly deleting them would change or
damage the task. However, a context-free French sentence often does not specify
the discourse referent, spatial locus, exact depiction or signer-specific
performance needed to reproduce this particular annotated sequence.

1,430 rows contain overlapping event intervals that our target serializes in
start-time order. 1,349 contain consecutive repeated canonical units. Some
overlaps and repetitions are linguistically valid; neither count is an error
rate. The output representation needs a deliberate policy for simultaneous
hands, genuine repetitions and contextual annotation, verified on source data.

Our normalizer retains lexical lemmas but removes the fingerspelling marker
and other lexical specifiers from model targets. 311 training rows include
fingerspelling events. For example, raw `fs:fk(ok)` becomes `lexical:fk`, losing
the spelling hint `ok` as well as the operation marker. For an avatar renderer,
an ordinary sign and spelling a word require different actions. The original
metadata survives in source events, but the model output cannot express it.

Similarly, a generic `depictive:ds` label does not specify a unique motion.
There are 1,970 occurrences of this target in the training pairs. A one-label
to one-motion lookup cannot reconstruct the original depicted actions from
that label alone.

## 4. Output inventory is much larger than its supervision

The experiment added 6,743 ordinary tokens to T5. Only 4,103 target types occur
in the paired training data:

- 2,640 output units (39.2% of the inventory) never appear as positive targets.
- 1,507 observed types appear exactly once.
- 2,748 observed types (67.0%) occur at most five times.
- 477 validation target types never occur in paired training, accounting for
  782 validation target occurrences before length filtering.
- The inventory covers 93.5% of validation occurrences, but that only means
  an ID exists. It does not mean a translation mapping has been learned.

The vocabulary builder includes all training-shard annotations, including ones
without French translations, plus the ISOL inventory. This is allowed by the
split protocol, but those labels have no paired positive training examples.
The decoder still allows them at generation time. Initializing a new token
from the average of its label's subwords does not teach its translation usage.

The ISOL-500 training inventory actually observed in the chosen shards has
499 classes. Just 523 training pairs (11.9%) fit entirely inside that inventory;
164 of those have only one target unit. Only 354 have between 2 and 12 units.
The corresponding validation counts are 74 fully covered pairs and 41 with
2–12 units. Filtering to this inventory is therefore a small, different task,
not an automatic repair of the existing full corpus. Those rows still need
linguistic review. Larger ISOL variants exist, but their motion assets and
annotation-to-asset mappings have not been verified by this audit.

## 5. The selected model has not learned the full training task adequately

The small 11-example/13-unit memorization gate passed. It did not test the
6,743-token output space or the full sentence data. Scaling immediately from
that gate to the full experiment was insufficient validation.

New read-only probe on the selected model, with the same constrained greedy
decoding and 128-token cap:

| Diagnostic sample | Examples | Exact match | WER | Teacher-forced next-token accuracy |
|---|---:|---:|---:|---:|
| Training | 32 | 0% | 312.45% | 15.53% |
| Validation | 32 | 9.38% | 255.66% | 12.90% |

The sample is too small and deliberately stratified to estimate aggregate
quality or compare split difficulty. It is sufficient to show serious failure
on training examples, including low accuracy when the correct preceding tokens
are supplied. This is not merely unseen-signer generalization or unknown test
vocabulary. Full-run late training losses around 5.3–5.9 support the concern.

On the complete official test, constrained outputs contain 20,224 units versus
5,189 reference units. Independent edit alignment reproduces WER 392.37%:

- 15,602 insertions;
- 4,191 substitutions;
- 567 deletions.

Insertions account for 76.6% of edit errors under this deterministic minimum-edit
alignment (ties can change the operation breakdown). 212/435 predictions contain
repeated trigrams versus 23/435 references. Some short inputs produce a common
gloss more than 100 times. The data's mean training target length is 14.4, so
the model's 46.5-unit average test output is not simply copying average target
length. The model has a severe repetition/decoding collapse.

Raw decoding also fails (WER 357.18%; 201/435 invalid sequences). Constraining
the alphabet makes outputs syntactically valid; it does not repair translation.
Most outputs eventually emit EOS, so this is not simply a missing EOS label.
Imposing a shorter cap or forbidding repetition may reduce WER while preserving
incorrect meaning. Such changes must be compared on validation, not sold as
successful translation.

The relative effects of sparse new embeddings, optimizer settings, target
policy, noisy supervision and decoding require controlled ablations. There is
not enough evidence to attribute the entire failure to one hyperparameter.

## 6. Evaluation and the downstream application also need repair

Train and validation have no shared recording IDs, signer IDs or normalized
French inputs. However, **14 exact conversation prefixes are shared** across
participants: 287 training rows and 262 validation rows are in these sessions.
Signer-disjoint is not conversation-disjoint. This may inflate evaluation or
produce task dependencies; it cannot explain the very poor scores by itself.

For a claim about unseen conversations, split by the complete conversation,
grouping participants together. Keep official-split results separately labeled.
The official test has already been opened and inspected; it is no longer an
untouched development holdout for choosing the next experiment.

LSFB is Belgian sign language. The project's Hamburg-derived France-LSF assets
must not be treated as interchangeable with LSFB solely because their glosses
have similar French names. Text translation, motion lookup and avatar quality
need separate, language-consistent evaluations. More ISOL clips or a transition
GAN cannot fix incorrect French-to-gloss supervision.

## Recovery order

1. Preserve this failed experiment and its frozen source data as a baseline.
   Stop scaling the same preparation/training recipe.
2. Define the deliverable explicitly: a limited-domain lexical gloss prototype
   or context-aware full signing. Do not silently change task by deleting
   pointing, repetitions and depictions.
3. Build a source review set that includes short, long, fingerspelled, temporal
   boundary, simultaneous-hand and context-dependent examples from every source
   shard. Inspect original tiers/video and neighboring turns with an LSFB
   reviewer. Record accept/reject/corrected/context-required decisions. Existing
   automatic review flags are a queue, not approval.
4. Establish a small reviewed, coherent-domain dataset and a conversation-grouped
   validation set. Create a fresh holdout for the new development cycle where
   possible. Do not fabricate sentence pairs by concatenating ISOL labels.
5. On identical approved pairs, compare readable gloss strings using existing
   subwords with a small, adequately supported atomic inventory. Retain typed
   fingerspelling operations and contextual fields where required by the task.
6. Require a learning check at the actual vocabulary scale: generated training
   outputs, teacher-forced token accuracy, repetition/length statistics and
   validation performance against retrieval. Passing an 11-pair gate alone
   does not authorize a full run.
7. Scale only the approach that learns the reviewed training task and improves
   meaningful validation behavior. Select optimizer/decoding changes on that
   validation protocol. Confirm motion coverage and signing quality separately.

## Reproduction and evidence

```bash
python3 training/t5_rebuild/audit_dataset.py \
  .runtime/colab-results/t5-lsfb-atomic/data \
  .runtime/lsfb-full-audit/source \
  .runtime/lsfb-full-audit/results
```

Additional saved evidence: `hand_tier_audit.json`, `train_fit_probe.json`,
`train_fit_summary.json`. The read-only Colab probe source is backed up as
`train_fit_probe.py` alongside these results. The existing five unit-scoring and
preparation tests passed; the independent audit WER equals the saved test report.

Primary dataset documentation:

- [LSFB-CONT: temporal sign and French translation annotations](https://huggingface.co/datasets/ppoitier/lsfb-cont)
- [LSFB-ISOL: isolated sign clips, labels, metadata and vocabulary variants](https://huggingface.co/datasets/ppoitier/lsfb-isol)
- [Author's data loader](https://github.com/ppoitier/sign-language-data-loading)

These dataset cards establish structure and intended uses. The counts and
diagnoses above come from the local frozen experiment, not claims made by the
dataset author. No percentage of semantically incorrect pairs has been established.
