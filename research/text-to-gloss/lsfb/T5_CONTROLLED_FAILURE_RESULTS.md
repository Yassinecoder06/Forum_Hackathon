# Controlled failure experiments — 16 November 2026

Completed three additional 1,200-step experiments (3,600 optimizer steps), two decoding/shuffle comparisons, a phrase-overlap audit and metric regression checks. No official test data or human linguistic approvals were used. Existing adapters and original data were preserved. Machine-readable training results: `T5_CONTROLLED_FAILURE_RESULTS.json`.

## New vulnerabilities confirmed

1. **Repeated French inputs inflated the aggregate result.** Eight of51 validation rows share case/whitespace-normalized inputs with training. Normalizing punctuation too identifies12 shared rows, leaving39 with unseen French inputs. For the previously selected bounded checkpoint, exact match is8/12 on shared inputs but6/39 on unseen inputs. The reported14/51 was accurate for that split, but hid the difference in difficulty. The original conversation split remained intact; this is phrase overlap rather than identical source-record leakage.
2. **Sequence composition remains weak.** The bounded checkpoint gets13/27 single-sign cases exactly correct and only1/24 multi-sign cases. A result dominated by short dictionary-like cases must not be described as sentence translation success.
3. **Identical current French has conflicting targets.** Punctuation-normalized training groups include `Oui`, `D’accord`, `Tu as raison`, and `C’est tout` with multiple lexical projections. Dropping context can remove the information that distinguishes them. Different valid realizations and omitted nonmanuals are also possible. We cannot automatically declare the original annotations wrong.
4. **Free generation escapes the allowed inventory.** Fresh T5 experiments output labels outside the128-label candidate vocabulary on about10–15% of validation rows at later checkpoints. This is an executable-inventory failure, not proof every such string is an invented sign. Examples include `couvercle`, `premier`, `maternelle`, `direction`, and `super`; some reflect ordinary words in the input. A language model's plausible French word is not a verified corpus ID or a motion asset.
5. **Capacity and generalization are separate.** Increasing adapted layers substantially improves training fit without corresponding validation improvement. This demonstrates that capacity contributes to training fit; it does not establish the remaining labels or translation task as sound.
6. **A grammar constraint does not ensure meaning.** A training-phrase trie guarantees supported complete phrases, but can select a semantically incorrect phrase. It helped one model and hurt another. It cannot compose arbitrary new sequences and must be evaluated as constrained phrase selection.

## Experiment controls

All three fresh runs use pinned `google-t5/t5-base` revision `a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1`, seed73, BF16, batch8, learning rate3e-4, 1,200 optimizer steps, rank16 LoRA, JSON-array lexical targets and current French only. No gate warm start. Evaluations every200 steps; the same51 development rows are retained, with a fixed39-row unseen-input subset defined against the original361 training inputs. No tokenizer extension.

The ambiguity-filter experiment removes whole groups of training rows whose normalized input has multiple different targets, leaving307 rows. This is an intentionally tested exclusion policy, not linguistic cleaning or approval. The broad-layer run uses exactly those307 rows and adds `k`, `o`, `wi`, `wo` adapters to the q/v baseline. Comparing these two isolates adapter scope under that data policy. Both were worse on unseen inputs than the unfiltered control; deleting ambiguous groups indiscriminately is not supported as a fix.

The following table reports final step1200, not a claim of a selected release model:

| Experiment | Train rows | Training exact | All51 validation exact | Unseen39 exact | Unseen39 WER |
|---|---:|---:|---:|---:|---:|
| Fresh T5 q/v control | 361 | 39.89% | 35.29% | 20.51% | 87.95% |
| Exclude ambiguous groups, q/v | 307 | 34.53% | 21.57% | 7.69% | 146.99% |
| Same exclusions, broader layers | 307 | 68.08% | 25.49% | 10.26% | 115.66% |

No evaluated checkpoint in these three runs satisfied the strict selection requirement of zero malformed and zero outside-inventory predictions on the unseen subset, so `best` is null. Final adapters were saved for the latter two runs. The first fresh control did not save an adapter because its initial runner version saved only eligible checkpoints; its final measurements and history remain available. Do not describe any of these as a promoted best model. The previously saved bounded adapter remains available.

## Decoder and conditioning experiments

Using the existing bounded adapter, training-only candidate phrases and beam4:

| Decoder / source | All51 exact | Unseen39 exact | Unseen39 WER | Outside inventory |
|---|---:|---:|---:|---:|
| Greedy | 27.45% | 15.38% | 96.39% | nonzero |
| Unconstrained beam4 | 25.49% | 12.82% | 95.18% | nonzero |
| Complete training-phrase trie, beam4 | 29.41% | 17.95% | 85.54% | 0% |
| Same trie, cyclically shuffled French | 5.88% | 2.56% | 104.82% | 0% |

The correct-vs-shuffled difference supports genuine source conditioning. It does not establish sentence-level translation competence. The trie is built only from training targets; no validation reference phrases are inserted. These are development ablations, so the39 rows are no longer an independent fresh holdout.

For the broader-layer model, a trie built only from its retained307 training rows reduced unseen exact to2/39 and increased WER to165.06%. Thus, constraining a decoder is not sufficient to fix wrong semantic preferences or a poor candidate bank. Its shuffled-source run scored zero exact matches.

The standalone decoder probe uses batch4 and max128 new tokens, versus batch8/max96 during training evaluations. Minor greedy WER differences were observed for the broad-layer model; the table of training comparisons uses its original batch8 evaluation. Standardize these before any future release benchmark. Full decoded outputs/settings are retained in the run directories.

## Download assessment

Inspected the official FLAN-T5-base model card and pinned revision `7bcac572ce56db69c1ea7c8af255c5d7c9672fc2`. Configuration and tokenizer files were retrieved. The990,345,061-byte weight transfer was stalled/very slow through both the Hub client and direct HTTP, including a range-based attempt. Transfers were stopped without installing an unverified checkpoint. Partial files remain untrusted/incomplete; sparse allocated size is not proof of completed bytes. FLAN was **not trained or evaluated** and no improvement from it is claimed. Download logs are `.runtime/flan-*`. This network limitation did not prevent the cached T5 experiments.

Official card: https://huggingface.co/google/flan-t5-base

## What to retain and change next

- Retain the original bounded adapter for reproducible diagnostics; do not deploy these research outputs as accessible signing.
- Preserve conversation grouping and report normalized source overlap separately. Future fresh holdouts must be protected from phrase/paraphrase overlap and repeated checkpoint tuning.
- Preserve multiple references and discourse/nonmanual information during curation. Removing every ambiguous input group loses useful examples and did not help this experiment.
- Use a verified phrase bank and calibrated unsupported outcomes for a limited demo. Candidate constraints are an inventory safeguard, not an adequacy check.
- Test multi-sign phrases explicitly. A one-sign success must not dominate the sentence-composition score.
- Broader adaptation is feasible on this RTX4060: observed memory around1.5GB and no OOM. Larger models remain an experiment, not a demonstrated remedy for missing supervision.
- Curate source-video/pose-backed, self-contained multi-sign pairs in one signed language. The current lexical projection can omit depiction, pointing and nonmanual negation; its exact label sequence is not a complete meaning representation.

Checks:19 targeted tests passed, covering operation scoring, simultaneous repetition, source normalization, empty JSON predictions, outside-inventory labels, and missing prediction counts.

## Artifacts

- `.runtime/lsfb-controlled-t5-v2/{MANIFEST,history,RESULT}.json`
- `.runtime/lsfb-controlled-clean-v1/` — manifest, final adapter, results.
- `.runtime/lsfb-controlled-alllinear-v1/` — manifest, final adapter, results and `DECODING_PROBE.json`.
- `.runtime/lsfb-bounded-v1/SOURCE_OVERLAP_AUDIT.json` and `DECODING_PROBE.json`.
- Reusable runners: `research/code/training/lsfb_cont/experiments.py` and `decode_probe.py`.

These experiments found concrete failure mechanisms. They do not establish a state-of-the-art or linguistically validated translator.
