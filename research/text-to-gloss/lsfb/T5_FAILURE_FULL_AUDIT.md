# T5 failure audit and recommended direction

Assessed 16 November 2026. This audit inspects the completed local v3 run, frozen prepared examples, original hand-tier snapshots, application motion library, supplied T5–GAN paper, and primary dataset/research sources. It includes new local inference diagnostics and a text-retrieval baseline. It does not change the training data, production application, saved checkpoint, or historical scores.

## Decision

The main problem is an unreliable task construction combined with a model that has not learned stable free-running generation. There is a confirmed metric defect, a confirmed flattening of simultaneous hand annotations into sequential duplicate lexical operations, sparse supervision, and a language mismatch with the motion library. Hardware is not the demonstrated cause of these failures. Another large run with the same data and metric would be premature.

LSFB-CONT is useful and potentially compatible with **French-to-LSFB** research. Our automatic projection of it is not yet a verified translation dataset or a full signing plan. The corpus itself should not be declared corrupt based on errors introduced by our extraction, representation, or evaluation.

## Measured evidence

| Observation | Measurement | Interpretation |
|---|---:|---|
| Local training | 3,866 examples, 5 epochs, 4,835 optimizer steps | Completed without OOM |
| Full validation | 679 examples, 6,193 reference operations | All scores below refer to validation, not untouched test |
| Historical operation WER | 152.07% | Defective malformed-output handling; retain only as historical score |
| Historical exact plan match | 20/679 = 2.95% | Does not establish semantic adequacy |
| Invalid plans | 162/679 = 23.86% | Substantial structural failure |
| Complete SIGN chunks discarded by scorer inside invalid outputs | 6,147 | The metric conceals much of the overgeneration |
| Diagnostic WER recovering complete SIGN chunks and adding an invalid sentinel | 215.61% | Illustrates bias; this partial recovery is not an authoritative replacement metric |
| Empty-output control WER | 100% | A decoder emitting nothing beats the historical model WER |
| Current-French word TF-IDF retrieval, returning a training target | WER 112.74%; exact 1.18%; invalid 0% | Stronger WER than T5 under historical scoring, still inadequate translation |
| Distinct training operation labels | 2,813 | Includes full fingerspelling operation signatures; lexical lemma count alone is 2,588 |
| Labels occurring once / at most five times | 867 / 1,841 | Severe supervision sparsity |
| Validation operations unseen in training | 651/6,193 = 10.51% | Meaning of opaque unseen identifiers cannot reliably be inferred from spelling |
| Validation rows containing unseen operations | 304/679 = 44.77% | Explain performance separately on seen-only and open-vocabulary rows |
| Extra lexical events with identical interval and same lemma | 327 | Hand-tier projection can duplicate a simultaneous sign |
| Adjacent same-lemma pairs overlapping in time | 851, across 637 rows | Requires hand-aware inspection; not automatic permission to delete repetitions |

New generation probes use saved best adapter checkpoint 2,901, seed 108, 32 randomly sampled rows per split, FP16, greedy decoding, batch four, maximum length 320, cache enabled. These are small diagnostic samples, not confidence-qualified benchmark estimates.

| Probe | Historical operation WER | Exact | Invalid | Missing EOS at cap |
|---|---:|---:|---:|---:|
| 32 training rows | 145.25% | 1/32 | 7/32 | 13/32 |
| 32 validation rows | 139.66% | 0/32 | 7/32 | 10/32 |
| Same 32 validation targets with cyclically shuffled inputs | 153.10% | 0/32 | 7/32 | 10/32 |

The training sample shows that failure is not explained solely by unseen validation signs or a difficult held-out split. The shuffle degradation indicates some conditioning on input; it does not justify claiming the model completely ignores the source. A separate deliberate 32-pair overfit experiment has **not** been performed, so optimizer/adapter capacity versus supervision ambiguity has not been causally isolated.

Audit artifacts are in `.runtime/lsfb-failure-audit/`: `probe.py`, `probe_results.json`, `data_and_scoring.json`, `retrieval_baseline.json`, and dataset-card snapshots. Original run artifacts remain in `.runtime/lsfb-cont-local-v3/`.

## Confirmed preparation issue: simultaneity becomes repetition

For recording `CLSFBI0103A_S002_B`, time 5590–6270 ms:

- Left hand: `aller.1(7)`, lemma `aller.1`.
- Right hand: `aller.1`, lemma `aller.1`.
- Combined tier contains both events at that identical interval.
- `pair_events()` deduplicates `(start, end, raw gloss)`, so these survive because their raw glosses differ.
- `lexical_operation()` discards the variation and hand identity. The final target becomes two consecutive `SIGN "aller.1"` operations.

This target describes sequence where the source describes simultaneous activity. Same-lemma overlapping holds are another difficult case. The `genuine_repetition_requires_review` flag is only a flag: it does not establish that a repetition is genuine.

Repair the annotation representation before projecting to a sequence: preserve hand, lemma, variation, onset/end, and overlap groups. Establish a reviewed policy for jointly realized lexical events, dominant-hand events, holds and true repeated movements. Do not collapse all repeated labels or discard variants globally. Rebuild train/validation from source snapshots under a new version and inspect changes with source-video/pose excerpts.

## Confirmed evaluation issue: malformed output is rewarded relative to its content

In `research/code/training/lsfb_cont/train.py`, `score()` catches a parse failure and replaces the whole generated plan with `[]`. Its edit cost is then just deletion of the reference. Thus even a very long invalid output costs approximately 100% WER; a valid overlong repetition can cost substantially more. Since operation WER selects checkpoints, this creates a systematic incentive in model selection. We have not rescored all epoch predictions, so the true best checkpoint under repaired evaluation is unknown.

Keep syntax validity as a separate hard success criterion. Define a deterministic scoring protocol that preserves recoverable content and explicitly counts invalid spans, plus a raw serialized-token edit score that never drops malformed output. Report both, rather than presenting a recovery heuristic as gold standard. Include insertion/deletion/substitution counts, EOS termination, cap hits, repetition, lengths, unknown IDs and executable-motion coverage. Unit checks must include empty output, valid repetition, a missing closing quote, trailing garbage, fingerspelling and ordinary valid plans.

Exact match is strict because alternative sign realizations, repetitions and source annotation choices may express the same meaning. It remains useful for deliberate memorization and parser checks. Low exact match does not by itself prove semantic failure; the observed repetition and malformed outputs independently show real failure. Human LSFB/LSF comprehension evaluation is needed for accessibility claims. Back-translation can be a secondary diagnostic, but cannot replace that evaluation.

The split is a positive aspect: technical checks show conversation/signer separation, no silent target truncation and tokenizer round trips. There are 89 training conversations/41 signers and 32 validation conversations/6 signers. Preserve this separation, report scores by conversation/length/seen vocabulary, and quantify uncertainty by conversation rather than treating every conversational segment as independent. A 128-row validation subset is acceptable for initial screening but noisy for selection. Final 679-row scores are validation, not test; protect a documented fresh holdout when the protocol is fixed. Historical test inspection must be disclosed rather than relabeled unseen.

## Data compatibility and missing information

The corpus authors explicitly distinguish ID-glosses from French translations: labels are arbitrary identifiers, despite their French-looking spelling. `avec.poignet`, `pres.coude`, `bien.papier` and similar labels are not instructions to sign those component French words. T5 must learn the label semantics from examples or an explicit lexical database.

French translations are natural translations of signed discourse. Reversing them to recreate the exact original event transcript can be ambiguous: the translation can omit repetition, explicit pointing or interactional detail; it can add explanatory words; the signer can refer to previous visual events or interlocutors. Our temporal-overlap rule establishes a time association, not a verified minimal semantic unit. The context window helps some cases but does not supply all visual/discourse information. Some translations include interpreter notes or bracketed sign descriptions, which differ from ordinary application input. Audit those rather than treating them as user prose or declaring them source mistakes.

The lexical target omits nonlexical events from many examples: 2,608/3,866 training rows and 509/679 validation rows include omitted categories. Training aligned events include 5,574 pointing, 2,102 depicting, 1,349 palm-up and 185 buoy events. Validation includes 1,369 pointing, 412 depicting, 168 palm-up and 47 buoys. The target also omits timing, hand identity and facial grammar. These omissions define a limited lexical projection, not full translation. Adding a transition GAN cannot reconstruct omitted linguistic meaning.

## Model and training assessment

T5-base is a defensible baseline: its official model card includes French, German, Romanian and English. It is incorrect to explain this failure by claiming T5-base cannot process French. mT5 is another baseline worth comparing under identical data and evaluation, not an automatic fix. Prior failed mT5 or atomic-token experiments do not settle either architecture's potential.

The current experiment trains only rank-eight q/v LoRA adapters; the supplied paper describes fine-tuning T5-base with a much smaller 319-sign inventory. Adapter capacity, learning rate, trainable output embeddings, target format and training length should be tested only after a tiny fit gate. None is currently proved the principal cause. Current five epochs versus the paper's twenty is not evidence that adding fifteen epochs solves the problem: validation operation WER stopped improving after epoch three under the existing defective score.

The `SIGN "..." ; ...` target spends output steps on boilerplate. Teacher-forced loss can fall as syntax and frequent local transitions improve while free-running generation loops. Long unconstrained generation and sparse opaque labels increase the difficulty. Greedy decoding, beam search and repetition penalties are diagnostics; a global no-repeat constraint would forbid genuine signed repetition. Prefer a compact canonical lexical sequence, render syntax outside the model, and constrain candidates to a reviewed language-specific inventory. Unknown concepts should trigger an explicit unsupported result or reviewed fingerspelling, not substitution of a convenient sign. Preserve raw annotations as provenance, not a requirement that the model invent an exact annotation spelling.

RTX 4060 Laptop 8 GB supported this run; the process used around 2 GB GPU memory with no OOM. That establishes feasibility for this LoRA configuration, not a benchmark guarantee for larger models or full fine-tuning. Hardware limits speed and experiment scale; it does not explain hand-tier duplication, scorer behavior, or missing supervision.

## Downstream incompatibility

The active motion library contains 1,446 entries, all labeled `LSF`: 1,007 Dicta-Sign portal entries and 439 captured entries. The six-entry `data/glossary.json` is only the original fixture inventory, not the whole current library. The local model predicts Belgian LSFB identifiers. There is no demonstrated reviewed LSFB-ID-to-France-LSF-motion equivalence map. A French string match is insufficient to establish one. This matters even if T5 predictions improve.

Choose a target language for each experiment and preserve it through lexical IDs, motion assets, grammatical representation and review. For an LSFB experiment, extract/canonicalize LSFB motions from the aligned corpus or isolated counterpart. For the existing France-LSF app, train and review France-LSF outputs. Treat cross-language mappings as explicit reviewed translation data, not aliases.

## What other data would help?

| Resource | Appropriate use | Limitation for this experiment |
|---|---|---|
| Existing LSFB-CONT | Reviewed French↔LSFB segments, hand-aware annotations, context research, continuous motion | Requires task-specific projection/alignment review; pose-only unannotated data adds no French↔gloss labels |
| LSFB-ISOL | LSFB lexical motion inventory, sign recognition/embeddings, link IDs to pose clips | Single-sign clips do not teach sentence translation or grammar |
| Existing Dicta-Sign lexical clips | France-LSF motion retrieval and capture quality | A dictionary is not parallel sentence training data |
| Dicta-Sign-LSF-v2 | France-LSF continuous annotations/poses; assess available translation alignment | The cited 2020 paper is primarily lexical/nonlexical recognition, not a pretrained text-to-gloss translator; verify actual files and semantics |
| MEDIAPI-SKEL | French subtitles↔LSF poses, alignment/retrieval, motion work | Published description supplies 27 h of skeletons/subtitles, not a ready gold gloss sequence for every sentence |
| Rosetta-LSF | French↔LSF with gloss/context and AZee structure, designed for text-to-sign | Better representation fit; verify release size, access and conversion effort; not a large turnkey T5 training set |
| Signsuisse / WMT-SLT | Lexical clips plus example sentences for Swiss French Sign Language | Different variety (LSF-CH), sentence video does not imply gold sentence glosses; access/use restrictions apply |
| New project-specific pairs | Self-contained French, reviewed target IDs and complete signed example using the same inventory | Needs qualified signing review; pair count alone does not establish quality |

The old `/home/jmal/Downloads/mediapi-skel` and `/home/jmal/Downloads/dicta-sign-lsf-v2` paths were absent at audit time. Only `/home/jmal/Dataset/lsfb-cont` was found in the stated dataset directory. This does not prove no other copies exist elsewhere.

Pose extraction from more videos creates motion observations, not automatic semantic sentence labels. Likewise LLM-generated gloss pairs can augment reviewed examples, but cannot certify themselves or serve as an independent test set. Synthetic translation into a desired annotation scheme may simply teach the model that scheme's errors.

## Recommended alternative: retrieve complete meaning-bearing units, then learn a bounded planner

For a working prototype, exploit complete reviewed signed phrases rather than forcing a small model to recreate arbitrary conversational transcripts and stitching every word independently. Build a bank of short self-contained utterances from the chosen language, each linked to its French meaning, canonical IDs, original full motion and provenance. Retrieve by French intent/meaning, with confidence and explicit unsupported cases. Initially play the whole verified phrase to retain its original timing, coarticulation and nonmanuals. Semantic retrieval must be reviewed on held-out paraphrases: cosine similarity alone is not evidence of equivalence.

After that baseline works, train a compact T5 planner with retrieved compatible examples and a bounded language-specific inventory. A semantic/intent representation with explicit entities and reviewed templates can reduce what the model must infer. This is a proposal to test, not a claim of superior measured accuracy. Numbers, names and negation require explicit handling; do not expose unrestricted template recombination as validated signing. Use continuous LSFB data for the LSFB branch or retain France-LSF assets and gather matching phrase supervision for the app.

The supplied paper's 2,500 pairs cover 319 isolated signs, with expert correction. Our 3,866 conversational rows cover 2,588 lexical lemmas and additional fingerspelling operations. More rows do not imply a better supported task. The paper's GAN models short transitions; it cannot repair incorrect text translation, wrong sign IDs, missing fingers, facial grammar, or language mismatches. Its reported BLEU/WER cannot be directly compared without the exact split, representation and correction protocol.

## Ordered experiments before scaling

1. Freeze this run as a negative baseline. Repair evaluation, then rescore saved full-validation predictions without rewriting historical metrics. Rescore/re-generate checkpoints under that protocol before changing model-selection claims.
2. Rebuild hand-aware targets with a reviewed simultaneity policy and source alignment audit. Choose French→LSFB research or French→France-LSF application explicitly. Keep all original tiers and corrected-row provenance.
3. Establish 200–500 reviewed self-contained pairs as an initial curation target, including a protected held-out set, plus reviewed lexical meanings. This is a starting point, not a guaranteed sufficient sample size. Report how many original candidates are usable and why others are excluded.
4. Run a deliberate 32-pair memorization gate, training and generating on those same 32 pairs. Require at least 90% exact canonical IDs, valid structure and reliable EOS termination. This tests implementation learnability, not generalization. Include a copy task if it fails; inspect labels, padding, gradients, optimizer and saved/reloaded adapter consistency. Compare LoRA rank/modules to a small full-fine-tuned baseline only as needed.
5. Compare TF-IDF, semantic phrase retrieval and compact T5 on the same reviewed split. Compare context/no-context, plain IDs versus rendered operations, original T5 versus mT5, and only then adapter settings. Change one factor at a time. Include seen-only and unknown-vocabulary evaluation, and source-shuffle controls.
6. Add reviewed paraphrase augmentation to training only; group all source phrases and their paraphrases into the same split. Evaluate both novel intents and new paraphrases separately.
7. Run qualified signer comprehension review independently of automatic gloss metrics, then verify motion coverage/handshape/nonmanuals/retargeting. Improve motion generation after linguistic and inventory fidelity pass.

## Primary sources checked

- LSFB-CONT dataset card: https://huggingface.co/datasets/ppoitier/lsfb-cont
- LSFB-ISOL dataset card: https://huggingface.co/datasets/ppoitier/lsfb-isol
- Corpus authors on ID-glosses, depiction and the lexical database: https://www.corpus-lsfb.be/content3.php
- T5-base official model card: https://huggingface.co/google-t5/t5-base
- mT5 original paper: https://arxiv.org/abs/2010.11934
- Dicta-Sign-LSF-v2 original paper: https://aclanthology.org/2020.lrec-1.740/
- MEDIAPI-SKEL original paper: https://aclanthology.org/2020.lrec-1.743/
- Rosetta-LSF original paper: https://aclanthology.org/2022.lrec-1.529/
- Spoken-to-signed baseline paper: https://arxiv.org/abs/2305.17714
- WMT-SLT data specifications: https://www.wmt-slt.com/data
- User-supplied local paper: `research/paper-b/research_paper.pdf`; extracted text `.runtime/paper-reread.txt`; prior detailed review `research/paper-b/PAPER_B_VERIFIED_REVIEW.md`.

Research direction is informed by these sources; the proposed retrieval/intent/template experiments and thresholds are engineering recommendations, not reproduced published results.
