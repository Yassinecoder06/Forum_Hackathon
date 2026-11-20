# France-LSF construction and target audit

Date: 2026-11-17. This audit assesses whether a construction-supported T5 experiment can be interpreted on the current source-disjoint split. It does not change or open the official test targets.

## Source design

- France-LSF UD-STK: 108 sentences / 615 tokens according to the treebank card.
- Current derived split: 39 training sentences from one CUC022 recipe recording, 28 development sentences from the STK vacation narration, 41 retained test sentences not opened in this audit.
- UD documents the French text as free translation and explicitly lists `Parallel: no`. The labels are manually annotated French glosses for signs, but those do not form a strict parallel translation target.
- Same UD conversion convention only; no SUD or Belgian LSFB rows are included.

## Training-only retrieval baseline

A word 1–3-gram TF-IDF style overlap retriever selected the most similar current French training segment. All 28 development references, predictions, similarity and IDs are preserved in `reports/private/fsl-construction-audit-v1/retrieval_predictions.json`.

| Metric | Result |
|---|---:|
| Exact sequence | 0 / 28 |
| Gloss-token WER | 152.58% |
| Development gloss token occurrences | 194 |
| Reference token occurrences absent from training vocabulary | 28 |
| References whose entire token vocabulary occurs in training | 0 / 28 |
| References whose every adjacent gloss pair occurs in training | 1 / 28 |
| Reference sequences previously present in training | 0 / 28 |

These counts show vocabulary and construction support are both inadequate for a held-out T5 comparison. For context, the preceding unconstrained T5-base run also scored 0/28 exact and emitted repeated/invented strings. Constraining it to seen labels may eliminate unseen outputs but cannot recover the 28 OOV reference occurrences, unseen constructions, free-translation ambiguity or missing sign features. Therefore a new T5 run on this unchanged split is not a valid test of robust composition; it would mostly measure how the OOV labels are penalized. No constrained run was launched.

## Target information loss

Parsed the full original UD CoNLL-U file and matched the exact 39/28 research split without consulting its official retained test target values. The current target generator copies FORM only. Yet the source's manual FEATS/MISC annotations contain 251 feature occurrences across 39 training sentences and 273 across 28 development sentences. `NMM` occurs 98 times in train and 116 times in development. The feature family also includes spatial `Src`, `Tgt`, `Loc`, personal/situational transfer `PT`/`ST`, pointing/proform-related labels, `Exp`, gaze, hand usage and repetition. Seven `Reported` occurrences in development do not appear in the training split.

This audit records fields lost by the current target representation; it does not assert all such features are obligatory output for every segment. It confirms that the present flat gloss objective cannot reproduce the complete annotations from these records. Machine-readable results and per-sentence examples are under `reports/private/fsl-construction-audit-v1/source_feature_coverage.json` and `source_feature_examples.json`.

## Decision and next experimental gate

Do not spend another model-size or constrained-decoding run on the same unsupported split and call it construction generalization. First source enough France-LSF annotated sentences so that (a) target labels and recurring combinations occur in training, (b) the independent holdout contains constructions represented in training, (c) recording/conversation groups stay disjoint, (d) FORM plus selected feature layers can be evaluated under an explicitly defined convention, and (e) a qualified LSF reviewer verifies translation/annotation compatibility. Then compare retrieval, a simple alignment/lemmatization baseline and T5 under a frozen protocol. Until that gate passes, report the T5 results as feasibility failures, not definitive model-quality estimates.

No original labels were modified, no additional resources downloaded, and no test split inspected. Existing zero-review gate remains in force.
