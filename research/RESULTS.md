# SignaCulture: results of all experiments, approaches and limitations

Evidence inventory updated17November2026. This report separates measured results, historical author claims, failed attempts and proposed work. Source reports/checkpoints are linked below. Scores across different datasets/splits/tasks cannot be ranked as one benchmark. BLEU is0–100 unless a source explicitly uses another scale. WER can exceed100% because insertions count.

## Latest verified status

See [current implementation status](STATUS.md) for the complete requirement audit and [machine-readable evidence](CURRENT_RESEARCH_STATUS.json). France-LSF mT5-small and T5-base runs completed but both scored 0/28 development exact; mT5-base never trained and its transfer is incomplete. The narrow Belgian LSFB study has three T5-base seeds that beat current-text retrieval on aggregate WER over the same 23 previously inspected development examples, but all remain 0/6 multi-sign exact and are worse than retrieval on multi-sign WER. A matched context ablation found current-only input stronger than context on that same small split. Targets remain unreviewed lexical projections; no local qualified review exists. Actual video/Animics ingestion, experimental 3D studio/export and real emotion2vec execution work; natural signing and calibrated 3D quality remain unproven. Later dated sections below supersede earlier running/pending snapshots.

## Overall finding

No experiment has established a reliable general text-to-sign translator or linguistically validated3Dsigning avatar. Data/supervision alignment, sparse label coverage, language mismatch and evaluation failures have mattered more than GPU capacity. Local RTX4060 Laptop8GB supports the compact LoRA and motion experiments attempted. Available disk was about45GB at the latest audit. Colab was used for downloads, selected GPU experiments and verified Drive backups; network failures and ephemeral sessions remain practical limits.

## Dataset inventory

|Resource|What was actually available/applied|Limitation|
|---|---|---|
|LSFB-CONT Belgian|24source files scanned;1,619pose recordings;2,146train/385development paired lexical projections;2,823quarantined|Unreviewed lexical targets, free French translations, incomplete nonmanual/simultaneous grammar|
|LSFB-ISOL Belgian|Earlier isolated-sign PoC/data audits|Isolated labels do not supply continuous sentence translation supervision|
|GeoQuery-LSFB Belgian|250 original signed examples with DeepL French; 4,519 synthetic source records kept separate; clean 186 train/37 development graph study completed|Graph generation and reranking failed to beat retrieval; 27 source graphs quarantined; one signer and narrow domain|
|France-LSF UD/SUD|128source-annotated research pairs in mixed-convention historical package;106UD-only reverified,39train/28val/39test|Free translations, upstream says Parallel:no; zero locally approved translation pairs; conventions kept separate in new runs|
|SignVerse France-LSF|900fsl metadata IDs,810train/45val/45test;860mapped;7pose/VTT records sampled|40unmapped; no gold glosses; embedded language absent; body scores are indices; metadata/README schema mismatch|
|Rosetta/continuous Dicta-Sign|Metadata and existing terms inspected|Institutional eligibility unconfirmed; restricted content not downloaded; AZee is not glosses|
|Dicta-Sign portal/Commons lexical assets|Existing captured lexical library and application motions inspected|Dictionary videos are not sentence pairs; individual permissions and motion review remain required|

No silent merging of Belgian LSFB, France-LSF or Swiss varieties. Later explicit user instructions changed the active language between experiments; historical runs remain separate. Raw identifying records/poses are private and ignored by Git.

## Belgian text-to-gloss experiments

|Experiment|Measured evidence|Interpretation|
|---|---|---|
|Early isolated/Colab T5 and FLAN trials|Historical reports record zero-exact trials and plumbing checks|Some tiny fixtures verified execution only; not evidence of a successful translator|
|Broad T5-base CONTv3|Original scoring later audited|Malformed JSON and duplicate/simultaneous operations invalidated optimistic interpretations|
|Train-only memorization gate|>=90%training exact required and passed in recovery experiment|Implementation can fit examples; not generalization|
|Bounded128-label T5,361train/51val|14/51exact;13/27one-sign and1/24multi-sign|Normalized source overlap12/51; unseen39exact6/39; not reliable composition|
|Fresh q/v control,1,200steps|Train39.89%exact;all51val35.29%;unseen39exact20.51%,WER87.95%|No checkpoint passed strict validity/inventory gate; control final adapter not saved|
|Remove ambiguous groups,q/v,307train|Train34.53%;all51val21.57%;unseenexact7.69%,WER146.99%|Blind removal of ambiguous inputs did not help|
|Same307rows,broader LoRA|Train68.08%;all51val25.49%;unseenexact10.26%,WER115.66%|Better fit did not yield better generalization|
|Bounded greedy decoding|Unseen39exact15.38%,WER96.39%|Free generation left inventory|
|Bounded training-phrase trie,beam4|Unseen39exact17.95%,WER85.54%;shuffled input2.56%exact,WER104.82%|Conditioning matters; phrase constraints do not validate semantics or arbitrary composition|
|Broad-layer model,trie|Unseenexact2/39,WER165.06%|Constrained decoder can hurt adequacy|
|Fully extracted CONT T5-base,2,146train/385val|3epochs807steps;15/385exact3.90%;4/328multi-sign;WER240.60%;132/385malformed JSON|Confirmed failure despite verified pose extraction; seen-vocab12/207exact|
|Same385 current-French TFIDF baseline|9/385exact2.34%;WER113.92%|Lower error than broad T5; context-inclusive retrievalWER144.69%|
|ACL-inspired1encoder/2decoder Transformer|32/32train-only fit gate including24multi-sign;2,146train/385val|Architecture can memorize; full task still sparse|
|ACL-inspired corrected runv2|50epochs;selected26;14/385exact3.64%;0/328multi-sign;WER104.48%;BLEU0.258;chrF15.215;0cap hits|Worse WER than empty-output100%diagnostic; not usable despite improvement versus prior T5|

The ACL-inspired runv1 selected an epoch2degenerate model with224/385cap hits; nominal99.12%WER was misleading because long malformed gloss strings counted as a single substitution. Runv2 required cap-free checkpoint eligibility. Final train sample23/32exact71.88%versus weak validation demonstrates overfitting. Targets fit within68subtokens, below96generation cap, so reference truncation does not explain that failure.

## France-LSF model-size comparison

Same39UDtraining/28development examples, seed71,BF16,rank16LoRA,LR3e-4,effective batch8,validation-token-loss selection. No new official test use. This is a tiny published-annotation feasibility study; it cannot establish general translation capability.

|Model|Outcome|
|---|---|
|mT5-small|40epochs;selected40;validation loss3.5671;0/28exact;WER594.85%;BLEU0;chrF4.907;60%unknown prediction tokens;28/28cap hits|
|T5-base|Selected epoch16;validation loss2.6149;0/28exact;WER235.05%;BLEU0.248;chrF14.442;94.16%unknown prediction tokens;0cap hits|
|mT5-base|Minimal publicApache2.0weights downloaded in Colab;local transfer timed out/retried;no completed training result found at this report's preparation|

T5-base first attempts failed tokenizer preflight. Four training-derived annotation characters(~,Â,È,Ê)were added and embeddings trained/saved with LoRA. Two validation examples retained unseen subtokens; no validation-derived vocabulary expansion. This compares size, pretraining, tokenization and adaptation—not size alone. Existing earlier France-LSF mT5 runs are documented separately in research/runs and are not overwritten or promoted here.

## Motion experiments

|Experiment|Result|Scope|
|---|---|---|
|Legacy5point GAN|Cached benchmark reproduced:normalizedRMSE linear0.320293;quintic0.329619;GAN0.314898,1.68%lower than linear|10interior frames within isolated signs;not lexical coarticulation;native12.5/14fps|
|Legacy provenance tests|5passed/1failed:source JSON hash changed|Cached result reproducible;fresh source provenance not fully confirmed|
|SignVerse2Dinbetweening|125train/224val/198test windows,7videos,8masked frames at24fps|Known signer identity unavailable;official video splits preserved|
|Linear|MPJPE0.195204;fingertip0.342989;boundary velocity1.973686|Normalized2Dunits|
|Hermite|MPJPE0.267180;fingertip0.372063;boundary2.362187|Same heldout windows|
|Non-adversarial temporal network|MPJPE0.194308;fingertip0.342837;boundary1.995438|Reconstruction-only ablation|
|Adversarial network|MPJPE0.194355;fingertip0.342902;boundary1.995126|No improvement over reconstruction-only baseline|
|Full-loss GAN|MPJPE0.194276;fingertip0.342715;boundary1.992309|About0.48%position improvement,worse boundary velocity;no established superiority|

Local new motion training3.97seconds,62.12MiBpeak Torch allocation;one seed,very small convenience sample.16.88%heldout landmarks invalid/masked. Body index availability substituted explicitly for unavailable probabilities;invalid hand/face scores rejected. Projected bone consistency is not true3Danatomy. No30%improvement reproduced. Seven new preprocessing/leakage/loss tests passed.

## Avatar and application work

Existing Three.js avatar,retargeting,clip lookup,hand-contact corrections,webcam and video-capture infrastructure were inspected rather than replaced. Application previews and software tests in docs are engineering evidence,not fluent-signer comprehension studies. New synchronized body/hands/face2Dpreview generated. SignVerse XY+score is notXYZ;confidence is never depth. Guarded bridge rejects raw2Das3Davatar motion. No validated new lifting stage,end-to-end signing translation or grammatical facial-expression model was established. Emotion/rhythm controls cannot substitute for nonmanual grammar.

## Confirmed failure mechanisms and limits

1. Sparse supervision and weak coverage: broad Belgian training1,756labels,601singletons;307/2,054validation operations unseen.
2. Free French translations are not literal gloss transcripts;context,simultaneity,depiction,pointing and facial grammar affect meaning. Automatically removing every ambiguous pair reduced performance.
3. Model fit is distinct from heldout translation. Increasing adapted layers and training epochs improved fit without adequate composition.
4. JSON serialization,EOS behavior,repetition,outside-inventory glosses and malformed text are separate failures;all need scoring,not discarded predictions.
5. Earlier scoring dropped or mishandled invalid plans;corrected measurements supersede optimistic interpretation. WER-only selection can reward empty/degenerate outputs. Exact match misses valid alternatives;neither metric replaces signer review.
6. Conversation separation alone does not prevent repeated French phrases. Historic dev sets have repeated tuning exposure;they are not untouched benchmarks.
7. Dataset technical integrity does not establish semantic alignment or tracking correctness. Verified25,310pose-component comparisons withzero mismatches establish extraction fidelity only.
8. No qualified new LSFB/LSFreviewer approval exists. No accessibility,intelligibility,state-of-the-art or GAN-superiority claim is supported.
9. Partial downloads are not usable checkpoints;session expiration and network timeout must be distinguished from model/hardware failures. Some earlyFP16losses were nonfinite;BF16supported subsequent local training.

## Current narrow context experiment

Training-only fixed-point vocabulary:35labels,each>=3operations across>=2training conversations AFTER filtering;targets1–4lexical operations.145trainexamples(49multi-sign),23development(6multi-sign/17single-sign).Original conversation/signer/source-phrase separation retained;complete preceding French context used. Labels remain pending linguistic review;no approval fabricated. Low multi-sign holdout count limits inference.

Current-French char3–5gram TFIDFretrieval:7/23exact30.43%,WER63.33%,multi-sign0/6. Context-inclusive retrieval:2/23exact8.70%,WER123.33%. Empty outputWER100%. All vocabulary/IDF fitted on training only. T5-base context run is freshly initialized and trained with reversible plain labels;full23rowdevelopment selection. Historical development exposure remains disclosed. Outcome appended after completion;beating retrieval is required before claiming progress.

## Primary local evidence

- research/text-to-gloss/lsfb/T5_FAILURE_FULL_AUDIT.md
- research/text-to-gloss/lsfb/T5_CONTROLLED_FAILURE_RESULTS.md and .json
- research/text-to-gloss/lsfb/LOCAL_T5_RECOVERY_RESULT.md
- research/text-to-gloss/lsfb/LSFB_ONLY_SCAN_AND_TRAIN.md
- research/text-to-gloss/lsfb/ACL2023_TRAINING_RUN.md
- research/paper-b/PAPER_B_VERIFIED_REVIEW.md
- research/EXECUTED_RESULTS.md and fsl_dataset_audit.md
- research/text-to-gloss/fsl/FSL_MODEL_SIZE_COMPARISON.md
- research/private/*/RESULT*.json and source run histories
- .runtime/lsfb-narrow-context-data-v1/{SCAN_REPORT,RETRIEVAL}.json
- .runtime/lsfb-narrow-context-run-v1/{MANIFEST,history,RESULT}.json when completed

Muanji etal2026reported BLEU.4347,WER.2364,exact30%and GANMSE.021419 in their own setup;these remain authors' claims on unprovided artifacts,not our reproduced scores. ACL2023German/ASLresults motivate experiments but do not establish LSFBtransfer. SignVerse provides subtitle-to-pose weak supervision rather than gold text-to-gloss labels.

## Historical run metric appendix

Directly extracted from saved JSON; no rerun or new linguistic approval implied. These include older France-LSF mT5 and concept-planner approaches omitted from the newer controlled tables above.

### mt5-small-lora-long-validation-selected

Source: `research/runs/mt5-small-lora-long-validation-selected/metrics.json`.

```json
{
  "epochs": 21,
  "best_epoch": 11,
  "counts": {
    "train": 61,
    "val": 28,
    "test": 39
  },
  "metrics": {
    "val": {
      "retrieval": {
        "exact_match": 0.0,
        "token_wer": 1.1443298969072164,
        "corpus_bleu": 0.0,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.0,
        "reference_oov_vs_train": 0.8144329896907216,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 28,
        "reference_tokens": 194,
        "predicted_tokens": 155
      },
      "lexical": {
        "exact_match": 0.0,
        "token_wer": 1.8762886597938144,
        "corpus_bleu": 0.1710565357228816,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.896457765667575,
        "reference_oov_vs_train": 0.8144329896907216,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 28,
        "reference_tokens": 194,
        "predicted_tokens": 367
      },
      "model": {
        "exact_match": 0.0,
        "token_wer": 4.185567010309279,
        "corpus_bleu": 0.0,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.5360303413400759,
        "reference_oov_vs_train": 0.8144329896907216,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 28,
        "reference_tokens": 194,
        "predicted_tokens": 791
      }
    },
    "test": {
      "retrieval": {
        "exact_match": 0.0,
        "token_wer": 1.2087378640776698,
        "corpus_bleu": 0.3009639808175593,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.0,
        "reference_oov_vs_train": 0.8252427184466019,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 39,
        "reference_tokens": 206,
        "predicted_tokens": 227
      },
      "lexical": {
        "exact_match": 0.0,
        "token_wer": 1.9320388349514563,
        "corpus_bleu": 0.19352402920994116,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.8673218673218673,
        "reference_oov_vs_train": 0.8252427184466019,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 39,
        "reference_tokens": 206,
        "predicted_tokens": 407
      },
      "model": {
        "exact_match": 0.0,
        "token_wer": 5.179611650485437,
        "corpus_bleu": 0.04220886971265283,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.6925996204933587,
        "reference_oov_vs_train": 0.8252427184466019,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 39,
        "reference_tokens": 206,
        "predicted_tokens": 1054
      }
    }
  },
  "limitations": "ENGINEERING PROOF ONLY; free translations; tiny single-source corpus; no local linguistic approval; no SOTA claim"
}
```

### mt5-small-lora-expanded-slow

Source: `research/runs/mt5-small-lora-expanded-slow/metrics.json`.

```json
{
  "epochs": 8,
  "best_epoch": 8,
  "counts": {
    "train": 61,
    "val": 28,
    "test": 39
  },
  "metrics": {
    "val": {
      "retrieval": {
        "exact_match": 0.0,
        "token_wer": 1.1443298969072164,
        "corpus_bleu": 0.0,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.0,
        "reference_oov_vs_train": 0.8144329896907216,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 28,
        "reference_tokens": 194,
        "predicted_tokens": 155
      },
      "lexical": {
        "exact_match": 0.0,
        "token_wer": 1.8762886597938144,
        "corpus_bleu": 0.1710565357228816,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.896457765667575,
        "reference_oov_vs_train": 0.8144329896907216,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 28,
        "reference_tokens": 194,
        "predicted_tokens": 367
      },
      "model": {
        "exact_match": 0.0,
        "token_wer": 1.1082474226804124,
        "corpus_bleu": 0.0,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.9864864864864865,
        "reference_oov_vs_train": 0.8144329896907216,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 28,
        "reference_tokens": 194,
        "predicted_tokens": 74
      }
    },
    "test": {
      "retrieval": {
        "exact_match": 0.0,
        "token_wer": 1.2087378640776698,
        "corpus_bleu": 0.3009639808175593,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.0,
        "reference_oov_vs_train": 0.8252427184466019,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 39,
        "reference_tokens": 206,
        "predicted_tokens": 227
      },
      "lexical": {
        "exact_match": 0.0,
        "token_wer": 1.9320388349514563,
        "corpus_bleu": 0.19352402920994116,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.8673218673218673,
        "reference_oov_vs_train": 0.8252427184466019,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 39,
        "reference_tokens": 206,
        "predicted_tokens": 407
      },
      "model": {
        "exact_match": 0.0,
        "token_wer": 1.0,
        "corpus_bleu": 0.0,
        "bleu_signature": "sacrebleu 2.6.0; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100",
        "predicted_oov_vs_train": 0.975,
        "reference_oov_vs_train": 0.8252427184466019,
        "motion_registry_oov": null,
        "motion_registry_note": "No approved source-gloss-to-motion mapping; train vocabulary OOV is separate",
        "examples": 39,
        "reference_tokens": 206,
        "predicted_tokens": 40
      }
    }
  },
  "limitations": "ENGINEERING PROOF ONLY; free translations; tiny single-source corpus; no local linguistic approval; no SOTA claim"
}
```

### pretrained-concept-planner

Source: `research/runs/pretrained-concept-planner/metrics.json`.

```json
{
  "kind": "SOFTWARE_CONCEPT_PLANNING_NOT_LSF_TRANSLATION",
  "metrics": {
    "model": {
      "examples": 24,
      "ordered_exact": 0.5416666666666666,
      "micro_precision": 0.8636363636363636,
      "micro_recall": 0.6785714285714286,
      "micro_f1": 0.76,
      "critical_flag_recall": 1.0,
      "critical_flag_hits": 6,
      "critical_flag_total": 6,
      "unknown_term_coverage": 1.0,
      "abstentions": 0,
      "latency_mean_seconds": 7.7954437012907265,
      "latency_median_seconds": 6.770297874998505,
      "latency_max_seconds": 23.744482606998645,
      "latency_first_seconds": 23.744482606998645,
      "per_category_ordered_exact": {
        "exact": 0.0,
        "lemma": 1.0,
        "multi_exact": 1.0,
        "multi_semantic": 0.3333333333333333,
        "named_entity": 0.5,
        "negation": 1.0,
        "number": 1.0,
        "question": 0.0,
        "synonym": 0.5,
        "unknown": 0.5
      }
    },
    "literal_existing": {
      "examples": 24,
      "ordered_exact": 0.5,
      "micro_precision": 0.85,
      "micro_recall": 0.6071428571428571,
      "micro_f1": 0.7083333333333333,
      "critical_flag_recall": 0.0,
      "critical_flag_hits": 0,
      "critical_flag_total": 6,
      "unknown_term_coverage": 1.0,
      "abstentions": 0,
      "latency_mean_seconds": 0.04191068216611408,
      "latency_median_seconds": 0.003111502999672666,
      "latency_max_seconds": 0.9340005139965797,
      "latency_first_seconds": 0.9340005139965797,
      "per_category_ordered_exact": {
        "exact": 1.0,
        "lemma": 0.6666666666666666,
        "multi_exact": 1.0,
        "multi_semantic": 0.0,
        "named_entity": 0.5,
        "negation": 1.0,
        "number": 1.0,
        "question": 0.0,
        "synonym": 0.0,
        "unknown": 1.0
      }
    },
    "literal_shared_guards": {
      "examples": 24,
      "ordered_exact": 0.4166666666666667,
      "micro_precision": 0.875,
      "micro_recall": 0.5,
      "micro_f1": 0.6363636363636364,
      "critical_flag_recall": 1.0,
      "critical_flag_hits": 6,
      "critical_flag_total": 6,
      "unknown_term_coverage": 1.0,
      "abstentions": 0,
      "latency_mean_seconds": 0.04191068216611408,
      "latency_median_seconds": 0.003111502999672666,
      "latency_max_seconds": 0.9340005139965797,
      "latency_first_seconds": 0.9340005139965797,
      "per_category_ordered_exact": {
        "exact": 0.0,
        "lemma": 0.6666666666666666,
        "multi_exact": 1.0,
        "multi_semantic": 0.0,
        "named_entity": 1.0,
        "negation": 1.0,
        "number": 1.0,
        "question": 0.0,
        "synonym": 0.0,
        "unknown": 1.0
      }
    }
  },
  "limitations": [
    "Authored software expectations, not LSF ground truth or STK test",
    "Only24cases; no general French or signing quality claim",
    "Deterministic linguistic guards are incomplete on unrestricted language; manual review always required",
    "Model/guard failures preserved; no tuning on benchmark results"
  ]
}
```

## Completed narrow LSFB context result

25epochs/475optimizer steps completed locally in349.85seconds. Selected epoch14adapter was freshly reloaded before scoring all23development rows.

|Method|Exact|GlossWER|Multi-sign exact|Multi-signWER|
|---|---:|---:|---:|---:|
|Current-French TFIDFretrieval|7/23(30.43%)|63.33%|0/6|61.54%|
|Context-inclusive TFIDFretrieval|2/23(8.70%)|123.33%|0/6|92.31%|
|ContextT5-base|10/23(43.48%)|56.67%|0/6|76.92%|

The model beats retrieval on aggregate development WER by6.67percentage points and exact by3examples, but its multi-signWER is WORSE than current-French retrieval. This is a narrow single-sign-dominated improvement, not established sequence-composition progress. All10exact cases are single-sign. No unfinished or empty outputs. No significance or fresh-test superiority claim;23rows,6multi-sign,previously inspected conversations and checkpoint selection on those rows limit inference. Context's causal benefit is unproven without a matched current-only T5ablation. No vocabulary expansion or deployment justified by this result.

Seven targeted checks passed (five existing scorer tests,two new tests for train-only repetition/context/split separation and empty/repeated-token scoring). Compilation passed. Local copied data/model/history/predictions and SHA256receipt: `/home/jmal/Dataset/lsfb-narrow-context-v1/`. Selected adapter requires the pinned T5-base checkpoint. Source poses remain in the original usableLSFBfolder;the new package references them rather than duplicating them.

Next best step: acquire/review more context-aligned multi-sign examples for these labels, reserve a genuinely fresh conversation holdout, and test a matched context/current-only ablation. Keep retrieval as the multi-sign benchmark. No automatic label correction or new human approval was invented.

## Matched context ablation completed

Same145train/23development examples,seed84,T5-base rank16LoRA,25epochs/475steps,BF16,batch2/accumulation4,LR2e-4schedule,plain lexical labels and validation-WER selection. Current-only removes preceding French dialogue;task prefix and current French remain. This is the only intended input change. Each arm selects its own best checkpoint under the same rule. No new official test labels opened.

|Input|Exact|WER|Multi-sign exact|Multi-signWER|
|---|---:|---:|---:|---:|
|context|10/23|56.67%|0/6|76.92%|
|current|12/23|46.67%|0/6|69.23%|

Paired exact counts:{"both_correct": 10, "context_only_correct": 0, "current_only_correct": 2, "both_wrong": 11}. Small repeatedly inspected development set,one seed and six multi-sign examples prevent claims of robust or statistically established context benefit. Input lengths change padding and dropout random draws,so identical seed is a control rather than bitwise-identical optimization. Full paired predictions are private in research/private/lsfb-context-ablation-v1; no human semantic approval. Current-only adapter and result backup saved alongside previous context run under /home/jmal/Dataset/lsfb-narrow-context-v1/.

Interpretation: current-only selected epoch11,12/23exact52.17%,WER46.67%,no empty/unfinished outputs;context selected epoch14,10/23exact43.48%,WER56.67%. Current-only reduces WER by10percentage points versus context and16.67points versus current-text retrieval. All12exact outputs remain single-sign. Multi-signWER69.23%current-only versus76.92%context and61.54%retrieval. Do not claim sentence-composition improvement or universally reject dialogue context. Five targeted tests passed; the input-length audit found mean173.19context tokens versus30.68current-only tokens.

Review queues prepared:1,869eligible Belgian multi-sign training examples with original context/events/timestamps/pose references;only49fit the35labels and all49are already trained. Additional23quarantined train examples fit these labels but carry fragment/boundary/simultaneity/source-overlap flags;none admitted. Qualified review and a new independent holdout remain outstanding.

## Multi-sign train-only fit gate and pipeline recheck

Warm-started the selected current-only narrowT5adapter,trained only the49multi-sign TRAIN examples with distinct current French inputs;seed91,LR3e-4,BF16,batch8,no dropout/weight decay,up to600steps,first>=95%training exact chooses checkpoint. No validation/test labels participate in this gate. At100steps:48/49exact97.96%,WER0.83%,zero unfinished/empty predictions,60.83seconds. This establishes memorization capacity for these targets,not translation competence.

A separate one-shot developmental evaluation after freezing that checkpoint scored4/23exact17.39%,WER143.33%,multi-sign0/6exact and107.69%WER,zero unfinished outputs. This failed variant is not promoted. Uniform multi-sign-only adaptation with these settings caused loss of aggregate performance despite excellent fit. Original current-only12/23exact46.67%WERcheckpoint retained. The experiment changes learning rate/dropout/distribution together,so it does not isolate which caused degradation.

Runtime evidence and source:research/code/training/lsfb_cont/multisign_fit_gate.py,evaluate_multisign_fit.py;.runtime/lsfb-multisign-fit-v1/RESULT.json and DEVELOPMENT_RESULT.json. Copied data/adapter/predictions with hashes to /home/jmal/Dataset/lsfb-narrow-context-v1/multisign-fit-diagnostic/.

Application recheck:frontend production build and motion/avatar tests passed;22backend API/capture tests passed. Existing installed Chrome used after Playwright's bundled binary was absent. Real browser checks passed composition,seeking,WebMdownload,actualWAV/pause-aware timing,JSONcapture/provenance/persistent lookup,unknown handling,mobile layout and offline errors. New browser WebM341,196bytes at research/private/pipeline-browser-recheck/preview.webm. This verifies software behavior,not sign-language comprehension.

Local emotion2vec+seed invoked directly on8seconds of existing French narrated demo audio,CPUinference completed with9finite category scores andneutral0.99255. Model-loading log reports missing AUDIO.decoder weights;this needs architecture/checkpoint interpretation before asserting exact weight coverage. These category scores are unvalidated hypotheses;no emotion-label accuracy study conducted. Result:research/private/emotion2vec-recheck.json;log:reports/emotion2vec_recheck.log.

Real video ingestion recheck completed:Bonjour61sampledframes/62outputframes,100%pose/face,left-hand100%,right-hand52.5%,motion schema validated. Source/output hashes and actual quality saved in reports/BONJOUR_CAPTURE_RECHECK.json. MediaPipe depth is estimated rather than calibrated3D. See research/text-to-gloss/lsfb/PIPELINE_AND_MULTISIGN_RECHECK.md for exact checks,commands and remaining goal gaps.

## GeoQuery-LSFB graph audit and retrieval experiment

Reused sourcecommit879731568a5cd93177d11bf742740b187176c99a locally. Provenance correction:Frenchinputs are DeepLtranslations per upstreamREADME;original signed examples derive from recorded native signing andHamNoSystranscription. Earlier`human_train`filenames and`synthetic:false`do not mean manually authored French. Corrected source/input axes are preserved in derived records. Expanded4,519synthetic release not trained.

Original203train/47valsplit has disjoint type IDs andzeroentity-normalizedFunQLstructure overlap.27/250graphs contain temporal edges to nonexistent articulation IDs;17train/10valquarantined without guessing repairs. Clean package186train/37val,separate fromCONT,withzero normalizedFrench overlap across splits.14train/3valnormalizedFrench groups map to multiple authored graphs;could represent alternatives,not automatically labeled errors.48/564valarticulation occurrences haveHNSstrings unseen in training.

Original47reference graph retrieval:characterTFIDF0/47exact,22.50%tripleF1,36.66%articulationF1;word/bigramTFIDF0/47exact,22.67%tripleF1,35.35%articulationF1. All47references retained,including10structural failures. Predictions include3and2invalid sourcegraphs respectively. These are graph-annotation overlap metrics,not glossWERor human comprehension.

Four new graph tests passed. Local graph review tool created with dangling-edge highlighting and notes export;no approvals fabricated. Prepared dataset /home/jmal/Dataset/geoquery-lsfb-graph-feasibility-v1/. Details and commands:reports/GEOQUERY_LSFB_ASSESSMENT.md. No graphT5training or3Davatar integration claimed.

## GeoQuery graph codec and generation gate

Implemented reversible full temporal-graph serialization withtrain-only241entryHNSdictionary andUnicode literal fallback. All223sourcegraphs andT5tokenizer roundtrips exact;0unknown subtokens. Train targetsmean244.12/max536;developmentmean331.46/max1196(oneabove1024). No truncation. Original hand roles,occurrence names,HNS,temporal relations andtripleorder retained. Six graph tests passed.

First freshT5-base rank16LoRA,BF16,seed92,LR5e-4,8training examples,max300updates. At300:2/8exactgraphs,6/8structurallyvalid,0capfailures,277.68seconds. Declared threshold7/8exact+8/8valid not met. No heldout evaluation/full research/code/training/promotion. A bounded600update extension began with identical seed/settings and fresh initialization;pending result at this report update. Missingfailedweights inv1explicitly documented;runner now saves failure checkpoints. Detail:reports/GEOQUERY_GRAPH_CODEC_AND_FIT.md.

## GeoQuery graph fit passed and full-data experiment launched

Extended same-seed gate passed at400updates:8/8exactgraph,8/8structurallyvalid,0capfailures,370.91seconds. A new process reloaded the adapter and exactly reproduced one selected original training graph. This verifies serialization/weight reload/training-set learnability,not generalization. Gate adapter/tokenizer/HNSdictionary and hashes backed up under /home/jmal/Dataset/geoquery-lsfb-graph-feasibility-v1/fit-v2/.

Full-data feasibility run launched:186train/37development,DeepL French input,full authored LSFBgraphs,no synthetic expansion,no gold-gloss conflation. Warmstart from passed8TRAINexampleadapter;seed93,BF16,effectivebatch4,LR2e-4,max8epochs,developmenttoken-loss earlystop,explicit1,280targettokenbudget coversallreferences. Model/optimizer/RNGresume andsourcehash checks implemented. Full37graphdecoding andsame-splitretrievalbaselines remain pending. Outputs remain unvalidated candidates.

Studio language provenance fix: new capture registration requires explicitfsl;missing/sfb/ssr/und declarations rejected,raw preview preserved. New captures carry declared_by_uploader_pending_review status,not human approval. Registry excludes explicitnon-fslmetadata;legacyrows retain prior historicalhandling,not automatically certified. UI offers language selection/reset fornewextractions.24backendchecks,frontendbuild/lint andupdatedrealbrowsercomposition/export/import/lookupsmoke passed. APIcontract andbackendREADME updated. Browser initially caught a selectaccessible-nameissue;explicitaria-label fixed it before rerun passed.

## Full GeoQuery graph experiment completed

EightTRAINexamplegraphgate passed400updates8/8exact+valid;freshprocessinferenceverifiedoriginalgraph. Full186train/37developmentrun then warmstarted thatadapter:4epochs188updates,selectedepoch1bytokenloss,236.48seconds,772.46MiBpeakTorchallocation. All37references retained;longest1196tokensfitexplicit1280budget.

T5:0/37exactgraphs,1/37valid,2.49%stricttripleF1. Sameclean-splitcharacterretrieval:0/37exact,37/37valid,22.96%F1;word/bigramretrieval22.04%F1.15repeatedIDand21danglingedgefailures,no generationcaps. Everymalformedoutputretained;separateserializedsubtokenerror75.20%(notglossWER). Validationlossworsened2.1156→2.2346;earlystoppedafter4epochs. Adapterretainedasfailedfeasibility,noavatarintegrationorlinguisticapproval.

Loadableresumestateandverifiedlocalbackup:/home/jmal/Dataset/geoquery-lsfb-graph-feasibility-v1/full-graph-v1/. Seven codec/metric tests passed;fullresumeequivalenceuntested. Detail:reports/GEOQUERY_FULL_GRAPH_RESULT.md. Nextapproachmustaddressstructuralconsistencywithoutconfusingvalidsyntaxwithmeaning.

## Frozen graph reranking and centralized run metadata

On the same186train/37developmentgraphsplit,top8traininggraphcandidatesretrievedbycharTFIDFwere rescoredbyfrozenselectedT5epoch1meanteacherforcedloss. Completegraph0/37allmethods;validity37/37bycandidateconstruction. RerankertripleF117.67%versusretrieval22.96%;cyclicshuffledFrenchsamebank18.16%. ArticulationF127.37%versus36.53%retrievaland27.68%shuffled. No demonstratedsemanticrerankingbenefit;validityisnotadequacy. Oraclemeanper-examplebestcandidateF134.41%usesreferencesonlyposthocandisnotdirectlycomparabletocorpusmicroF1. Runtime14.26seconds,noadditionaltraining,noofficialtest. Frozenbankcannotproduceunseengraphstructures;notatranslatorreplacement. Details:reports/GEOQUERY_GRAPH_RERANKING.md.

Registered6completedrecentexperimentsinexistingSQLitecatalogwithlanguage(fsl/sfb),representation,datasethashes,resultdigests,configs,metricsandcheckpointpaths. Source/pair/review/annotationtablesverifiedunchangedbyexactordered-rowhashesandlegacycatalogtests. Noapprovalsfabricated. Nineledger/graphtestsandeightlegacycatalogtestspassed. Reports:RECENT_EXPERIMENT_CATALOG.md/.json. Modelcorrectness/access/reviewerrequirementsremainunfulfilled;goalnotcomplete.

## Research studio integration

The Research panel now displays six recent cataloged experiments with explicit France-LSF/Belgian LSFB filters, representation/task labels, counts, WER or graph F1, and multi-sign/validity details. No recent diagnostic model is enabled for avatar inference. Different tasks and inspected development splits are explicitly not ranked. Missing or changed evidence is shown without measures.

Read-only API checks result hashes, metric consistency and language declarations before showing scores. Private filesystem paths and source sentences are excluded. Combined summary/API suite:22 tests passed; frontend build/lint and actual Chrome desktop/mobile/filter/privacy/no-inference checks passed. A cramped table was replaced with cards after visual review. No model quality or human approval is implied. Detail:reports/RESEARCH_STUDIO_LEDGER.md.

Verified mT5-base transfer update: Colab server reports no active sessions and pruned the stale local session. No local retry process was found. The partial file remains incomplete and is not a usable checkpoint; no mT5-base training result exists. Exact byte counts and observation are recorded in reports/MT5_BASE_TRANSFER_STATUS.json. A new download session would be needed before continuing that optional comparison.

## Additional supervision audit and narrow LSFB replication protocol (2026-11-17)

Public `pedroodb/glosl-lsfb` revision `754fc076b3239c04dbaf8c63d0c8b6df92d0de5a` was inspected using metadata downloaded through Colab. Its actual CSV contains 5,436 French text rows, zero nonempty gloss targets and zero populated gloss-type labels. It is a conversion of Belgian LSFB-CONT, not new independent France-LSF supervision. No pose/video bundles were downloaded or admitted. CC BY-NC-SA 4.0 is declared by the distributor; underlying redistribution permission and coverage discrepancies were not independently resolved. Matignon-LSF, STK motion capture and 40brèves papers/catalogs did not establish a currently accessible set of additional verified flat sentence-gloss pairs in this bounded audit. This is not a claim that no such data exists. See research/datasets/ADDITIONAL_SUPERVISION_AUDIT.md and data/audit/additional_supervision_sources.json.

Next experiment registered before training: replicate the original current-text narrow LSFB T5-base run with seeds 85 and 86, retaining all settings, data, development checkpoint selection and evaluation. Compare all three seeds with same-split retrieval, without selecting a winning seed for deployment. The 23-example development set is previously inspected and has only six multi-sign examples; this measures training stability on that set, not new independent generalization. No language mixing, newly inferred glosses or qualified approval. See reports/LSFB_SEED_REPLICATION_PLAN.md. Eight data-integrity, scoring and result-aggregation tests passed. Training results are recorded in the following section only once completed.

## Narrow LSFB seed replication completed

Completed 2026-11-17 on the local RTX 4060 Laptop GPU using existing local data and T5-base weights. Two new 25-epoch / 475-update LoRA runs replicate seed 84 with seeds 85 and 86. All run manifests match exactly except the seed. Selected adapters were reloaded before final decoding. No official test set was opened.

| Method | Exact / 23 | Operation WER | Multi-sign exact / 6 | Multi-sign WER | Selected epoch |
|---|---:|---:|---:|---:|---:|
| Current-text retrieval | 7 | 63.33% | 0 | 61.54% | — |
| T5-base seed 84 | 12 | 46.67% | 0 | 69.23% | 11 |
| T5-base seed 85 | 10 | 53.33% | 0 | 69.23% | 12 |
| T5-base seed 86 | 11 | 50.00% | 0 | 69.23% | 20 |

Across seeds: mean exact 47.83% (sample SD 4.35 percentage points), mean WER 50.00% (sample SD 3.33 points). These quantify optimization variation on the same examples, not uncertainty across independent populations. All three seeds have lower aggregate development WER than retrieval; none improves exact multi-sign performance, and all have worse multi-sign WER than retrieval. All final generations terminate without empty outputs or generation-cap hits.


All three seeds beat retrieval on aggregate inspected-development WER, but every exact success is single-sign. Multi-sign WER is 69.23% for every seed versus retrieval 61.54%; no sentence-composition improvement. Mean WER 50.00% ± 3.33 percentage points sample SD; these shared-example seed variations are not independent test evidence. Five of six multi-sign sequences are unseen in training despite every label having at least three training occurrences. No additional language mixing, official-test access or human approval. Eight targeted tests, compilation and whitespace checks passed. Both new adapters were freshly reloaded for evaluation and hash-verified backups retained. Full commands, outcomes and limitations: research/text-to-gloss/lsfb/LSFB_SEED_REPLICATION_RESULT.md/.json. Next: qualified alignment/multi-sign review and a frozen, adequately supported conversation holdout before expanding the model.

## Source alignment audit and video-backed multi-sign review

All 2,531 retained Belgian research/code/training/development rows match their original frozen candidate recording IDs, translation intervals and recomputed lexical projections; source candidate content hashes checked. No event below prescribed 80% overlap, excluded boundary event or annotation-frame deviation above 40 ms was found in these retained rows. This verifies preprocessing consistency, not independent French/sign semantic or video-clock alignment. Of these rows, 1,577 (62.31%) contain nonlexical events omitted by the prediction target; sidecar events remain preserved. Thus lexical accuracy alone cannot establish complete signing translation.

Created 55 private indexed-TAR video excerpts for all 49 narrow multi-sign training examples and six development examples, with synchronized clickable hand-tier annotations and segment highlighting. All clips generated successfully and passed ffprobe; actual Chrome playback/seeking/rejection gates passed. Fifteen source/preparation/release tests passed. Zero review decisions or approvals created, no data labels changed, no official test opened. Review pack: research/private/lsfb-alignment-audit/video-review/index.html. Full evidence and commands: research/text-to-gloss/lsfb/LSFB_SOURCE_ALIGNMENT_AUDIT.md/.json. Qualified LSFB review remains essential.

## emotion2vec checkpoint coverage and unused-decoder experiment

The earlier ten missing reconstruction-decoder tensors were investigated against actual local FunASR 1.4.26 and the existing seed checkpoint. All 185 inference tensors (93,178,133 values), including classification head, match the checkpoint exactly. On the eight-second French demo, the decoder had zero forward calls and NaN perturbation of its weights left all nine scores exactly unchanged. The library’s generic all-keys-matched message alone was insufficient; independent tensor equality supplies the evidence.

Production worker now checks inference weights, narrowly exempts the audited decoder entries and rejects any attempt to execute that untrained decoder. Score-label/distribution validation added. Five weight/score tests and 24 API/capture regression tests passed. Guarded real worker and actual service reproduced previous model scores; service returned 8,000 ms / 200 acoustic windows / nine categories. No new download, training or checkpoint modification. Emotional accuracy/calibration and French-domain validation remain unmeasured; model card license is other/model-license, with no independent commercial-use conclusion. Full evidence: research/audio/EMOTION2VEC_WEIGHT_AUDIT.md/.json.

## Audio-rhythm modulation wiring fix

The live avatar component received `energyFactor` from the audio timeline but did not pass it to the motion interpolator, so the existing bounded acoustic-energy amplitude adjustment was inactive in rendered playback. `AvatarCanvas` now forwards the current expressive toggle and energy value into interpolation. A direct runtime check confirmed the measured low/high energy settings produce wrist scaling factors 0.96/1.05, while finger coordinates remain unchanged. Existing frontend motion/avatar checks passed and the production TypeScript/Vite build succeeded. This is a visual rhythm cue, not sign-language prosody modeling or evidence of more natural signing.

## Local French ASR integration

Added local transcription to the audio analysis path using faster-whisper 1.2.1 and the MIT-licensed Systran/faster-whisper-small multilingual checkpoint, pinned at revision `536b0662742c02347bc0e980a01041f333bce120`. The 483,546,902-byte model was downloaded through a Colab CPU session, transferred to the PC in 16 MB pieces after the Google Drive authorization endpoint returned HTTP 400, and verified against the Hub model-file SHA-256. No user recordings or private datasets were uploaded. Local inference used the RTX 4060 with CTranslate2 FP16.

`POST /analyze-audio?include_transcription=true` now returns a French transcript draft and timestamped segments in addition to acoustic features and optional emotion2vec results. The studio lets the user insert the transcript into the editable French text area; it does not overwrite existing text automatically. ASR failure leaves acoustic analysis intact and returns a warning. A local request against the existing short `data/public_examples/bonjour.webm` sample returned HTTP 200, transcript “Bonjour.”, one segment at 0–2,000 ms, 0.53 seconds model decoding and 3.22 seconds total API time. This confirms execution and schema integration on one short sample only, not French ASR accuracy or translation quality. The Playwright smoke exercised upload → transcript → unchanged existing text → explicit draft insertion → French text-to-motion compose. Exact model hashes, setup, limits and repeat commands are in `research/audio/ASR_MODEL_SETUP.md` and `configs/asr_model.json`.

## Ingestion registration repair and source-language migration

Both Commons and Dicta-Sign ingestion scripts were repaired to supply the required explicit France-LSF identifier at registration, with source-language/eligibility checks before IO. Real cached captures passed both ingestion paths in a temporary offline library without modifying live assets.

All 1,446 legacy library entries were then identified against explicit France-LSF source inventories and recorded capture hashes (eight early Commons files matched cached original file SHA1s where sidecars were absent). Added source-language evidence only; every existing non-language field, review and rights status is exactly unchanged. Original library backed up. The registry now excludes missing language declarations instead of defaulting them to fsl. All 1,452 registered assets, including base assets, pass actual API asset-file validation; 41 regression tests passed. No new training, motion replacement, external download or human approval. This verifies source identity and integration, not linguistic correctness. Full report: reports/INGESTION_LANGUAGE_INTEGRATION.md.

## France-LSF supported-construction and target-coverage audit

A training-only retrieval baseline on the source-recording-held-out 39/28 France-LSF UD split scored 0/28 exact, 152.58% gloss-token WER. Of 194 validation gloss tokens, 28 are labels absent from train vocabulary; zero of 28 reference sequences have all labels supported, only one has all adjacent gold-label pairs observed in training, and no complete target sequence repeats. This makes another constrained-decoding T5 score on the same split a weak test of construction ability, so no new model was run.

The target projection also discards UD FEATS/MISC layers. Parsed original source annotations show 251 such feature occurrences in train/273 in development, including NMM 98/116, spatial/action features and 7 `Reported` values in development absent from train. This measures representation loss, not which features are obligatory, and French remains a free translation (`Parallel:no`). Official test content was not examined. See research/code/training/fsl_construction_audit.md and machine-readable/private evidence under research/private/fsl-construction-audit-v1. Next: source/review an adequately supported dataset with aligned FORM and selected manual features before another meaningful T5 comparison.

## Full current-browser integration rerun (2026-11-17)

Fresh production frontend build and motion/avatar checks passed; full Playwright browser flow against the current research UI/API passed composition, seeking, WebM download, WAV pitch/pause analysis and timing, explicit fsl fixture registration/provenance, persistent lookup, unknown words, mobile overflow and offline errors. The video was 340,012 bytes / 3.664058 sec by ffprobe. The temporary fixture row was cleaned; no TestPipeline entry remains. Private outputs: research/private/full-pipeline-smoke-20261117. Logs in reports/full_pipeline_*.log. This UI capture uses a synthetic procedural fixture; it is not evidence of new CV capture or natural sign production. No human linguistic review or intelligibility result.

## Belgian LSFB conversation-held-out multi-sign experiment

Context T5-base on 303 training and 83 development rows produced **0/83**
exact matches and **92.00% operation WER** after five local epochs.
Current-turn retrieval produced 2/83 and 105.33% WER. Repeating the most
frequent training sequence produced 2/83 and 93.33% WER. The lower edit rate
did not yield exact multi-sign predictions. Conversation IDs were disjoint,
13 signers were shared, and the lexical targets still require qualified review.
The selected checkpoint used this development set for selection.

[Run settings, split audit, controls, and limitations](text-to-gloss/lsfb/LSFB_FREQUENT_CONTEXT_RESULT.md).

## France-LSF SUD recording holdout baseline

Across 130 examples and six recording folds, French TF-IDF retrieval produced
0 exact matches and 134.75% FORM-token error. The most frequent sequence per
fold also produced 0 exact matches, with 128.19% error. Only 3 references had
complete training-fold FORM vocabulary coverage. These are lexical annotation
metrics on previously inspected source material, with local review pending.
[Method, limitations and reproduction](text-to-gloss/fsl/FSL_SUD_RECORDING_BASELINE.md).

## France-LSF 2D motion with visible endpoints

Selecting the nearest visible joint sample on each side reduced 2D mean
joint error from 0.19520 to 0.13782 on the 198 previously inspected test
windows. The earlier full-loss GAN scored 0.19428. Finger error improved
slightly and wrist error was unchanged. This is a masked 2D motion
completion result. [Metrics and reproduction](motion/FSL_VISIBLE_ENDPOINTS.md).

The subsequent France-LSF residual GAN run using visible endpoints reached
0.137803 mean 2D joint error versus 0.137817 for interpolation alone, with
worse boundary velocity error. It adds no practical gain on these reused
windows. [Measured results and reproduction](motion/FSL_VISIBLE_ENDPOINTS.md).

Wrist-relative finger interpolation lowered projected bone length error
from 0.05087 to 0.04296 on the reused France-LSF test windows, while
fingertip error rose from 0.33378 to 0.36366. The independent visible
endpoint baseline remains preferable for joint and velocity error.
[Results and reproduction](motion/FSL_WRIST_RELATIVE.md).

## France-LSF weak subtitle-to-pose feasibility

On the 19 test cues from two held-out recordings, French caption retrieval
had normalized 2D joint error 0.49007, versus 0.36477 for a training template
that ignores text. Hand error was also worse. Only 21 cues were available for
training and their subtitle alignment is unreviewed.
[Protocol and results](motion/FSL_CAPTION_POSE_BASELINE.md).
