# Held-out evaluation

Validation contained 47 cues from two source recordings and test contained 19 cues from two other recordings. The 66 held-out records, their intervals and their pose checksums matched across the initial and expanded experiments. The phase template averages valid training poses independently of caption text. The mismatched-caption control rotates the caption assignments within each evaluation split. Each run applies its training support mask consistently across its three methods. The template and support are recomputed for expanded training, so comparisons between training sets are descriptive.

| Training set and method | Validation joint error | Test joint error | Test DTW joint error | Test hand error | Test frame-count MAE at 24 fps |
| --- | ---: | ---: | ---: | ---: | ---: |
| 21 cues, learned encoder | 0.6378 | 0.5424 | 0.5425 | 0.8397 | 59.2 |
| 21 cues, frozen French encoder | 0.6033 | 0.4801 | 0.4796 | 0.8062 | 31.3 |
| 21 cues, phase template | 0.5045 | 0.3645 | 0.3603 | 0.7579 | 49.9 |
| 102 cues, frozen French encoder | 0.5450 | 0.3985 | 0.3981 | 0.7211 | 33.1 |
| 102 cues, mismatched captions | 0.5492 | 0.4035 | 0.4031 | 0.7418 | 51.8 |
| 102 cues, phase template | 0.4937 | 0.3259 | 0.3255 | 0.6942 | 52.1 |

The expanded model reduced test joint error by 17% relative to the smaller frozen-encoder pilot. Its phase template still had lower test joint, wrist and hand errors. The difference from mismatched captions was 0.0050 in joint error and 18.7 frames in duration MAE. Two held-out recordings and one caption rotation cannot establish robust motion conditioning. Expanded generated phase displacement averaged 0.0189, compared with 0.0404 for the phase template. Across-cue coordinate diversity declined from 0.1048 to 0.0794 for the frozen model, consistent with a tendency toward conservative motion.

Each evaluation generated a comparison of observed motion, the model prediction and the template. The comparison aligns each sequence to the observed duration. Standalone predictions use their own predicted duration. The museum demo is exploratory and linguistic correctness is unverified. Detailed per-split velocity, acceleration, wrist, invalid-coordinate and diversity measures are in the evaluation logs.

## Frozen-checkpoint follow-up on additional recordings

The existing 102-cue checkpoint was evaluated without retraining. Two newly staged official test recordings provided 26 cues. Together with the original test set, evaluation covers 45 cues from four source recordings. Each cue was compared with up to five distinct captions drawn deterministically from other recordings within its evaluation split. These controls supersede the earlier single-rotation comparison. Frozen-encoder embeddings were computed on a Colab T4 and transferred to the local evaluator after a full evaluation bundle upload timed out. Pose scoring and video rendering ran locally.

| Test recording | Cues | Matched joint error ↓ | Mismatched joint error ↓ | Phase template joint error ↓ | Matched / mismatched / template frame MAE ↓ |
| --- | ---: | ---: | ---: | ---: | --- |
| Original 01 | 7 | 0.4266 | 0.4475 | **0.3660** | 64.3 / 95.3 / 80.3 |
| Original 02 | 12 | 0.3821 | 0.3704 | **0.3025** | 14.8 / 49.1 / 35.6 |
| Additional 01 | 13 | 0.4155 | 0.4406 | **0.3802** | 37.8 / 75.0 / 50.0 |
| Additional 02 | 13 | 0.4302 | 0.4253 | **0.3752** | 33.6 / 65.9 / 48.4 |
| Additional set | 26 | 0.4228 | 0.4329 | **0.3777** | 35.7 / 70.4 / 49.2 |

The phase template had lower joint error on every test recording. Correct captions lowered joint error versus mismatched captions on two of the four recordings, and lowered duration error on all four. The additional-set hand error was 0.7759 with matched captions, 0.7962 with mismatched captions and 0.7812 with the template. The small hand advantage over the template came from one recording. Caption-conditioned duration is the clearest measured signal. The duration target reflects released subtitle spans and has not been checked against human-annotated signing boundaries.

Both comparison MP4s passed FFprobe. The four local unit tests passed. The full machine-readable per-recording report and videos live under private `outputs/holdout/`. The [review worksheet protocol](ALIGNMENT_REVIEW.md) covers the remaining fluent-review step.

## Retrieval and caption-length controls

The fixed checkpoint was re-evaluated locally. The retrieval index uses French word and within-word trigram TF–IDF fitted on the 102 training captions only. Top-1 returns the nearest training trajectory. The top-3 variant averages the three closest trajectories with valid landmarks and blends 50% of the phase template into the result. Missing retrieved landmarks fall back to the template. These settings were fixed before scoring.

| Method | Validation joint error ↓ | Original test joint error ↓ | Additional test joint error ↓ |
| --- | ---: | ---: | ---: |
| Frozen checkpoint | 0.5450 | 0.3985 | 0.4228 |
| Top-1 retrieval | 0.6001 | 0.4101 | 0.5076 |
| Top-3 retrieval blended with template | 0.5112 | 0.3364 | 0.3987 |
| Phase template | **0.4937** | **0.3259** | **0.3777** |

Blended retrieval reduced joint error relative to the checkpoint on all four test recordings. The phase template scored lower than blended retrieval on all four. On the additional test, the blended retrieval hand error was 0.8324, compared with 0.7759 for the checkpoint.

A separate duration control fit ordinary least squares on the training cues only, regressing log duration in seconds on an intercept and `log1p(len(caption.strip()))`. It exponentiates and clips predictions to 0.25–15 s and rounds to 24 FPS for frame-count MAE. The fitted coefficients were −1.690808 and 0.757617. Its MAE was 14.8 frames on validation, 30.5 on original test, and 34.2 on additional test. The checkpoint scored 19.3, 33.1 and 35.7, respectively. Caption length alone explains enough of the released subtitle duration for this simpler predictor to outperform the checkpoint on all three aggregate splits.

Re-run the [additional-recording evaluation command](../COLAB_SSH.md) with `--out outputs/retrieval`. The evaluator includes both retrieval controls. Five unit tests passed. The updated comparison MP4 passed FFprobe. The private review worksheet still had zero judgments across 34 clips at the time of evaluation.

## Local training-set expansion from 102 to 179 cues

The pinned 317,245,440-byte shard `Sign_DWPose_NPZ_000636.tar` indexed 32 eligible training recordings. The bounded staging pass found 31 French-caption candidates and staged eight. Of 83 candidate cues, 77 passed pose coverage. All eight recordings were disjoint from earlier train, validation and both test cohorts. The combined training set contains 179 cues across 19 recordings.

Both decoders were trained from scratch for 100 steps on the same RTX 4060 Laptop GPU, with the same configuration, seed, frozen encoder and checkpoint selection criterion. Encoder features were generated on Colab and cached for both local runs. Validation and test rows stayed fixed. The expanded training set had six additional supported phase–landmark cells out of 4,096. Each evaluator applied its own training support mask consistently across methods.

| Train cues | Validation joint error ↓ | Original test joint error ↓ | Additional test joint error ↓ |
| ---: | ---: | ---: | ---: |
| 102 | 0.5401 | 0.3922 | 0.4437 |
| 179 | 0.5470 | 0.4080 | 0.4473 |

The added data did not improve joint error in this single-seed, fixed-step experiment. The lower checkpoint selection loss for the 179-cue run did not translate into lower held-out joint error. Matched versus mismatched caption joint error differed by less than 0.002 on both test cohorts for the 179-cue run. The caption-independent template still scored lower. These results do not establish what would happen with more steps or improved alignment.

Training logs are `scale-local-102.log` and `scale-local-179.log`. The corresponding evaluation reports and private videos are under `outputs/local_102/` and `outputs/local_179/`.

## Longer local training budget

Both datasets were rerun from scratch for 500 updates with the previous RTX 4060 configuration and seed 17. Validation selected step 150 for the 102-cue run and step 40 for the 179-cue run. Both the 100- and 500-step runs were rescored using the same training-derived 4,033 phase–landmark cells and the same held-out cues. The 179-cue run selected the same step at both budgets and yielded identical scores.

| Train cues | Maximum updates | Selected step | Validation joint error ↓ | Original test ↓ | Additional test ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 102 | 100 | 70 | 0.5401 | 0.3922 | 0.4437 |
| 102 | 500 | 150 | 0.5237 | 0.3558 | 0.4146 |
| 179 | 100 | 40 | 0.5470 | 0.4080 | 0.4473 |
| 179 | 500 | 40 | 0.5470 | 0.4080 | 0.4473 |

The longer budget improved the 102-cue result by 0.0364 on the original test and 0.0291 on the additional test. Its caption-independent template still scored 0.3259 and 0.3777. The 179-cue run did not recover with extra steps. This remains a single-seed comparison with weak subtitle timing. All five unit tests passed.

Reproduce with the commands from the earlier local comparison, using `configs/frozen_500.yaml` for training and `--support-manifest .runtime/direct-fsl-expanded/cues.jsonl --metrics-only` for evaluation. Local checkpoints are under `checkpoints/local_102_500/` and `checkpoints/local_179_500/`.

## Three-seed local comparison

The 500-update experiment was repeated on the local RTX 4060 with seeds 29 and 43. All four new runs reached 500 updates. Each run selected its checkpoint using validation loss and was scored on the same held-out cues with the same landmark mask.

| Seed | Training cues | Selected step | Validation joint error ↓ | Original test ↓ | Additional test ↓ |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 17 | 102 | 150 | 0.5237 | 0.3558 | 0.4146 |
| 17 | 179 | 40 | 0.5470 | 0.4080 | 0.4473 |
| 29 | 102 | 80 | 0.5263 | 0.3773 | 0.4271 |
| 29 | 179 | 400 | 0.5094 | 0.3622 | 0.4328 |
| 43 | 102 | 80 | 0.5397 | 0.3773 | 0.4501 |
| 43 | 179 | 120 | 0.5312 | 0.3702 | 0.4249 |
| Mean | 102 | — | 0.5299 | 0.3701 | 0.4306 |
| Mean | 179 | — | 0.5292 | 0.3801 | 0.4350 |

With 179 cues, seed 29 improved on the original test and seed 43 improved on both tests. The 179-cue mean was higher on both tests. The caption-independent template had lower joint error on both test cohorts in every run. Three seeds help show variability under this training setup. They do not increase the number of independent test recordings. All five unit tests passed. [Reproduction commands](../COLAB_SSH.md#reproduce-the-local-data-scale-study)

## Template-residual motion decoder

The decoder starts from the masked average training pose at each phase. A zero-initialized head learns a caption-conditioned correction bounded to ±0.15 normalized units per coordinate. The duration head is unchanged. Three seeds ran for 500 local RTX 4060 updates on 102 training cues. The same held-out recordings and scoring mask were used. Validation loss selected checkpoints at steps 50, 90, and 10.

| Method | Original test mean joint error ↓ | Additional test mean joint error ↓ |
| --- | ---: | ---: |
| Direct decoder, three seeds | 0.3701 | 0.4306 |
| Residual decoder, correct captions | 0.3352 | 0.3901 |
| Residual decoder, mismatched captions | 0.3365 | 0.3948 |
| Caption-independent phase template | **0.3259** | **0.3777** |

Anchoring to the template improved the neural model’s joint error. The template alone still scored lower on both test sets in all three runs. The matched-caption advantage averaged 0.0013 and 0.0047 and reversed on both tests for seed 17. The results support template anchoring as a pose error reduction, with limited evidence for learning the movements implied by the caption. Six unit tests passed, including zero-deviation initialization and the correction bound. [Reproduction commands](../COLAB_SSH.md#reproduce-the-local-data-scale-study)
