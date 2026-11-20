# SignaCulture direct France-LSF motion progress

## What we built

The prototype takes French text and predicts continuous 2D motion for 128 body, hand, and face landmarks. It operates alongside SignaCulture’s existing gloss-to-avatar pipeline. The retained model uses a frozen DistilCamemBERT encoder and a trained cross-attention motion decoder. The decoder predicts 32 normalized motion phases and estimates duration separately. The experiment uses no gloss targets, T5 fine-tuning, or GAN. [Architecture and commands](../README.md)

## Data and processing

- The pinned SignVerse-2M metadata identified 900 France-LSF recordings, with 810 assigned to train, 45 to validation, and 45 to test. The archive inventory mapped 860 of the 900 recordings.
- We used the exact release language code `fsl` and French VTT tracks. Transcription provenance and signing alignment are unknown. The original source-recording splits were preserved.
- We checked cached pose archives, tracked the signer, normalized 2D positions around the shoulders, kept landmark validity masks, and filtered caption intervals for continuous frame coverage and sufficient hand and body visibility.
- The initial pilot used 21 training cues from three recordings. Two bounded training-shard downloads increased usable training data first to 102 cues from 11 recordings, then to 179 from 19. Validation stayed at 47 cues from two recordings. Two further bounded downloads added 26 usable cues to the original 19 test cues, for 45 test cues across four recordings.
- The release is CC BY-NC 4.0. Derived media and source identifiers remain private. [Dataset audit](FSL_DATASET_AUDIT.md)

## Experiments we ran

| Experiment | Compute | Training cues | Key result |
| --- | --- | ---: | --- |
| Learned word encoder and motion decoder | Colab Tesla T4 | 21 | Original test joint error 0.5424 |
| Frozen DistilCamemBERT and motion decoder | Colab Tesla T4 | 21 | Original test joint error 0.4801 |
| Frozen encoder with expanded training data | Colab Tesla T4 | 102 | Original test joint error 0.3985 |
| Same 102-cue configuration rerun | Local RTX 4060 Laptop GPU | 102 | Original test 0.3922, additional test 0.4437 |
| Same configuration with more data | Local RTX 4060 Laptop GPU | 179 | Original test 0.4080, additional test 0.4473 |

Each production run used 100 optimization steps, with checkpoints selected on validation. The matched local comparison used the same configuration and seed. Increasing training from 102 to 179 cues did not improve held-out joint error in that comparison. This was one seed and one fixed training budget. [Training details](TRAINING_RESULTS.md) · [Full evaluation](EVALUATION.md)

## What the results tell us

Lower joint error means closer agreement with the recorded 2D landmarks. The average training motion beat the original 102-cue checkpoint on joint error across all four test recordings. Blending the three closest retrieved motions with the training average also beat the checkpoint on joint error across all four. Correct captions improved duration estimates compared with mismatched captions, but a regression using caption length alone predicted the released subtitle spans more accurately than the checkpoint. The model generates motion. The experiments have yet to establish that its movements convey the meaning of novel French sentences. [Metrics and controls](EVALUATION.md) · [Limitations](LIMITATIONS.md)

## Demo and review workflow

- The local demo plays observed skeletons, predictions, and baselines side by side. It also includes the saved prediction for « Bonjour, bienvenue au musée. » and two exact training captions, « Je serai présent pour vous guider en LSF ! » and « Nous allons préparer le jardin pour l'été. » The training views show the observed skeleton alongside the original 102-cue checkpoint’s prediction. Playback and example switching were verified in Chromium. [Open the local demo](http://127.0.0.1:8104/) · [Start the demo](../../../../docs/test-sites/direct-fsl-demo.md)
- The private review worksheet contains 34 observed clips from 17 recordings. It supports local browser ratings and CSV export. The intake validates reviewer responses, re-slices source-backed timing corrections, and can score original and corrected intervals with the same checkpoint. We tested the round trip with a simulated correction. The last check found zero real reviewer judgments. [Review instructions](ALIGNMENT_REVIEW.md)

## Next step

Have a fluent France-LSF reviewer confirm meaning alignment and, where authorized source footage is available, verify signing boundaries. Run the paired evaluation on those reviewed intervals before deciding whether to prioritize data alignment, training length, or a different model. Linguistic intelligibility and browser-based 3D avatar playback are still unverified. [Colab and reproduction commands](../COLAB_SSH.md)

## Follow-up after the 179-cue comparison

We reran both local models with a 500-step budget and rescored all four local runs with a shared landmark mask. The smaller dataset improved to 0.3558 and 0.4146 joint error on the two test cohorts. The larger run selected the same early checkpoint and stayed at 0.4080 and 0.4473. The caption-independent template still led. [Longer-budget results](EVALUATION.md#longer-local-training-budget)

The 500-update comparison now covers three seeds. The larger set helped in some runs, while its mean held-out joint error was higher on both test cohorts. The caption-independent template led in all six runs. [Three-seed results](EVALUATION.md#three-seed-local-comparison)

A template-residual decoder improved mean joint error over direct prediction across three seeds. The caption-independent template still scored best, and the correct-caption advantage over mismatched captions remained small. [Residual decoder results](EVALUATION.md#template-residual-motion-decoder)
