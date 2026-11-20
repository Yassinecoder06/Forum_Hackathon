# Executed results

France-LSF motion feasibility:125train/224val/198test windows from3/2/2source videos, official video assignments preserved.16frames per window:4left context,8masked frames,4right context at24fps. Known signer identities unavailable. Overlapping windows remain within source splits. Selection uses validation2DMPJPE; test opened after selecting all three variants. One seed71, no statistical superiority claim. Raw normalized trajectories are reconstruction targets; smoothing is for previews. Availability substitutes explicitly for missing body confidence. Test missing/invalid landmark fraction16.88%.

| Method | normalized2DMPJPE | fingertip error | boundary velocity error |
|---|---:|---:|---:|
| Linear |0.195204|0.342989|1.973686|
| Hermite |0.267180|0.372063|2.362187|
| Reconstruction neural |0.194308|0.342837|1.995438|
| Reconstruction+adversarial |0.194355|0.342902|1.995126|
| Full losses GAN |0.194276|0.342715|1.992309|

FullGAN improves aggregate position error about0.48% versus linear but worsens boundary velocity. Adversarial loss alone does not beat the reconstruction neural baseline. No natural signing/coarticulation/intelligibility claim. Retain linear for integration. Projected bone metrics measure2Dlength consistency, not anatomy. Jerk/velocity are errors versus captured trajectories, not perceptual ratings.

Local RTX4060 execution took3.97seconds and62.12MiB peak allocated Torch memory (runtime overhead separate). Checkpoints, resumable optimizer states, history and per-window predictions are privately saved in `research/private/fsl-motion-v1/`.

Seven new unit tests pass: language rejection, confidence handling, restricted NPZ unpickling, archive traversal rejection, multi-person selection, hidden-target independence and masked-loss gradients. Compilation passed. Legacy tests:5passed/1failed because current source JSON hash differs from frozen provenance. Cached legacy experiment reproduced RMSE linear0.320293, quintic0.329619, GAN0.314898 (1.68%lower than linear). That is cached reconstruction only; source drift prevents fresh provenance confirmation. No30%improvement reproduced.

Text: the subsequent UD-only feasibility comparison actually completed. mT5-small: 0/28 development exact, WER 594.85%, BLEU 0, chrF 4.907, 28 cap hits. T5-base with train-derived tokenizer characters: 0/28 exact, WER 235.05%, BLEU 0.248, chrF 14.442, no cap hits. These are 39-training-example feasibility runs with zero local linguistic approvals, not matched model-size ablations. mT5-base never trained; its partial local file is unusable and Colab currently reports no active sessions. See FSL_MODEL_SIZE_COMPARISON.md and MT5_BASE_TRANSFER_STATUS.json.

Belgian LSFB remains separate: three current-text seeds scored 10–12/23 development exact and mean WER 50.00%, versus retrieval 63.33%; every seed scored 0/6 multi-sign exact. The development set is repeatedly inspected, not an independent test. GeoQuery graph generation also failed to beat retrieval. Full evidence: LSFB_SEED_REPLICATION_RESULT.md and GEOQUERY_FULL_GRAPH_RESULT.md.

Avatar/software: existing real capture/import, composition, timing, browser playback and WebM export were executed; preview is 341,196 bytes / 3.654063 seconds. emotion2vec checkpoint coverage and actual inference were verified independently. All 1,446 legacy captures now have source-linked explicit fsl metadata; the registry excludes missing declarations, and all 1,452 registered assets pass API file validation. Those are software checks, not linguistic approval or calibrated 3D lifting. See PIPELINE_AND_MULTISIGN_RECHECK.md, EMOTION2VEC_WEIGHT_AUDIT.md and INGESTION_LANGUAGE_INTEGRATION.md.

Current full requirements audit: ../IMPLEMENTATION_STATUS.md. No reliable general translator, GAN superiority or natural/intelligible signing claim is established.

France-LSF support audit: training-only retrieval on the 39/28 split scored 0/28 exact and 152.58% WER; 28/194 development target tokens are OOV, zero full targets have all labels supported, and only one target has all adjacent pairs seen. Original UD feature fields omitted from FORM-only targets occur 251 times in train and 273 in development, with 7 held-out `Reported` features unseen in train. This shows that a constrained T5 run on the existing split would conflate OOV/annotation coverage with generalization, so it was not run. See research/code/training/fsl_construction_audit.md and private auditable predictions/feature examples.

Current browser integration rerun on Nov 17 passed build, motion/avatar tests and the full UI flow; fresh WebM is 340,012 bytes / 3.664058 seconds, independently checked with ffprobe. Generated WAV audio timing, explicit fsl test capture registration/lookup, cleanup, mobile/offline states also passed. The capture UI step uses a synthetic procedural fixture; real video capture evidence and partial-hand limitation are separate. No linguistic quality conclusion follows.
