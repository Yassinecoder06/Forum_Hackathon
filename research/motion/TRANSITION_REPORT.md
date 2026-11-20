# Five-point transition component: measured within-sign reconstruction

A functioning endpoint-to-10-frame component and a trained conditional GAN now exist. **The measured task is intra-sign masked-window reconstruction, not real inter-sign coarticulation.** For the lead's actual inter-sign renderer, retain deterministic interpolation until genuine paired transition supervision and motion review exist.

## Paper gap addressed

Inspected the actual supplied `research_paper.pdf`, PDF page 4 §3.5, and the lead's seven-page verified review. The paper specifies five selected points, ten x/y features, ten output frames, three generator hidden layers with batch normalization, dropout .4 in Eqs1–2, discriminator LeakyReLU .2/dropout .3, and adversarial + L1 loss with lambda1000. Hidden widths, conditioning details and complete preprocessing are not specified.

Our choices: generator widths256/256/128, BN/ReLU, dropout.4 after the first two hidden layers, linear100 output; discriminator widths256/128/64, LeakyReLU.2/dropout.3, scalar logit. Generator input is two10-feature endpoints plus16Gaussian-noise values. Discriminator sees endpoint20 plus interior100. BCEWithLogits implements adversarial loss; G loss is adversarial+1000L1. Adam lr.0002/betas(.5,.999), batch64, seed42, max50epochs/patience8, minimum validation normalizedRMSE. CPU with4threads.

**Engineering addition:** G predicts a residual over linear interpolation, with zero-initialized final weights. This provides a stable initial linear bridge. It is not the paper's exact Eq4 architecture. There is no L1-only ablation, so a benefit specifically attributable to adversarial training has not been established. All choices and parameter counts are recorded before training in TRANSITION_TRAINING_PLAN.json: G122340parameters, D72193parameters.

## Real source data and split

Read only the lead's439runtime motions plus the curated real Bonjour capture. Canonical isolated-clip source URLs and identical reduced geometry/timestamps are deduplicated before splitting; no duplicates were found among440clips. Seeded source-hash order assigns308train/66validation/66test clips **before** windows are created. No source clip or its mirror crosses splits. Shared original YouTube compilations and signer identities are not disjoint or independently annotated; do not claim signer-level generalization.

Five captured points: `head` (nose/head proxy), left/right wrists and `left_middle_4`/`right_middle_4`. Keep x/y only in the existing signaculture-v1 rig coordinates. Twelve consecutive captured frames supply the two outer anchors and the known middle ten targets, stride3. Reject missing points/nonfinite data; no imputation or30fps resampling.3154candidate windows are rejected for missing points. Targets remain captured/retargeted estimates pending motion and linguistic review, not expert-validated ground-truth signing.

Usable windows: **646train/152validation/93test**, from **178/40/32eligible source clips**. Native clip rates are12.5or14fps; outer-anchor durations785–880ms, unlike the paper's approximately.33s. Rounded native timestamps are uniform to1.01ms; interpolation uses uniform frame-index fractions. Overlapping windows stay within one split. Frozen source/checksum/window provenance is in TRANSITION_DATA_MANIFEST.json. Normalization is a per-feature z-score fit on training windows only and shared by every method.

## Actual run

Fifty epochs completed; validation selected epoch48. CPU training3.065s, total research/code/training/evaluation3.094s, no CUDA tensor allocation. Existing Qwen and all other processes were untouched. The test file was opened only after selection, and all three methods were evaluated in one held-out pass. All93test windows retain targets, three predictions, source/license/credit and per-window errors in runs/transition-intrasign-v1/test_predictions.jsonl.

| Method | Normalized RMSE | Normalized MAE | First/last interior RMSE | Native rig-XY RMSE | Mean source-clip RMSE |
|---|---|---|---|---|---|
| Linear |0.320293|0.165943|0.166854|0.053032|0.273291|
| Quintic easing |0.329619|0.163539|0.172697|0.054806|0.285029|
| Conditional GAN |0.314898|0.163036|0.166884|0.051878|0.251897|

GAN aggregate reconstructionRMSE is1.68% lower than linear; this small result has no statistical-significance or inter-sign-quality claim. Its first/last interiorRMSE is slightly worse than linear. Native boundaryRMSE is0.027977GAN versus0.027761linear. Supplied outer-anchor error is zero for all methods because anchors are copied, not predicted: this is a structural guarantee, not learned continuity evidence. No comparison to the paper's scores is valid because data, timing, coordinates, normalization and split differ.

The run's `recommendation=gan` refers only to this reconstruction metric. **Integration recommendation for actual inter-sign joins: deterministic linear/quintic interpolation remains the supported component; the GAN is research-only.** Neither method synthesizes the rest of a full3Dhand/face rig or validates linguistic coarticulation. Missing paper components still include the authors'379paired sentence transitions, exact training artifacts and professional perceptual evaluation.

## Runnable handoff

CLI accepts JSON `{start:[10numbers],end:[10numbers]}` with optional exact feature_order. It rejects wrong shapes, booleans, NaN/Infinity and invalid checkpoints. Output contains exactly ten interior frames, plus a separate12-frame array with original anchors copied exactly; timing fractions are explicit. No30fps guarantee is inferred. GAN inference uses zero noise, CPU and the validation-selected checkpoint. Linear/quintic need no checkpoint. All outputs are candidate/pending review, never validated translation.

```bash
research/code/training/.venv-train/bin/python research/code/training/transitions/infer.py \
  --endpoints research/runs/transition-intrasign-v1/example_endpoints.json --method linear

research/code/training/.venv-train/bin/python research/code/training/transitions/infer.py \
  --endpoints research/runs/transition-intrasign-v1/example_endpoints.json --method gan \
  --checkpoint /home/jmal/Projects/Active/Forum_Hackathon/.worktrees/wt-codex2/training/transitions/artifacts/transition-intrasign-v1.pt
```

All three CLI modes were actually run on a validation endpoint example. Six tests pass: endpoint schema/rejection, deterministic formulas/anchor preservation, source/mirror leakage, train-only normalization and actual source windows, selected-checkpoint/heldout metrics, and the trained bridge. Code, source/license metadata, full losses and predictions are committed; weights and NPZ caches are ignored. No existing catalog, text model, application, renderer or other panes were modified.
