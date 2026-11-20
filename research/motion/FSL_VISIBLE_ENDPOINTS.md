# France-LSF 2D motion baseline with visible endpoints

The previous linear and GAN baselines used the samples immediately before
and after the gap even when the corresponding joint samples were masked.
About 26% of joint-window endpoint pairs in the existing test set lack
at least one nominal endpoint. The new baseline searches the observed
context for the closest valid sample on each side for each joint. It
interpolates over the actual time interval, holds the position if only one
side is valid, and returns zero if both sides are absent.

| Metric | Fixed linear | Visible endpoints |
|---|---:|---:|
| Validation 2D mean joint error, 224 windows | 0.24793 | 0.17695 |
| Test 2D mean joint error, 198 windows | 0.19520 | 0.13782 |
| Test fingertip error | 0.34299 | 0.33378 |
| Test wrist error | 0.18328 | 0.18328 |
| Test boundary velocity error | 1.97369 | 1.77983 |

On the same previously inspected test windows, the earlier full-loss GAN
reported 0.19428 mean joint error. Visibility-aware interpolation lowers
the fixed linear baseline's error by 29.4%. The fingertip gain is smaller
and wrist error is unchanged. The windows reuse a small set of continuous
signing videos, and the results measure masked 2D reconstruction only.
The baseline ran without model fitting and has been added to the comparison
run so future GANs are evaluated against it. The avatar's playback remains
independent of this 2D representation.

The command checked the window file hashes, verified that replacing the
hidden frames leaves predictions identical, and confirmed agreement with
fixed linear interpolation where both original endpoints are visible.

```bash
.runtime/t5-local-venv/bin/python -m src.motion.evaluate_visible_endpoints \
  --data data/poses/private/windows-v1 --output /tmp/fsl-visible-endpoints.json
```

[Full metrics](FSL_VISIBLE_ENDPOINTS.json)

## Residual models trained on visible endpoints

The local RTX 4060 ran reconstruction-only, adversarial, and full-loss
variants on 125 training windows. Both networks used visible endpoints.
The critic compared real and generated gap frames under the same validity
mask. Validation used 224 windows and selected epoch one for all three
variants, followed by early stopping at epoch nine. Test scores reuse
198 previously inspected windows.

| Method | Test 2D joint error | Test boundary velocity error |
|---|---:|---:|
| Visible endpoint interpolation | 0.137817 | 1.779834 |
| Reconstruction-only residual | 0.137807 | 1.780869 |
| Adversarial residual | 0.137803 | 1.780859 |
| Full-loss residual | 0.137811 | 1.780837 |

The best residual reduces joint error by about 0.01% and increases
boundary velocity error. The visible endpoint baseline remains the practical
choice for these windows. These results use one seed and a small number of
source videos. The new training option leaves earlier checkpoints compatible
with the original conditioning, and resume rejects a conditioning mismatch.

```bash
.runtime/t5-local-venv/bin/python -m src.motion.train_inbetween --visible-endpoints \
  --data data/poses/private/windows-v1 --out .runtime/fsl-motion-visible-reproduction
```

[Residual experiment metrics](FSL_VISIBLE_ENDPOINTS_LEARNED.json)
