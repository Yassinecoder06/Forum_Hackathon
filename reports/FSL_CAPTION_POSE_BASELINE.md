# France-LSF subtitle-to-pose feasibility comparison

The 21 training cues supply two simple baselines. The first averages the
available training coordinates at each of 32 relative positions in a cue.
The second retrieves the motion of the nearest French training subtitle
using word TF-IDF cosine similarity. It falls back to the template for
missing landmarks or queries with zero overlap. Scoring uses identical
reference landmarks with training support for both methods.

| Method | Validation joint error | Test joint error | Test hand error |
|---|---:|---:|---:|
| Caption-independent template | 0.50669 | 0.36477 | 0.76169 |
| French caption retrieval | 0.59188 | 0.49007 | 1.04965 |

The caption retrieval baseline lost on both splits. Nine of 47 validation
cues and two of 19 test cues had no training word overlap. Evaluation
preserves recording disjointness, with two source recordings in each held
out split and three in training. Unknown signer IDs prevent signer
disjointness claims.

The units are shoulder-normalized 2D distances. Each cue is aligned by
relative phase, so this comparison measures shape and position at supplied
relative times. It does not predict absolute duration, identify glosses,
validate subtitle alignment, or establish natural signing. The template
can average away actual movement. Its lower error still establishes a
useful minimum baseline for this particular measurement.

```bash
.runtime/t5-local-venv/bin/python -m src.motion.evaluate_caption_retrieval \
  --manifest data/manifests/signverse_fsl_caption_pose.jsonl \
  --output /tmp/fsl-caption-pose-baseline.json
```

[Full metrics and evaluation manifest hash](FSL_CAPTION_POSE_BASELINE.json)
