# Five-point bridge experiment

This component generates10interior x/y frames for head, two wrists and two middle fingertips. It is tested on masked windows INSIDE real isolated captured signs, not true inter-sign coarticulation. It is not a full-rig motion generator. See `research/TRANSITION_REPORT.md` for exact scope and measured results.

Use the existing own Python3.11environment (torch2.7.1+cu128, numpy2.4.6, pytest9.2.0). Experiment and inference run onCPU; no Qwen/server/process control is performed. Source paths in common.py refer to the lead's local read-only library. Trained weights and NPZ caches are under artifacts/ and ignored.

Recreate deterministic windows from the same source files (a committed manifest is verified, never overwritten):

```bash
PYTHONDONTWRITEBYTECODE=1 training/.venv-train/bin/python training/transitions/prepare.py
PYTHONDONTWRITEBYTECODE=1 training/.venv-train/bin/python training/transitions/experiment.py --run transition-reproduction
```

A fresh run name prevents overwriting measured results. Source or preprocessing mismatch with the frozen manifest fails. Training fits per-feature mean/std only on train; validation selects a checkpoint before test is opened. Up to50epochs, patience8; test is decoded/evaluated once. Do not tune on held-out results.

Input shape:

```json
{"start":[0,1.6,0.2,1.1,-0.2,1.1,0.1,1,-0.1,1],"end":[0,1.6,0.3,1.3,-0.3,1.3,0.3,1.2,-0.3,1.2]}
```

Numbers above are schema-only fixtures, not LSF training data. Feature order is head_x/head_y, left_wrist_x/y, right_wrist_x/y, left_middle_4_x/y, right_middle_4_x/y. JSON stdout includes frames10x10, start/end, bridge_with_anchors12x10, time_fractions, method, candidate and pending-review flags. Supplied anchors are preserved exactly. Native timing is supplied by the caller, not inferred.

```bash
PYTHONDONTWRITEBYTECODE=1 training/.venv-train/bin/python training/transitions/infer.py --endpoints research/runs/transition-intrasign-v1/example_endpoints.json --method linear
PYTHONDONTWRITEBYTECODE=1 training/.venv-train/bin/python training/transitions/infer.py --endpoints research/runs/transition-intrasign-v1/example_endpoints.json --method quintic
PYTHONDONTWRITEBYTECODE=1 training/.venv-train/bin/python training/transitions/infer.py --endpoints research/runs/transition-intrasign-v1/example_endpoints.json --method gan --checkpoint training/transitions/artifacts/transition-intrasign-v1.pt
PYTHONDONTWRITEBYTECODE=1 training/.venv-train/bin/python -m pytest training/transitions/test_transitions.py -q -o cache_dir=training/transitions/.pytest_cache
```

The local v1checkpoint exists in the codex2worktree; a fresh reproduction writes its own named checkpoint. Weights are deliberately not inGit. For actual cross-sign application joins, use deterministic interpolation pending real transition supervision; the GAN's small within-sign reconstruction improvement does not validate that use. Paper widths/input choices are unspecified, so the residual conditioning, chosen widths and optimizer are documented engineering choices rather than an exact reproduction.
