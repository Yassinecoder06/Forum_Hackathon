# Reproduction

Run from repository root:

```bash
.runtime/t5-local-venv/bin/python -m src.data.prepare_parallel
.runtime/t5-local-venv/bin/python -m pytest research/code/tests/test_fsl_research.py -q
.runtime/t5-local-venv/bin/python -m src.motion.train_inbetween --data data/poses/private/windows-v1 --out research/private/fsl-motion-reproduction --epochs 40
.runtime/t5-local-venv/bin/python -m src.motion.train_inbetween --data data/poses/private/windows-v1 --out research/private/fsl-motion-reproduction --epochs 40 --resume
.runtime/t5-local-venv/bin/python -m src.data.visualize data/poses/private/windows-v1/5E7GtcKwDQM.npz research/motion/fsl_pose_preview.html
.runtime/t5-local-venv/bin/python research/code/training/transitions/experiment.py --run transition-fsl-another-reproduction
```

Preparation requires the privately downloaded sources. Sample preparation refuses existing output: `python -m src.data.prepare_signverse_samples`. Download scripts in research/code/src/data use `/content/fsl-audit` and run in Colab; audit,index,selection,range inspection and indexed fetch are sequential. Range fetch budgets are bounded; no whole archive extraction. `fetch_signverse_samples.py` is the initial schema-discovery scan; `fsl_fetch_indexed.py` in runtime contains the subsequent observed-path fetch, not yet packaged as a stable general downloader.

Generic HF mT5 entrypoint (not executed; verify dependencies and explicit feasibility acceptance before use):

```bash
.runtime/t5-local-venv/bin/python -m src.text_to_gloss.train_mt5 --config configs/mt5_small.yaml --output research/private/mt5-fsl-feasibility --feasibility
```

Trainer uses Transformers/Datasets/PEFT, deterministic seed, BF16, gradient checkpointing/accumulation, early stopping and resumable HF checkpoints. It is compile-checked only. Run a train-only fit gate before a full trial; obtain qualified review before describing translations as gold. Do not invoke the training example as evidence it already ran.


## Actually executed France-LSF comparison

The separate minimal runner completed these two feasibility runs on the verified UD-only files. Use new output names when reproducing:

```bash
.runtime/t5-local-venv/bin/python -m training.fsl_size_comparison --model .worktrees/wt-codex2/training/weights/mt5-small --name fsl-mt5-small-new-reproduction
.runtime/t5-local-venv/bin/python -m training.fsl_size_comparison --model /home/jmal/.cache/huggingface/hub/models--google-t5--t5-base/snapshots/a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1 --name fsl-t5-base-new-reproduction
python3 scripts/audit_research_status.py
```

These runner commands are reproduced for provenance, not a recommendation to repeat failed 39-example runs unchanged. The runner saves selected adapters and optimizer diagnostics but cannot automatically resume mid-run. The generic HF trainer's resume support is a separate unexecuted path. mT5-base transfer stopped incomplete; no training process is live. Detailed measured comparison: FSL_MODEL_SIZE_COMPARISON.md. Qualified alignment review is the next substantive requirement.

The mT5 entry point can validate the local France-LSF splits before loading
model weights or using the GPU.

```bash
.runtime/t5-local-venv/bin/python -m src.text_to_gloss.train_mt5 \
  --config configs/mt5_small.yaml --feasibility --preflight-only
```

The current source-annotation dataset passes as 39 training and 28
development pairs in feasibility mode. Strict mode checks the reviewed
alignment and approval fields on every row in both splits. An aggregate
review count does not establish eligibility. This was verified by editing
only the count in a temporary copy and observing rejection. The checks
validate dataset fields and split integrity. They do not authenticate a
reviewer's qualifications or create reviewed examples.
