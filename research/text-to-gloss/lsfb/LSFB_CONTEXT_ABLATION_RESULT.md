# Narrow Belgian LSFB context ablation

25epochs475steps per arm,same145train/23development examples and frozen35labels. Selected checkpoints freshly reloaded.

|Method|Exact|WER|Multi-sign exact|Multi-signWER|
|---|---:|---:|---:|---:|
|Current-French retrieval|7/23|63.33%|0/6|61.54%|
|Context T5|10/23|56.67%|0/6|76.92%|
|Current-only T5|12/23|46.67%|0/6|69.23%|

Current-only is the better aggregate developmental model in this experiment. Retrieval remains better on multi-sign token error. All neural exact matches are single-sign. One seed,23repeatedly inspected development examples and6multi-sign cases do not establish robust generalization. No new human approval or official-test evaluation. Five targeted checks passed.

Paired exact outcomes: {"both_correct": 10, "context_only_correct": 0, "current_only_correct": 2, "both_wrong": 11}

Evidence:research/text-to-gloss/lsfb/LSFB_CONTEXT_ABLATION_RESULT.json and private paired predictions. Current-only adapter copied and hashed under /home/jmal/Dataset/lsfb-narrow-context-v1/current-only-run/. Run command and limitations:LSFB_CONTEXT_ABLATION_PLAN.md. Next: reviewed multi-sign data and train-only multi-sign learnability checks before broader translation claims.
