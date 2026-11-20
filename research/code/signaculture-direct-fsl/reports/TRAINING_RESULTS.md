# Training results

Three bounded pilots ran over Colab SSH on a Tesla T4. Checkpoints were selected by validation objective with seed 17, batch size 4 and 32 motion phases. Each pilot completed a 10-step synthetic diagnostic, an independent 35-step two-cue overfit diagnostic and 100 real optimization steps.

| Model | Train cues / recordings | Trainable parameters | Two-cue loss, initial → final | Best step | Validation objective | PyTorch peak allocated MiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Learned word encoder | 21 / 3 | 714,881 | 3.592 → 0.500 | 90 | 1.0053 | 52.3 |
| Frozen DistilCamemBERT | 21 / 3 | 549,857 | 3.168 → 0.489 | 20 | 0.9409 | 298.0 |
| Frozen DistilCamemBERT, expanded | 102 / 11 | 549,857 | 3.463 → 0.470 | 60 | 0.8930 | 319.5 |

The frozen encoder is `cmarkea/distilcamembert-base` pinned to `82fb7671777bb680416031dcced25c7bfa6e859b`. The loader reported an unused language-model head and newly initialized pooler. The decoder consumes hidden states and does not use either component. Memory figures measure the PyTorch allocator, excluding other GPU overhead.

The loss combines confidence-masked joint positions, velocity and log duration. Two-cue loss verifies the model can fit limited data. Held-out scores and the mismatched-caption control are reported separately in [Evaluation](EVALUATION.md). See the [Colab SSH runbook](../COLAB_SSH.md) for the training and expansion commands.
