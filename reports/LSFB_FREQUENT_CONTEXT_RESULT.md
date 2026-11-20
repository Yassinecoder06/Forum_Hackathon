# Belgian LSFB short lexical sequences with dialogue context

This experiment uses local, technically screened LSFB-CONT rows from
`.runtime/lsfb-cont-task-v3/candidates.jsonl`. Each input contains the current
French turn and completed preceding dialogue. Targets retain two to eight
annotated lexical signs. Qualified LSFB review is pending, and the task measures
agreement with a lexical projection of the corpus annotations.

The deterministic split reserved 18 of 89 source training conversations before
label selection. Following filtering, 303 examples from 56 conversations went
to training and 83 examples from 13 conversations went to development. The two
sets share 13 signers, zero conversations and zero normalized full inputs.
Official validation and test rows were excluded from fitting and scoring.
Development examples come from the existing corpus and are used for checkpoint
selection. Previous runs used the broader corpus, so these conversations
cannot be considered an independent benchmark.

The frequency rule required at least 20 occurrences and three conversations
in the training pool before short sequence filtering. After filtering, 14 of
300 development reference operations carry labels absent from the retained
training rows, and only four development target sequences occur verbatim in
training. The frequency rule therefore does not guarantee support in the final
training set.

The initial attempt stopped during retrieval on an unseen French feature.
After correcting the TF-IDF vectorizer to ignore features outside its train-only
vocabulary, a second attempt stopped with incomplete result and adapter files.
The restart uses the same data and pinned pretrained T5-base weights, with five
epochs, seed 111, rank 16 LoRA, BF16 and an effective batch size of eight on the
local RTX 4060. The adapter is chosen by minimum development operation WER.

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 .runtime/t5-local-venv/bin/python -m training.lsfb_cont.frequent_context_experiment --epochs 5 --output .runtime/lsfb-frequent-context-reproduction
```

Both retrieval baselines use the same retained training and development
examples. The current-turn variant uses `french_text` in the same retrieval
function. Retrieval with context achieved 0/83 exact matches and 118.00% WER.
Current-turn retrieval achieved 2/83 and 105.33% WER. Operation WER is edit
count divided by reference operation count and can exceed 100%.

A sensitivity check applied the repetition requirement repeatedly to the
retained training set until no more rows could be removed. Requiring two
occurrences across two conversations left 193 training and 35 development
examples. Three occurrences across two conversations left 91 and 16.
Five across three left 14 and 3. These are coverage estimates on the same
split. They indicate that raising the threshold inside this slice alone would
shrink the evaluation again.

The baseline-only reproduction regenerated both split files byte for byte and
matched both retrieval scores exactly. Use the following command with a new
output directory to run the split and controls without fitting T5.

```bash
OPENBLAS_NUM_THREADS=2 .runtime/t5-local-venv/bin/python -m training.lsfb_cont.frequent_context_experiment --baselines-only --output .runtime/lsfb-frequent-context-baselines-reproduction
```

## Completed run

Five epochs and 190 updates completed in 211 seconds after training setup.
Epoch four was the first minimum-WER checkpoint. Epoch five tied it. The
first epoch produced 79 invalid or empty output sequences. The selected
checkpoint produced 83 parseable sequences with no generation cap hits.

| Method | Exact sequences | Operation WER |
|---|---:|---:|
| Retrieval with dialogue context | 0/83 | 118.00% |
| Retrieval with current French turn | 2/83 | 105.33% |
| Empty output | 0/83 | 100.00% |
| Most frequent training sequence | 2/83 | 93.33% |
| Context T5-base, selected epoch 4 | 0/83 | 92.00% |

T5 saved four edits out of 300 reference operations versus always returning
the most frequent training sequence, and produced two fewer exact matches.
It emitted an average of 2.06 operations for references averaging 3.61.
This run fails to establish useful multi-sign translation. A matched T5 run
using the current turn alone would be required to estimate the effect of
adding context. The retrieval comparison only measures that effect for the
retrieval method.

The next useful dataset should contain repeated, reviewer-confirmed LSFB
constructions before model fitting, with enough examples left after holding
out whole conversations. The existing source review pack makes it possible
to check meaning and timing against video. None of its annotations has qualified
approval yet. France-LSF translation, 2D-to-3D lifting and signer comprehension
remain separate open parts of the wider project.

Run files and adapter are private under `.runtime/lsfb-frequent-context-v3/`.
Early run history labels the last batch group's loss as mean training loss.
The reproduction script now aggregates the loss across the epoch. This logging
change has no effect on model updates or measured development predictions.

Reloading the saved adapter from the pinned base reproduced all 83 scored
predictions and the exact metrics. The reproduction receipt is
`.runtime/lsfb-frequent-context-v3/RELOAD_VERIFICATION.json`.
