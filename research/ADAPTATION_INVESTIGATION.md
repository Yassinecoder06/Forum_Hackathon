# mT5 adaptation follow-up

The user requested one longer, validation-selected mT5 LoRA run after the eight-epoch experiment collapsed to sentinel output. This follow-up keeps the licensed source-annotated research scope, every earlier run, and the frozen v002 splits. It is not a new translation benchmark. Earlier historical test results existed before this follow-up, so the final fixed test is a reused engineering test; it is not an independent confirmatory evaluation.

## Training-only checks

`research/runs/mt5-train-only-diagnostic/tokenizer_audit.json` records all 61 actual training targets and their token IDs/pieces. Every target round-trips exactly. There are zero unknown tokens and zero sentinel tokens in targets. All end in EOS. Padding is masked with -100, decoder inputs start at ID 0 and correctly shift labels, and source/target identity matches the authentic catalog. No target gloss normalization, synthetic pair generation or new approval occurs.

`source_tokenizer_audit.json` separately checks the French inputs. There are zero source unknown tokens. Maximum training lengths are 105 input tokens and 138 target tokens, below the 512-token training limit; no training truncation occurs. Tokenizer length 250,100 fits the base model's padded 250,112-row vocabulary. The official tokenizer config sets extra_ids=0. Sentinel-looking strings remain vocabulary tokens rather than configured special tokens, which explains why skip_special_tokens does not necessarily remove them. Filtering them would conceal a model failure, so decoding is unchanged. The generation cap remains the pre-registered 128 new tokens; this can limit unusually long outputs and is a documented limitation, independent of training truncation.

The four-example diagnostic selected the first four TRAIN pair IDs in sorted order. It read no validation or test records. LoRA targets q/k/v/o/wi_0/wi_1/wo with rank 16, alpha 32 and dropout 0.05, at learning rate 0.001. All trainable tensors are LoRA A/B weights; pretrained embeddings and output head stay frozen. Parameter count before freeze: 2,555,904. First pre-clipping gradient norm was nonzero (3419.33). After 200 optimizer steps, teacher-forced loss fell from 36.1055 to 0.1103; greedy train exact match was 2/4 and WER 0.4211. This demonstrates substantial adaptation and escape from sentinel output, but not complete memorization. Diagnostic runtime 94.07 seconds; peak allocated 1681.74 MiB. It does not demonstrate generalization.

## One locked longer run

`MT5_LONG_RUN_PLAN.json` was written before the longer run started. It records reasons, full settings, frozen data hashes and historical artifact hashes. No hyperparameter sweep is run. Settings: fresh pinned mT5-small base, rank 16 attention/feed-forward LoRA, alpha 32, dropout 0.05, AdamW learning rate 0.001, weight decay 0, batch 1, accumulation 4, BF16, seed 42, 5% optimizer-step warmup then linear decay, maximum 100 epochs, validation-loss patience 10. Greedy generation and the 128-new-token cap remain unchanged; no constraints or sentinel suppression are introduced.

Two training-loop details are made explicit: the last accumulation group is divided by its actual example count, and validation loss is weighted by the number of non-padding target tokens. The former corrects underweighting the final single example. The latter avoids giving short and long targets equal sequence weight when selecting a checkpoint. Logs retain the old mean-per-example validation loss as well, allowing a like-for-like loss comparison with the earlier run. These are planned before training/evaluation, not reactions to test output.

The script counts trainable parameters and stores tensor names before loading/freezing the selected adapter. It parses only train and validation during training. Once validation selection finishes, it writes selection_complete.json, then opens and decodes the frozen test exactly once. Full loss history and all held-out predictions are retained. Checkpoint hashes and explicit pending-review status are part of inference metadata. Test results must not trigger another configuration change.

## Observed result and decision

The one longer run stopped at epoch 21, after ten epochs without improving the token-weighted validation loss. Epoch 11 was selected: token-weighted loss 3.294314; historical mean-per-example loss 3.214754, versus 13.922054 in the eight-epoch run. There were 336 optimizer steps executed (176 at the selected checkpoint). Pre-freeze trainable parameter count was 2,555,904. Training took 113.9165 seconds; training plus validation/test generation took 246.7427 seconds. Peak CUDA allocated/reserved memory was 1961.35 / 2250 MiB.

| Final 39-example test | Exact match | Token WER | Corpus BLEU (0–100) | Predicted OOV vs training gloss labels |
|---|---|---|---|---|
| Longer mT5 LoRA | 0% | 517.9612% | 0.042209 | 69.2600% |
| Same-data retrieval | 0% | 120.8738% | 0.300964 | 0% by construction |
| Same-data lexical lookup | 0% | 193.2039% | 0.193524 | 86.7322% |

The test reference OOV rate against the training gloss vocabulary is 82.5243%; this is not tokenizer unknown-token rate. No approved motion-to-gloss map exists, so motion-registry OOV remains undefined rather than inferred from the lexical-video inventory.

Sentinel output disappeared on all 39 final test examples, and 28 different raw outputs were produced. However, outputs are largely repeated or malformed gloss strings, sometimes containing fabricated labels. The model produced 1054 whitespace tokens against 206 reference tokens. The large insertion count explains WER above 100%. All predictions are retained without filtering or correction. A representative failure repeats `[DÉPLACE]` instead of producing the source-annotated utterance; failure_analysis.json links exact examples and references. Generated labels are unapproved model outputs, not new corpus annotations.

Conclusion: the eight-epoch run did not exhaust adaptation, and the training-only checks do not support a label/tokenization corruption explanation. More capacity and training reduced teacher-forced loss and removed sentinel collapse, but autoregressive repetition and held-out failure remain. The longer run is worse than both baselines on WER and BLEU. These results do not establish which of corpus size/domain shift, representation limits or remaining model/training limitations causes each error; several settings changed together. No further run, decoding change, test-driven tuning or claim of general-purpose French-to-LSF translation is made. Existing controlled lookup remains the appropriate integration path.

Twelve verification tests passed, including locked settings, pre-freeze parameter count, validation-only checkpoint selection and ten-epoch stopping, exact data/prior-artifact hashes, and complete 39-example test output. A CUDA CLI call also used the selected checkpoint with checksum verification and pending-review output. Weights/environments remain ignored; only code, licensed textual artifacts and reports are committed.
