# Model selection — current measured status

Retain deterministic lookup and supported interpolation for experimental product behavior. No general text-to-gloss adapter has earned promotion.

The completed France-LSF comparison used 39 training / 28 development examples from one annotation convention. mT5-small achieved 0/28 exact and 594.85% WER; T5-base achieved 0/28 exact and 235.05% WER. Different pretraining/tokenizers and train-derived character adaptation prevent a pure size comparison. Neither is a usable translator. Zero local linguistic reviews are available.

mT5-base revision `2eb15465c5dd7f72a8f7984306ad05ebc3dd1e1f` remains an optional candidate, not a result. Its 1,747,976,192-byte partial file is incomplete against the expected 2,329,735,129 bytes. Current Colab session inspection finds no live session. Do not load partial weights or infer training completion. Additional acquisition must follow the user's Colab/download/permission rules, and is not currently justified as a cure for failed supervision.

Narrow Belgian LSFB current-text T5-base is the strongest inspected developmental lexical projection among these variants, but all three seeds still score 0/6 exact multi-sign examples. Keep it separate from France-LSF assets and do not promote it as a sentence translator. Full temporal GeoQuery-LSFB graph generation and frozen reranking failed to beat same-split retrieval.

For FSL 2D motion inbetweening, the full-loss GAN has a tiny position-error improvement and worse boundary velocity. Keep linear as the supported research baseline; no adversarial superiority or meaningful 3D transition benefit is established.

Select future models on reviewed, held-out task-appropriate performance, not parameter count. See FSL_MODEL_SIZE_COMPARISON.md, LSFB_SEED_REPLICATION_RESULT.md, GEOQUERY_FULL_GRAPH_RESULT.md, RESULTS.md and ../IMPLEMENTATION_STATUS.md.
