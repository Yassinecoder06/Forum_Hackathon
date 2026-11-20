#!/usr/bin/env bash
# Adapted experiment, not a bit-for-bit reproduction of ACL 2023.
set -euo pipefail
DATA=${1:?parallel data directory required}
OUT=${2:?new output directory required}
MARIAN=${MARIAN:-marian}
command -v "$MARIAN" >/dev/null
command -v spm_train >/dev/null
if [[ -e "$OUT" ]]; then echo 'Refusing existing output directory' >&2; exit 1; fi
mkdir -p "$OUT"
cat "$DATA/train.fr" "$DATA/train.gloss" > "$OUT/tokenizer_train.txt"
spm_train --input="$OUT/tokenizer_train.txt" --model_prefix="$OUT/joint" --model_type=bpe --vocab_size=2000 --character_coverage=1.0 --hard_vocab_limit=false
"$MARIAN" --devices 0 --type transformer --model "$OUT/model.npz" \
 --train-sets "$DATA/train.fr" "$DATA/train.gloss" --valid-sets "$DATA/val.fr" "$DATA/val.gloss" \
 --vocabs "$OUT/joint.model" "$OUT/joint.model" --tied-embeddings-all \
 --dim-emb 512 --transformer-dim-ffn 2048 --transformer-heads 8 --enc-depth 1 --dec-depth 2 \
 --mini-batch 32 --max-length 512 --learn-rate 0.0003 --lr-warmup 400 \
 --optimizer adam --optimizer-params 0.9 0.98 1e-09 --clip-norm 5 \
 --transformer-dropout 0.2 --label-smoothing 0.2 --beam-size 6 --normalize 0.6 \
 --valid-metrics ce-mean-words perplexity --valid-freq 200 --early-stopping 10 \
 --after-epochs 40 --disp-freq 50 --save-freq 200 --keep-best --seed 1111 \
 --log "$OUT/train.log" --valid-log "$OUT/valid.log"
