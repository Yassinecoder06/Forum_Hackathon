#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
case "${1:-all}" in
  motion|all)
    uv venv --python 3.11 --allow-existing .venv-motion
    uv pip install --python .venv-motion/bin/python -r experiments/motion_extraction/requirements-optional.txt
    ;;
esac
case "${1:-all}" in
  audio|all)
    uv venv --python 3.11 --allow-existing .venv-audio
    uv pip install --python .venv-audio/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    uv pip install --python .venv-audio/bin/python funasr librosa soundfile 'numpy<2' einops
    hf download emotion2vec/emotion2vec_plus_seed --revision 7be9a30d52bdc793cc24e6f7d25358d6648927c1 config.yaml configuration.json tokens.txt model.pt --local-dir .models/emotion2vec-seed --max-workers 1
    ;;
esac
