#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
pose_work="$PWD/.runtime/pose-comparison"
socket_dir="$PWD/.runtime/motion-inference"
mkdir -p "$socket_dir"
export PYTHONPATH="$pose_work/inference-deps:$pose_work/hamer-deps:$pose_work/runtime-deps:$pose_work/chumpy-source/chumpy-0.70"
export PYOPENGL_PLATFORM=egl
export SIGNACULTURE_HOST_CAPTURE_STORAGE
SIGNACULTURE_HOST_CAPTURE_STORAGE="$(docker compose config --format json | python3 -c '
import json, sys
studio = json.load(sys.stdin)["services"]["studio"]
storage = studio["environment"]["SIGNACULTURE_STORAGE"]
print(next(volume["source"] for volume in studio["volumes"] if volume["target"] == storage))
')"
exec .runtime/t5-local-venv/bin/python -m uvicorn motion_worker.server:app --uds "$socket_dir/motion.sock"
