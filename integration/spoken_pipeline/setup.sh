#!/usr/bin/env bash
set -euo pipefail
adapter_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
upstream_dir="$adapter_dir/runtime/upstream"
revision='fec7ac393989f4b6541ab5f860819262a54f27c7'
mkdir -p "$adapter_dir/runtime"
if [[ ! -d "$upstream_dir/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/sign-language-processing/spoken-to-signed-translation.git "$upstream_dir"
  git -C "$upstream_dir" checkout --detach "$revision"
fi
[[ "$(git -C "$upstream_dir" rev-parse HEAD)" == "$revision" ]] || { echo 'Unexpected upstream revision; refusing to alter checkout' >&2; exit 1; }
if [[ ! -x "$adapter_dir/.venv/bin/python" ]]; then uv venv --python 3.11 "$adapter_dir/.venv"; fi
uv pip install --python "$adapter_dir/.venv/bin/python" -r "$adapter_dir/requirements.lock.txt"
uv pip install --python "$adapter_dir/.venv/bin/python" --no-deps -e "$upstream_dir"
