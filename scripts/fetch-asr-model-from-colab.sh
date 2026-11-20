#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

session="${1:-signaculture-asr-model}"
model_id="Systran/faster-whisper-small"
revision="536b0662742c02347bc0e980a01041f333bce120"
remote_dir="/content/signaculture-asr-model"
model_dir=".models/faster-whisper-small"
transfer_dir=".runtime/asr-model-transfer"

mkdir -p "$model_dir" "$transfer_dir/parts"
colab ssh --session "$session" -- bash -lc \
  "hf download $model_id --revision $revision --local-dir $remote_dir"
colab ssh --session "$session" -- bash -lc \
  "split -b 16000000 $remote_dir/model.bin /content/asr-model-part-"

python3 - "$session" "$model_dir" "$transfer_dir/parts" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

session, model_dir, parts_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
manifest = json.loads(Path("configs/asr_model.json").read_text())
for index in range(31):
    suffix = chr(97 + index // 26) + chr(97 + index % 26)
    subprocess.run(
        [
            "colab", "download", "--session", session,
            f"/content/asr-model-part-{suffix}", str(parts_dir / suffix),
        ],
        check=True,
    )

part_file = parts_dir / "model.bin.assembled"
with part_file.open("wb") as output:
    for index in range(31):
        suffix = chr(97 + index // 26) + chr(97 + index % 26)
        with (parts_dir / suffix).open("rb") as source:
            while chunk := source.read(1024 * 1024):
                output.write(chunk)
digest = hashlib.sha256(part_file.read_bytes()).hexdigest()
expected = manifest["file_hashes"]["model.bin"]
if digest != expected or part_file.stat().st_size != manifest["model_bin_bytes"]:
    raise SystemExit("Transferred ASR model does not match its pinned Hub digest")
part_file.replace(model_dir / "model.bin")

for name, expected in manifest["file_hashes"].items():
    if name == "model.bin":
        continue
    subprocess.run(
        [
            "colab", "download", "--session", session,
            f"{remote_dir}/{name}", str(model_dir / name),
        ],
        check=True,
    )
    digest = hashlib.sha256((model_dir / name).read_bytes()).hexdigest()
    if digest != expected:
        raise SystemExit(f"Transferred ASR file failed hash verification: {name}")
print(f"Verified {manifest['model_id']} at {manifest['revision']}")
PY
