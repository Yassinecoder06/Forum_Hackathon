#!/usr/bin/env python3
"""Start the recorded local concept model, without changing another server."""
import json
from pathlib import Path
import subprocess
import sys
import urllib.request

root=Path(__file__).resolve().parents[1]
manifest=root/'research/concept_planner_runtime.json'
if not manifest.is_file():
    sys.exit('No measured local planner manifest. See docs/SEMANTIC_SOLUTION.md.')
config=json.loads(manifest.read_text())
try:
    with urllib.request.urlopen(config['server_url']+'/v1/models',timeout=3) as response:
        models=json.load(response)['data']
    if not any(m.get('id')==config['server_model_alias'] for m in models):
        sys.exit('This port belongs to another model. No process was changed.')
    print('Recorded concept model is already running at '+config['server_url'])
except OSError:
    if not Path(config['model_path']).is_file():
        sys.exit('Recorded local GGUF model is missing. Install it explicitly; no automatic download.')
    try: sys.exit(subprocess.call(config['server_cli_args'],cwd=root))
    except FileNotFoundError: sys.exit('llama-server is not installed or not in PATH.')
