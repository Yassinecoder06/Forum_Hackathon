"""Package compact text data for Colab; do not upload pose/video archives."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path


PREFLIGHT = '''import hashlib, json
from pathlib import Path
from tokenizers import Tokenizer
from prepare import parse_plan
root = Path(__file__).parent
manifest = json.loads((root/'bundle_manifest.json').read_text())
for name, sha in manifest['files'].items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest() == sha, name
tokenizer = Tokenizer.from_file(str(root/'tokenizer.json'))
splits = {}
for name in ['train','val']:
    rows = [json.loads(l) for l in (root/'data'/f'{name}.jsonl').read_text().splitlines()]
    assert rows
    for row in rows:
        src = tokenizer.encode(row['input_text'])
        target = tokenizer.encode(row['target_text'])
        assert len(src.ids) <= 512 and len(target.ids) <= 512
        assert tokenizer.token_to_id('<unk>') not in target.ids
        assert parse_plan(tokenizer.decode(target.ids)) == parse_plan(row['target_text'])
    splits[name] = rows
assert not {r['conversation_id'] for r in splits['train']} & {r['conversation_id'] for r in splits['val']}
print(json.dumps({'mode':manifest['mode'],'train':len(splits['train']),'val':len(splits['val']), 'added_model_tokens':0,'tokenization_and_split_checks':'passed','linguistically_reviewed':manifest['mode']=='reviewed'},indent=2))
if manifest['mode'] != 'reviewed':
    print('EXPLORATORY DATA: technical testing only; no linguistic approval or translation quality claim.')
'''


def bundle(dataset, output, mode):
    manifest = json.loads((dataset/'manifest.json').read_text())
    if mode == 'reviewed' and manifest.get('mode') != 'reviewed':
        raise ValueError('Use release.py to create a reviewed release first')
    if mode == 'exploratory' and manifest.get('mode') == 'reviewed':
        raise ValueError('Use reviewed mode for a reviewed release')
    if 'files' in manifest:
        for name, sha in manifest['files'].items():
            if hashlib.sha256((dataset/name).read_bytes()).hexdigest() != sha:
                raise ValueError('Frozen dataset changed: '+name)
    base = dataset/'exploratory' if mode == 'exploratory' else dataset
    files = {f'data/{name}.jsonl': (base/f'{name}.jsonl').read_bytes() for name in ['train', 'val']}
    tok = Path(manifest['tokenizer']['path'])
    if hashlib.sha256(tok.read_bytes()).hexdigest() != manifest['tokenizer']['sha256']:
        raise ValueError('Tokenizer checksum mismatch')
    files['tokenizer.json'] = tok.read_bytes()
    files['prepare.py'] = Path(__file__).with_name('prepare.py').read_bytes()
    files['colab_preflight.py'] = PREFLIGHT.encode()
    files['source_manifest.json'] = (dataset/'manifest.json').read_bytes()
    bundle_manifest = {'mode': mode, 'task': manifest['task'], 'model': 'google-t5/t5-base',
                       'model_revision': manifest['tokenizer']['revision'],
                       'test_included': False, 'training_started': False,
                       'reviewed_data_available': mode == 'reviewed',
                       'full_training_ready': False,
                       'requires_learning_gate': True,
                       'files': {k: hashlib.sha256(v).hexdigest() for k, v in files.items()}}
    files['bundle_manifest.json'] = json.dumps(bundle_manifest, ensure_ascii=False, indent=2).encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError('Bundle already exists; choose a new output name')
    with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as z:
        for name, content in files.items():
            z.writestr(name, content)
    print(json.dumps({'bundle': str(output), 'bytes': output.stat().st_size, 'mode': mode}))


def notebook(path):
    cells = []
    def cell(kind, text):
        item = {'cell_type': kind, 'metadata': {}, 'source': text.splitlines(True)}
        if kind == 'code':
            item.update(execution_count=None, outputs=[])
        cells.append(item)
    cell('markdown', '# LSFB-CONT lexical-plan data preflight\nUse a GPU Colab runtime for subsequent T5 training. This notebook validates the prepared data only. An exploratory bundle has no linguistic approval. Keep any later training outputs on Drive and retain a local backup.\n')
    cell('code', '%pip -q install tokenizers==0.22.2\n')
    cell('code', 'from google.colab import drive, files\nfrom pathlib import Path\nimport zipfile\ndrive.mount("/content/drive")\nuploaded = files.upload()\narchive = next(name for name in uploaded if name.endswith(".zip"))\nwork = Path("/content/lsfb_context_plan")\nwork.mkdir(exist_ok=True)\nwith zipfile.ZipFile(archive) as z:\n    for name in z.namelist():\n        assert not Path(name).is_absolute() and ".." not in Path(name).parts\n    z.extractall(work)\nbackup = Path("/content/drive/MyDrive/SignaCulture/lsfb-cont-context-plan")\nbackup.mkdir(parents=True, exist_ok=True)\n(backup/Path(archive).name).write_bytes(Path(archive).read_bytes())\n')
    cell('code', '!python /content/lsfb_context_plan/colab_preflight.py\n')
    cell('markdown', '## Training handoff\nLoad `data/train.jsonl` and `data/val.jsonl` using `input_text` and `target_text`. Use the pinned `google-t5/t5-base` revision from `bundle_manifest.json` with its existing tokenizer; add no gloss tokens. Use maximum source/target lengths of 512 and never silently truncate labels. Keep `conversation_id` for split assertions. First measure generated training-example accuracy and validation behavior on a small learning gate. A full quality run uses a reviewed release created by `release.py`, not an automatically approved exploratory export. This dataset predicts a lexical projection, not full signing or avatar motion.\n')
    path.write_text(json.dumps({'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'accelerator': 'GPU'}, 'nbformat': 4, 'nbformat_minor': 5}, indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=['exploratory', 'reviewed'], required=True)
    p.add_argument('--notebook', type=Path)
    a = p.parse_args()
    bundle(a.dataset, a.output, a.mode)
    if a.notebook:
        notebook(a.notebook)
