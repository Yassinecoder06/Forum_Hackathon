import json
from pathlib import Path
root=Path(__file__).resolve().parent;source=(root/'lsfb_poc.py').read_text();cells=[]
def md(s):cells.append({'cell_type':'markdown','metadata':{},'source':s.splitlines(True)})
def code(s):cells.append({'cell_type':'code','metadata':{},'execution_count':None,'outputs':[],'source':s.splitlines(True)})
md('''# SignaCulture: mT5 French → Belgian LSFB, with Google Drive checkpoints
Select **Runtime → Change runtime type → T4 GPU**, then run cells in order. This is an exploratory weak-alignment experiment, **not France-LSF translation**. The Dicta-Sign motion library is separate. No SSH or paid remote-control setup needed.

Uses official **training shard 000002**, never trains on official test shard 000000. Candidates are split by signer where possible, otherwise recording. This internal holdout is not the official benchmark. First run: 30 steps; change TRAIN_STEPS to 100 for a separate larger run after examining losses and predictions. Mount your own Drive interactively; no credentials go into this notebook.
''')
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess, sys, json, shutil
DRIVE = Path('/content/drive/MyDrive/SignaCulture/mt5-lsfb-poc')
DRIVE.mkdir(parents=True, exist_ok=True)
TRAIN_STEPS = 30
ALLOW_WEAK_LABELS = True  # Explicit exploratory labels; no human approval is implied.
RUN = DRIVE / f'mt5-small-steps-{TRAIN_STEPS}'
WORK = Path('/content/signaculture-lsfb'); WORK.mkdir(exist_ok=True)
print('Persistent checkpoints/model/results:', RUN)
print('Allow several GB of free Drive space; do not share private annotations or model weights automatically.')
''')
code('''subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'transformers==4.57.6', 'accelerate==1.15.0', 'sentencepiece==0.2.3', 'protobuf>=4.25', 'huggingface-hub==0.36.2'], check=True)
import torch
assert torch.cuda.is_available(), 'Choose a GPU runtime before training.'
print(torch.cuda.get_device_name(), torch.__version__)
subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=(DRIVE/'environment.txt').open('w'), check=True)
''')
code('SCRIPT = WORK / "lsfb_poc.py"\nSCRIPT.write_text('+repr(source)+', encoding="utf-8")\nprint(SCRIPT)\n')
code('''from huggingface_hub import hf_hub_download
DATA_REVISION = '64b0c112f59d3710367fb736ef1808f37ff15e28'
MODEL_REVISION = '73fb5dbe4756edadc8fbe8c769b0a109493acf7a'
archive = hf_hub_download('ppoitier/lsfb-cont', 'shards/annotated/shard_000002.tar', repo_type='dataset', revision=DATA_REVISION, local_dir=str(WORK/'raw'))
subprocess.run([sys.executable, str(SCRIPT), 'extract', archive, str(DRIVE/'candidates.jsonl'), '--limit', '300'], check=True)
(DRIVE/'source.json').write_text(json.dumps({'repository':'ppoitier/lsfb-cont','revision':DATA_REVISION,'shard':2,'official_split':'train','license':'CC-BY-NC-SA-4.0'},indent=2))
print((DRIVE/'candidates.jsonl.audit.json').read_text())
''')
md('''## Inspect candidates before training
These references come from timestamp overlap, not validated translations. Inspect the sample below; edit `candidates.jsonl` in Drive to remove bad alignments or mark reviewed examples `approved`. To require reviewed examples, set `ALLOW_WEAK_LABELS=False`. No poses or GAN are trained. If annotations are unusable, stop and improve the data.
''')
code('''import random
rows = [json.loads(line) for line in (DRIVE/'candidates.jsonl').read_text().splitlines()]
for row in random.Random(42).sample(rows, min(30,len(rows))):
    print(row['recording_id'], row['signer_id'], '\\nFR:', row['french_text'], '\\nLSFB:',row['gloss_sequence'], '\\n')
''')
code('''from transformers import AutoTokenizer
length_tokenizer=AutoTokenizer.from_pretrained('google/mt5-small',revision=MODEL_REVISION,use_fast=False)
path=DRIVE/'candidates.jsonl'
rows=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
kept=[];rejected=[]
for r in rows:
    n=max(len(length_tokenizer('translate French to LSFB glosses: '+r['french_text'])['input_ids']),len(length_tokenizer(text_target=r['gloss_sequence'])['input_ids']))
    (kept if n<=96 else rejected).append(r)
(DRIVE/'overlength-candidates.json').write_text(json.dumps(rejected,ensure_ascii=False,indent=2))
path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\\n' for r in kept))
print('Retained:',len(kept),'Overlength (retained separately):',len(rejected))
''')
code('''command=[sys.executable,str(SCRIPT),'split',str(DRIVE/'candidates.jsonl'),str(DRIVE/'splits')]
if ALLOW_WEAK_LABELS: command.append('--allow-weak')
subprocess.run(command,check=True)
print((DRIVE/'splits/split_report.json').read_text())
''')
md('''## Train and persist directly to Drive
FP16 is disabled. Adafactor and gradient checkpointing are enabled. Precision follows PyTorch capability detection: BF16 when supported (which newer runtimes can emulate on T4), otherwise FP32. The actual choice is saved in training_args.bin. Checkpoints include optimizer/trainer state every half-run; rerunning this cell resumes the latest complete checkpoint. If you change data/configuration, use a new run directory. Test predictions are generated after validation-based checkpoint selection.
''')
code('''subprocess.run([sys.executable,str(SCRIPT),'train',str(DRIVE/'splits'),str(RUN),'--steps',str(TRAIN_STEPS),'--revision',MODEL_REVISION],check=True)
assert (RUN/'COMPLETED.json').exists()
print((RUN/'metrics.json').read_text())
for example in json.loads((RUN/'predictions.json').read_text())[:10]: print(example)
print('Saved model/tokenizer:', RUN/'final')
''')
code('''from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
final=RUN/'final'
tokenizer=AutoTokenizer.from_pretrained(final,use_fast=False)
model=AutoModelForSeq2SeqLM.from_pretrained(final).to('cuda').eval()
text='Bonjour, comment allez-vous ?'  # Unseen input: output may be poor on this small PoC.
with torch.no_grad():
    inputs=tokenizer('translate French to LSFB glosses: '+text,return_tensors='pt').to('cuda')
    print(tokenizer.decode(model.generate(**inputs,max_new_tokens=96,num_beams=4)[0],skip_special_tokens=True))
''')
md('''## Saved artifacts and interpretation
Drive contains `candidates.jsonl`, extraction audit and source revision; frozen `splits/`; and each run's `checkpoints/`, `final/` model+tokenizer, `history.json`, `metrics.json`, `predictions.json`, `run_config.json` and `COMPLETED.json`. A completed run is not proof of usable translation. Compare against train-only retrieval; inspect empty/repetitive outputs. No upload/publishing happens.

Sources: [LSFB-CONT](https://huggingface.co/datasets/ppoitier/lsfb-cont), [mT5](https://huggingface.co/docs/transformers/model_doc/mt5). CC BY-NC-SA 4.0 source obligations apply; keep this experiment private while reviewing derivative terms.
''')
nb={'nbformat':4,'nbformat_minor':5,'metadata':{'colab':{'name':'SignaCulture_mT5_LSFB_Drive.ipynb'},'accelerator':'GPU','kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}},'cells':cells}
for i,c in enumerate(cells):c['id']=f'lsfb-{i:02d}'
(root/'SignaCulture_mT5_LSFB_Drive.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
for c in cells:
 if c['cell_type']=='code':compile(''.join(c['source']),'<cell>','exec')
print('Notebook generated; all code cells compile')

# Companion experiment reuses the existing Drive splits, preserving the comparison cohort.
mt5_cells=cells
cells=[]
md('''# SignaCulture: T5-base on the frozen LSFB splits
This companion reuses the splits already saved by the mT5 notebook or the SSH run. It does not re-extract or resplit data. Select a T4 GPU, run in order, and authorize Drive. Target is Belgian LSFB; weak corpus alignment is not linguistic validation.
''')
code('''from google.colab import drive
drive.mount('/content/drive')
from pathlib import Path
import subprocess,sys,json
DRIVE=Path('/content/drive/MyDrive/SignaCulture/mt5-lsfb-poc')
assert (DRIVE/'splits/split_report.json').is_file(), 'Prepare the frozen splits with the mT5 notebook first.'
WORK=Path('/content/signaculture-lsfb');WORK.mkdir(exist_ok=True)
TRAIN_STEPS=300
BASE_MODEL='google-t5/t5-base'  # Or: google/flan-t5-base
MODEL_REVISION={'google-t5/t5-base':'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1','google/flan-t5-base':'7bcac572ce56db69c1ea7c8af255c5d7c9672fc2'}[BASE_MODEL]
RUN=DRIVE/(BASE_MODEL.rsplit('/',1)[-1]+f'-steps-{TRAIN_STEPS}')
''')
code('''subprocess.run([sys.executable,'-m','pip','install','-q','transformers==4.57.6','accelerate==1.15.0','sentencepiece==0.2.3','huggingface-hub==0.36.2'],check=True)
import torch
assert torch.cuda.is_available(), 'Select a GPU runtime.'
print(torch.cuda.get_device_name())
''')
code('SCRIPT=WORK/"lsfb_poc.py"\nSCRIPT.write_text('+repr(source)+',encoding="utf-8")\n')
md('''## Train and select by validation
Identical 111/12/129 signer-disjoint examples as the mT5 comparison. T5 uses a 128-token limit to preserve every example under its tokenizer. Checkpoints save every 50 steps directly to Drive. Training resumes only complete checkpoints with a matching configuration. A 300-step model versus a 100-step model is an exploratory configuration comparison, not an equal-budget model benchmark.
''')
code('''subprocess.run([sys.executable,str(SCRIPT),'train',str(DRIVE/'splits'),str(RUN),'--base',BASE_MODEL,'--revision',MODEL_REVISION,'--steps',str(TRAIN_STEPS),'--max-length','128','--eval-every','50'],check=True)
print((RUN/'metrics.json').read_text())
for example in json.loads((RUN/'predictions.json').read_text())[:10]:print(example)
print('Persistent model/tokenizer:',RUN/'final')
''')
for i,c in enumerate(cells):
 c['id']=f't5-base-{i:02d}'
 if c['cell_type']=='code':compile(''.join(c['source']),'<t5-cell>','exec')
nb['cells']=cells;nb['metadata']['colab']['name']='SignaCulture_T5_Base_LSFB_Drive.ipynb'
(root/'SignaCulture_T5_Base_LSFB_Drive.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=2)+'\n')
print('T5-base companion generated against existing frozen Drive splits')
