"""Read completed experiment reports and invoke only registered local checkpoints."""
import json
from pathlib import Path
import subprocess
from threading import BoundedSemaphore
from .capture_library import ROOT,corpus_status

PREDICTION_LOCK=BoundedSemaphore(1)

def reports():
    result=[]
    for path in sorted((ROOT/'research/runs').glob('*/metrics.json')):
        try: value=json.loads(path.read_text())
        except (ValueError,OSError): continue
        if not isinstance(value,dict) or 'model_base' not in value or 'test' not in value.get('metrics',{}): continue
        value['inference_ready']=checkpoint_runtime(value) is not None
        result.append(value)
    return result

def checkpoint_runtime(report):
    checkpoint=Path(report.get('checkpoint','')).resolve()
    allowed=[ROOT/'training/checkpoints',ROOT/'.worktrees/wt-codex2/training/checkpoints']
    if not any(checkpoint.is_relative_to(parent) for parent in allowed) or not (checkpoint/'research_metadata.json').is_file(): return None
    owner=checkpoint.parents[2]
    choices=[owner/'training/.venv-train/bin/python',owner/'training/.venv-smoke/bin/python']
    runtime=next((p for p in choices if p.is_file()),None)
    return (checkpoint,runtime) if runtime else None

def transition_report():
    path=ROOT/'research/runs/transition-intrasign-v1/metrics.json'
    try:
        value=json.loads(path.read_text())
        return {k:value.get(k) for k in ['kind','counts','metrics','training_seconds','paper_score_comparison']}
    except (ValueError,OSError): return None

def status():
    manifests=[]
    for path in sorted((ROOT/'data/text_to_gloss').glob('v*/dataset_manifest.json')):
        try: manifests.append(json.loads(path.read_text()))
        except (ValueError,OSError): pass
    inventory=ROOT/'data/source_inventory/commons_lsf.json'
    try: lexical=len(json.loads(inventory.read_text())['entries'])
    except (ValueError,OSError,KeyError): lexical=0
    return {'lexical_video_references':lexical,'datasets':manifests,'runs':reports(),
        'corpus':corpus_status(),'transition_experiment':transition_report(),'linguistic_review':'pending','translation_validated':False}

def predict(text,run_id):
    report=next((r for r in reports() if r['run_id']==run_id),None)
    if not report: raise FileNotFoundError('Expérience inconnue.')
    config=checkpoint_runtime(report)
    if not config: raise FileNotFoundError('Checkpoint ou environnement local absent.')
    checkpoint,runtime=config
    if not PREDICTION_LOCK.acquire(blocking=False): raise BlockingIOError('Une prédiction est déjà en cours.')
    try:
        result=subprocess.run([str(runtime),str(ROOT/'training/infer.py'),'--text',text,'--model',str(checkpoint),'--device','cpu'],capture_output=True,text=True,timeout=120,cwd=ROOT)
        if result.returncode: raise RuntimeError('Le modèle n’a pas pu produire de prédiction. Consultez les logs de l’expérience.')
        return json.loads(result.stdout.strip().splitlines()[-1])
    finally: PREDICTION_LOCK.release()
