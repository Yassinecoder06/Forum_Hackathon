"""Read completed experiment reports and invoke only registered local checkpoints."""
import json
from pathlib import Path
import subprocess
from threading import BoundedSemaphore
from .capture_library import ROOT,corpus_status

PREDICTION_LOCK=BoundedSemaphore(1)

def relocated(path):
    """Map a path recorded before the 2026-11-20 restructure to its current location."""
    path=Path(path)
    try: rel=path.resolve().relative_to(ROOT).as_posix()
    except ValueError: return path
    try: moves=json.loads((ROOT/'research/PATH_MAP.json').read_text())['moves']
    except (ValueError,OSError,KeyError): return path
    for old in sorted(moves,key=len,reverse=True):
        if rel==old or rel.startswith(old+'/'): return ROOT/(moves[old]+rel[len(old):])
    return path

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
    checkpoint=relocated(report.get('checkpoint','')).resolve()
    allowed=[ROOT/'research/code/training/checkpoints',ROOT/'.worktrees/wt-codex2/training/checkpoints']
    if not any(checkpoint.is_relative_to(parent) for parent in allowed) or not (checkpoint/'research_metadata.json').is_file(): return None
    training=checkpoint.parents[1]
    choices=[training/'.venv-train/bin/python',training/'.venv-smoke/bin/python']
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
        'recent_experiments':recent_experiments(),'corpus':corpus_status(),'transition_experiment':transition_report(),'linguistic_review':'pending','translation_validated':False}

def predict(text,run_id):
    report=next((r for r in reports() if r['run_id']==run_id),None)
    if not report: raise FileNotFoundError('Expérience inconnue.')
    config=checkpoint_runtime(report)
    if not config: raise FileNotFoundError('Checkpoint ou environnement local absent.')
    checkpoint,runtime=config
    if not PREDICTION_LOCK.acquire(blocking=False): raise BlockingIOError('Une prédiction est déjà en cours.')
    try:
        result=subprocess.run([str(runtime),str(ROOT/'research/code/training/infer.py'),'--text',text,'--model',str(checkpoint),'--device','cpu'],capture_output=True,text=True,timeout=120,cwd=ROOT)
        if result.returncode: raise RuntimeError('Le modèle n’a pas pu produire de prédiction. Consultez les logs de l’expérience.')
        return json.loads(result.stdout.strip().splitlines()[-1])
    finally: PREDICTION_LOCK.release()


def recent_experiments():
    """Read-only research ledger summaries; never register these adapters for inference."""
    import hashlib
    import math
    import os
    import sqlite3
    from contextlib import closing
    path=Path(os.getenv('SIGNACULTURE_RESEARCH_CATALOG',str(ROOT/'data/catalog/catalog.sqlite')))
    if not path.is_file():return {'available':False,'entries':[],'reason':'catalog_absent'}
    try:
        with closing(sqlite3.connect(path.resolve().as_uri()+'?mode=ro',uri=True)) as db:
            rows=db.execute('SELECT m.run_id,m.model_base,m.hyperparameters_json,m.metrics_json,i.sign_language,i.representation,i.result_sha256,i.report_path FROM model_runs m JOIN research_experiment_index i ON m.run_id=i.run_id ORDER BY m.run_id').fetchall()
    except sqlite3.DatabaseError:return {'available':False,'entries':[],'reason':'index_unavailable'}
    entries=[]
    for run,model,config_raw,metrics_raw,language,representation,digest,report in rows:
        if language not in ('fsl','sfb'):continue
        entry={'run_id':run,'model_base':model,'sign_language':language,'representation':representation,'evidence_status':'unavailable','inference_ready':False,'linguistic_review':'pending','evaluation_scope':'development; inspected during model selection','metrics':None}
        report=relocated(report).resolve()
        if not any(report.is_relative_to(parent) for parent in (ROOT/'.runtime',ROOT/'research/private')):
            entry['evidence_status']='outside_allowed_roots';entries.append(entry);continue
        try:
            if report.stat().st_size>1024*1024:raise ValueError('Oversized report')
            if hashlib.sha256(report.read_bytes()).hexdigest()!=digest:
                entry['evidence_status']='changed';entries.append(entry);continue
            config=json.loads(config_raw);raw=json.loads(metrics_raw);proof=json.loads(report.read_bytes())
            if proof.get('status')!='complete' or raw!=proof.get('validation',proof.get('metrics',proof)):
                entry['evidence_status']='metrics_mismatch';entries.append(entry);continue
            expected='sfb' if config.get('language')=='LSFB' or config.get('task')=='French machine-translated input to full LSFB graph; feasibility' else 'fsl' if config.get('language')=='fsl' else None
            if expected!=language:
                entry['evidence_status']='language_mismatch';entries.append(entry);continue
            def number(value):return float(value) if type(value) in (int,float) and math.isfinite(value) else None
            graph=representation=='temporal_graph'
            metrics={'examples':number(raw.get('examples')),'exact':number(raw.get('graph_exact' if graph else 'exact')),'wer':None if graph else number(raw.get('operation_wer',raw.get('wer'))),'triple_f1':number(raw.get('triple_micro_f1')) if graph else None,'valid_graphs':number(raw.get('valid_graphs')) if graph else None}
            multi=raw.get('by_length',{}).get('multiple_signs',{})
            metrics['multi_sign_examples']=number(multi.get('examples'));metrics['multi_sign_exact']=number(multi.get('exact'))
            entry.update(evidence_status='verified',metrics=metrics,train_examples=number(config.get('train')),input_provenance='DeepL-translated French' if graph else 'French free translation paired with source annotations',limitations=['No qualified local signer approval','Results across tasks/splits are not directly comparable','Not enabled for signing or avatar inference'])
        except (OSError,ValueError,TypeError,AttributeError):entry['evidence_status']='unreadable'
        entries.append(entry)
    return {'available':True,'entries':entries,'translation_validated':False}
