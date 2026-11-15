"""Persistent local captures and a curated gloss/motion library."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
try:
    import fcntl
except ImportError:  # Windows: msvcrt byte-range lock as the flock equivalent
    import msvcrt
    class fcntl:
        LOCK_EX, LOCK_UN = msvcrt.LK_LOCK, msvcrt.LK_UNLCK
        @staticmethod
        def flock(handle, mode):
            handle.seek(0)
            msvcrt.locking(handle.fileno(), mode, 1)
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
from threading import Lock
import uuid

ROOT=Path(__file__).resolve().parents[3]
POOL=ThreadPoolExecutor(max_workers=1,thread_name_prefix='capture')
LOCK=Lock()
ACTIVE=set()

def storage():
    p=Path(os.getenv('SIGNACULTURE_STORAGE',str(ROOT/'.runtime'))).resolve()
    p.mkdir(parents=True,exist_ok=True)
    return p

def write_json(path,value):
    temp=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,allow_nan=False))
    temp.replace(path)

@contextmanager
def library_lock():
    with LOCK:
        with (storage()/'library.lock').open('a') as handle:
            fcntl.flock(handle,fcntl.LOCK_EX)
            try: yield
            finally: fcntl.flock(handle,fcntl.LOCK_UN)

def library():
    p=storage()/'library.json'
    return json.loads(p.read_text()) if p.exists() else []

def runtime_python(kind):
    override=os.getenv(f'SIGNACULTURE_{kind.upper()}_PYTHON')
    return Path(override) if override else ROOT/f'.venv-{kind}'/'bin/python'

def capabilities():
    entries=library()
    return {'capture':runtime_python('motion').is_file(),
        'emotion':runtime_python('audio').is_file() and (ROOT/'.models/emotion2vec-seed/model.pt').is_file(),
        'motion_model':'MediaPipe Holistic','emotion_model':'emotion2vec+ seed',
        'library_count':len(entries),'cv_corrected_count':sum(bool(r.get('cv_generation')) for r in entries),
        'corpus':corpus_status(),'dictasign':dictasign_status()}

def corpus_status():
    path=storage()/'corpus_status.json'
    if not path.exists(): return None
    try:
        value=json.loads(path.read_text())
        return {k:v for k,v in value.items() if k!='results'}
    except (OSError,ValueError): return None

def create_job(source_path,original_name):
    job_id=source_path.parent.name
    job={'job_id':job_id,'status':'queued','progress':0,'filename':original_name,
         'created_at':datetime.now(timezone.utc).isoformat()}
    write_json(source_path.parent/'status.json',job)
    with LOCK: ACTIVE.add(job_id)
    POOL.submit(process_job,job,source_path)
    return job

def process_job(job,source):
    folder=source.parent
    job['status']='processing';write_json(folder/'status.json',job)
    try:
        result=subprocess.run([str(runtime_python('motion')),str(ROOT/'backend/app/workers/capture.py'),str(source),str(folder/'motion.json'),str(folder/'progress.json')],capture_output=True,text=True,timeout=240,cwd=ROOT)
        if result.returncode:
            error=folder/'motion.error'
            raise ValueError(error.read_text() if error.exists() else 'Extraction impossible. Vérifiez le runtime MediaPipe et le format vidéo.')
        motion=json.loads((folder/'motion.json').read_text())
        job.update(status='ready',progress=100,duration_ms=motion['duration_ms'],quality=motion['quality'],warnings=motion['warnings'])
    except subprocess.TimeoutExpired:
        job.update(status='failed',error='Délai d’extraction dépassé (4 minutes). Réduisez la durée ou la résolution du clip.')
    except Exception as exc:
        job.update(status='failed',error=str(exc))
    finally:
        write_json(folder/'status.json',job)
        with LOCK: ACTIVE.discard(job['job_id'])

def job_folder(job_id):
    try: parsed=uuid.UUID(job_id)
    except (ValueError,AttributeError): raise ValueError('Capture inconnue')
    p=storage()/'captures'/str(parsed)
    if not p.is_dir(): raise FileNotFoundError('Capture inconnue')
    return p

def get_job(job_id):
    p=job_folder(job_id)
    job=json.loads((p/'status.json').read_text())
    if job['status'] in ('processing','queued'):
        with LOCK: active=job_id in ACTIVE
        if not active:
            job.update(status='failed',error='Extraction interrompue par un redémarrage. Réimportez le clip.')
        elif (p/'progress.json').exists():
            try: job['progress']=json.loads((p/'progress.json').read_text())['progress']
            except (ValueError,OSError): pass
    return job

def register(job_id,concept,aliases,source,license_name):
    folder=job_folder(job_id)
    if get_job(job_id)['status']!='ready': raise ValueError('Terminez l’extraction avant d’enregistrer le signe.')
    motion=json.loads((folder/'motion.json').read_text())
    identifier='CAPTURE_'+job_id.replace('-','')[:16]
    target=storage()/'motions'/f'{identifier}.json';target.parent.mkdir(exist_ok=True)
    write_json(target,motion)
    gloss={'gloss_id':'LSF_'+identifier,'concept_fr':concept.strip(),'gloss':concept.strip().upper(),
        'language':'LSF','variant':'capture locale à relire','source_id':identifier,
        'motion_id':identifier,'aliases':aliases,'review_status':'pending'}
    asset={'motion_id':identifier,'gloss_id':gloss['gloss_id'],'path':str(target),
        'rig_version':'signaculture-v1','fps':motion['fps'],'duration_ms':motion['duration_ms'],
        'license':license_name,'quality_status':'captured_pending_review','rights_status':'pending',
        'animation_review':'pending','synthetic':bool(motion.get('synthetic',False))}
    entry={'gloss':gloss,'motion':asset,'source':source,'capture_id':job_id,'quality':motion['quality']}
    with library_lock():
        entries=[x for x in library() if x['motion']['motion_id']!=identifier]
        entries.append(entry);write_json(storage()/'library.json',entries)
    return entry

def remove_entry(motion_id):
    with library_lock():
        entries=library()
        match=next((x for x in entries if x['motion']['motion_id']==motion_id),None)
        if not match: raise FileNotFoundError('Mouvement inconnu')
        write_json(storage()/'library.json',[x for x in entries if x['motion']['motion_id']!=motion_id])
        path=Path(match['motion']['path']).resolve()
        if path.is_relative_to(storage()/'motions'):
            path.unlink(missing_ok=True)
    # The original capture is retained, so the user can register it again.


def dictasign_status():
    path=storage()/'dictasign_status.json'
    if not path.is_file():return None
    try:return {k:v for k,v in json.loads(path.read_text()).items() if k!='results'}
    except (OSError,ValueError):return None
