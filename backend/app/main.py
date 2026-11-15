"""Local experimental lookup API; no translation or emotion model claims."""
import json
import os
from pathlib import Path
import sqlite3
import uuid
from datetime import datetime, timezone

import numpy as np
import subprocess
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
from .schemas import GlossEntry, MotionAsset, TranslationDraft, AnimationPlan, AudioFeatures, ReviewRecord

from .services.text_lookup import lookup_text, mode
from .services import capture_library

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title='SignaCulture experimental API')
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('SIGNACULTURE_CORS_ORIGINS', 'http://localhost:5173').split(','), allow_methods=['*'], allow_headers=['*'])

class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    audio_features: AudioFeatures | None = None
    include_synthetic: bool = False

class PlanRequest(BaseModel):
    plan_id: str
    purpose: Literal['preview', 'publication'] = 'preview'

class ReviewRequest(BaseModel):
    plan_id: str
    reviewer: str = Field(min_length=1)
    status: Literal['approved', 'rejected']
    notes: str

@app.middleware('http')
async def identify_fallback(request, call_next):
    response = await call_next(request)
    response.headers['X-NLP-Backend'] = mode()
    return response

def registry():
    root = Path(os.getenv('SIGNACULTURE_DATA_ROOT', str(ROOT)))
    try:
        gloss = [GlossEntry.model_validate(x) for x in json.loads((root/'data/glossary.json').read_text())]
        motions = [MotionAsset.model_validate(x) for x in json.loads((root/'data/motions_manifest.json').read_text())]
        imported = capture_library.library()
        concepts = {x['gloss']['concept_fr'].casefold() for x in imported}
        replaced = {g.motion_id for g in gloss if g.concept_fr.casefold() in concepts}
        gloss = [g for g in gloss if g.motion_id not in replaced]
        motions = [m for m in motions if m.motion_id not in replaced]
        gloss.extend(GlossEntry.model_validate(x['gloss']) for x in imported)
        motions.extend(MotionAsset.model_validate(x['motion']) for x in imported)
        if len({x.motion_id for x in motions}) != len(motions) or len({x.gloss_id for x in gloss}) != len(gloss):
            raise ValueError('Duplicate IDs')
        for g in gloss:
            if not any(m.motion_id == g.motion_id and m.gloss_id == g.gloss_id for m in motions):
                raise ValueError('Broken gloss reference')
        return root, gloss, motions
    except (ValueError, OSError) as e:
        raise HTTPException(503, f'Registry unavailable: {e}')

def asset(root, m):
    p = (root/m.path).resolve()
    allowed = p.is_relative_to((root/'data/approved_motion_assets').resolve()) or p.is_relative_to(capture_library.storage()/'motions')
    if not allowed or not p.is_file():
        raise HTTPException(503, 'Asset unavailable or outside registered directory')
    try:
        d = json.loads(p.read_text())
        if d['rig_version'] != m.rig_version or d['duration_ms'] != m.duration_ms or d['fps'] != m.fps:
            raise ValueError('Metadata mismatch')
        if not d['frames'] or not d['joints'] or d['frames'][0]['time_ms'] != 0 or d['frames'][-1]['time_ms'] != m.duration_ms:
            raise ValueError('Invalid endpoints')
        last = -1
        for f in d['frames']:
            if not last < f['time_ms'] <= m.duration_ms or not set(d['joints']).issubset(f['joints']):
                raise ValueError('Invalid joints or timestamps')
            last = f['time_ms']
            for v in f['joints'].values():
                if len(v) != 3 or not all(np.isfinite(v)):
                    raise ValueError('Invalid coordinates')
    except (ValueError, KeyError, TypeError, OSError, IndexError, AttributeError):
        raise HTTPException(503, 'Invalid motion file')
    return p

def db():
    p = Path(os.getenv('SIGNACULTURE_DB_PATH', str(Path.home()/'.local/state/signaculture/plans.sqlite3')))
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.execute('CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, body TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS reviews (plan_id TEXT, body TEXT)')
    return c

@app.get('/health')
def health(): return {'status': 'ok'}

@app.get('/capabilities')
def capabilities():
    return {**capture_library.capabilities(), 'nlp': mode()}

@app.get('/library')
def captured_library():
    entries=capture_library.library()
    root,glosses,motions=registry()
    existing={entry['motion']['motion_id'] for entry in entries}
    for motion in motions:
        if motion.synthetic or motion.motion_id in existing: continue
        payload=json.loads(asset(root,motion).read_text())
        gloss=next(g for g in glosses if g.gloss_id==motion.gloss_id)
        entries.append({'gloss':gloss.model_dump(),'motion':motion.model_dump(),
            'source':payload.get('attribution',{}).get('credit',gloss.source_id),
            'quality':payload.get('quality',{}),'curated':True,
            'source_video_url':'/examples/bonjour.webm' if motion.motion_id=='LSF_BONJOUR_COMMONS' else None})
    return {'entries': entries}

@app.delete('/library/{motion_id}')
def delete_library_entry(motion_id: str):
    try: capture_library.remove_entry(motion_id)
    except FileNotFoundError: raise HTTPException(404, 'Mouvement inconnu')
    return {'removed': motion_id}

@app.post('/captures', status_code=202)
async def capture(file: UploadFile = File(...), rights_confirmed: bool = Form(...)):
    if not rights_confirmed:
        raise HTTPException(422, 'Confirmez que vous avez le droit de traiter cette vidéo.')
    is_json=(file.filename or '').lower().endswith('.json')
    if not is_json and not capture_library.capabilities()['capture']:
        raise HTTPException(503, 'Runtime de capture absent. Lancez bash scripts/setup-models.sh motion.')
    with capture_library.LOCK:
        if len(capture_library.ACTIVE) >= 4:
            raise HTTPException(429, 'Quatre captures sont déjà en attente. Réessayez après leur traitement.')
    folder=capture_library.storage()/'captures'/str(uuid.uuid4())
    folder.mkdir(parents=True)
    source=folder/'source.mp4'
    size=0
    with source.open('wb') as stream:
        while chunk := await file.read(1024*1024):
            size+=len(chunk)
            if size>50*1024*1024:
                source.unlink(missing_ok=True)
                raise HTTPException(413, 'Vidéo limitée à 50 Mo et 60 secondes.')
            stream.write(chunk)
    if not size:
        source.unlink(missing_ok=True)
        raise HTTPException(422, 'Le fichier est vide.')
    if is_json:
        from .services.landmark_import import convert_animics
        try:
            imported=convert_animics(json.loads(source.read_text()))
        except (ValueError,KeyError,TypeError,UnicodeDecodeError) as exc:
            source.unlink(missing_ok=True)
            raise HTTPException(422,f'Export JSON invalide : {exc}')
        quality=imported.get('quality',{'pose_coverage':1,'left_hand_coverage':0,'right_hand_coverage':0,'face_coverage':0,'sampled_frames':len(imported['frames'])})
        imported['quality']=quality
        imported.setdefault('warnings',[])
        capture_library.write_json(folder/'motion.json',imported)
        job={'job_id':folder.name,'status':'ready','progress':100,'filename':(file.filename or 'export.json')[:200],'duration_ms':imported['duration_ms'],'quality':quality,'warnings':imported['warnings']}
        capture_library.write_json(folder/'status.json',job)
        return job
    return capture_library.create_job(source, (file.filename or 'video')[:200])

@app.get('/captures/{job_id}')
def capture_status(job_id: str):
    try: return capture_library.get_job(job_id)
    except (ValueError, FileNotFoundError): raise HTTPException(404, 'Capture inconnue')

@app.get('/captures/{job_id}/motion')
def capture_motion(job_id: str):
    try:
        folder=capture_library.job_folder(job_id)
        if capture_library.get_job(job_id)['status']!='ready':
            raise HTTPException(409, 'Le mouvement n’est pas encore disponible.')
        return FileResponse(folder/'motion.json',media_type='application/json')
    except (ValueError,FileNotFoundError): raise HTTPException(404, 'Capture inconnue')

@app.get('/captures/{job_id}/source')
def capture_source(job_id: str):
    try:
        path=capture_library.job_folder(job_id)/'source.mp4'
        if not path.is_file(): raise FileNotFoundError()
        with path.open('rb') as stream: magic=stream.read(4)
        media='video/webm' if magic==bytes.fromhex('1a45dfa3') else 'video/mp4'
        return FileResponse(path,media_type=media)
    except (ValueError,FileNotFoundError): raise HTTPException(404, 'Capture inconnue')

@app.get('/examples/bonjour.webm')
def bonjour_source():
    return FileResponse(ROOT/'data/public_examples/bonjour.webm',media_type='video/webm')

class RegisterCapture(BaseModel):
    concept: str = Field(min_length=1,max_length=100)
    aliases: list[str] = Field(default_factory=list,max_length=30)
    source: str = Field(min_length=1,max_length=500)
    license: str = Field(min_length=1,max_length=200)

@app.post('/captures/{job_id}/register')
def register_capture(job_id: str, body: RegisterCapture):
    if not body.concept.strip() or not body.source.strip() or not body.license.strip():
        raise HTTPException(422, 'Le concept, la source et les droits sont obligatoires.')
    try: return capture_library.register(job_id,body.concept,body.aliases,body.source,body.license)
    except FileNotFoundError: raise HTTPException(404, 'Capture inconnue')
    except ValueError as exc: raise HTTPException(409,str(exc))

@app.get('/glossary')
def glossary(include_synthetic: bool = Query(False)):
    _,entries,motions=registry()
    allowed={m.motion_id for m in motions if include_synthetic or not m.synthetic}
    return {'entries':[g for g in entries if g.motion_id in allowed]}

@app.get('/motions/{motion_id}')
def motion(motion_id: str):
    root, _, motions = registry()
    for m in motions:
        if m.motion_id == motion_id:
            asset(root, m)
            return m
    raise HTTPException(404, 'Unknown motion')

@app.get('/assets/{filename}')
def files(filename: str):
    root, _, motions = registry()
    for m in motions:
        if Path(m.path).name == filename:
            return FileResponse(asset(root, m), media_type='application/json')
    raise HTTPException(404, 'Unregistered asset')

@app.post('/analyze-text', response_model=TranslationDraft)
@app.post('/gloss-lookup', response_model=TranslationDraft)
def lookup(body: TextRequest):
    text = body.text
    if not isinstance(text, str) or not text.strip() or len(text) > 10000:
        raise HTTPException(422, 'text must contain 1–10000 characters')
    _, entries, motions = registry()
    allowed={m.motion_id for m in motions if body.include_synthetic or not m.synthetic}
    return lookup_text(text, [g for g in entries if g.motion_id in allowed])

@app.post('/compose-animation', response_model=AnimationPlan)
def compose(body: TextRequest):
    draft = lookup(body)
    return compose_draft(draft, body.audio_features)

def compose_draft(draft, audio=None):
    segments, t = [], 0
    selected = []
    for g in draft['gloss_sequence']:
        m = motion(g.motion_id)
        selected.append(m)
        segments.append(dict(gloss_id=g.gloss_id, motion_id=m.motion_id, start_ms=t, duration_ms=m.duration_ms, asset_url='/assets/'+Path(m.path).name))
        t += m.duration_ms
    from .services.animation_timing import segment_timing
    transition_ms=333 if len(selected)>1 else 0
    timing,t,timing_mode=segment_timing(selected,audio,transition_ms)
    for segment,(start,length) in zip(segments,timing):
        segment.update(start_ms=start,duration_ms=length)
    plan = AnimationPlan(transition_ms=transition_ms,timing_mode=timing_mode,plan_id=str(uuid.uuid4()), source_text=draft['source_text'], target_language='LSF', segments=segments, unknowns=draft['unknowns'], duration_ms=t, linguistic_review='pending', animation_review=('approved' if selected and all(m.animation_review == 'approved' for m in selected) else 'pending'), rights_status=('cleared' if selected and all(m.rights_status == 'cleared' for m in selected) else 'pending'), render_allowed_for_publication=False, audio_features=audio)
    with db() as c: c.execute('INSERT INTO plans VALUES (?,?)', (plan.plan_id, plan.model_dump_json()))
    return plan

def load(plan_id):
    with db() as c: row = c.execute('SELECT body FROM plans WHERE id=?', (plan_id,)).fetchone()
    if not row: raise HTTPException(404, 'Unknown plan')
    return AnimationPlan.model_validate_json(row[0])

@app.post('/review', response_model=ReviewRecord)
def review(request: ReviewRequest):
    body = request.model_dump()
    p = load(body['plan_id'])
    if not isinstance(body.get('reviewer'), str) or not body['reviewer'].strip() or body.get('status') not in ('approved', 'rejected') or not isinstance(body.get('notes'), str):
        raise HTTPException(422, 'reviewer, status and notes required')
    record = ReviewRecord(reviewer=body['reviewer'], status=body['status'], notes=body['notes'], version='1', date=datetime.now(timezone.utc).isoformat())
    p.linguistic_review = record.status
    with db() as c:
        c.execute('INSERT INTO reviews VALUES (?,?)', (p.plan_id, record.model_dump_json()))
        c.execute('UPDATE plans SET body=? WHERE id=?', (p.model_dump_json(), p.plan_id))
    return record

@app.post('/render')
def render(request: PlanRequest):
    body = request.model_dump()
    p = load(body['plan_id'])
    if body.get('purpose') not in ('preview', 'publication'): raise HTTPException(422, 'Invalid purpose')
    assets = [motion(s.motion_id) for s in p.segments]
    if body['purpose'] == 'publication':
        if p.unknowns or not assets or p.linguistic_review != 'approved' or any(m.rights_status != 'cleared' or m.animation_review != 'approved' or m.synthetic for m in assets):
            raise HTTPException(403, 'Publication requires linguistic, animation and rights approval; synthetic fixtures cannot publish')
        raise HTTPException(501, 'Video export is not implemented')
    return {'plan': p, 'assets': assets, 'video_exported': False}

@app.post('/analyze-audio', response_model=AudioFeatures)
async def audio(file: UploadFile = File(...), include_emotion: bool = Query(False)):
    from starlette.concurrency import run_in_threadpool
    from .services.audio_analysis import analyse
    raw = await file.read(10*1024*1024+1)
    if len(raw)>10*1024*1024: raise HTTPException(413,'Audio limité à 10 Mo.')
    if not raw: raise HTTPException(422,'Le fichier est vide.')
    try:
        return await run_in_threadpool(analyse,raw,file.filename or 'voice.wav',include_emotion)
    except ValueError as exc: raise HTTPException(422,str(exc))
    except subprocess.TimeoutExpired: raise HTTPException(504,'Le traitement audio a dépassé le délai autorisé.')
    except FileNotFoundError: raise HTTPException(503,'FFmpeg est nécessaire pour décoder les médias.')


@app.get('/research/status')
def research_status():
    from .services.research_status import status
    return status()

class ResearchRequest(BaseModel):
    text: str = Field(min_length=1,max_length=500)
    run_id: str = Field(min_length=1,max_length=100)

@app.post('/research/predict')
def research_predict(body: ResearchRequest):
    from .services.research_status import predict
    if not body.text.strip(): raise HTTPException(422,'Saisissez une phrase.')
    try: return predict(body.text,body.run_id)
    except FileNotFoundError as exc: raise HTTPException(404,str(exc))
    except BlockingIOError as exc: raise HTTPException(429,str(exc))
    except subprocess.TimeoutExpired: raise HTTPException(504,'Délai d’inférence dépassé.')
    except RuntimeError as exc: raise HTTPException(503,str(exc))


@app.get('/concepts/vocabulary')
def concept_vocabulary():
    _,glosses,motions=registry()
    available={m.motion_id for m in motions if not m.synthetic}
    return {'concepts':[{'id':g.gloss_id,'concept':g.concept_fr} for g in glosses if g.motion_id in available]}

class ConceptSequenceRequest(BaseModel):
    text: str = Field(min_length=1,max_length=4000)
    concept_ids: list[str] = Field(min_length=1,max_length=40)
    unsupported: list[str] = Field(default_factory=list,max_length=50)
    audio_features: AudioFeatures | None = None

@app.post('/compose-concepts',response_model=AnimationPlan)
def compose_concepts(body: ConceptSequenceRequest):
    _,glosses,motions=registry()
    available={m.motion_id for m in motions if not m.synthetic}
    allowed={g.gloss_id:g for g in glosses if g.motion_id in available}
    if any(identifier not in allowed for identifier in body.concept_ids):
        raise HTTPException(422,'Un concept sélectionné est inconnu ou synthétique.')
    return compose_draft({'source_text':body.text,'gloss_sequence':[allowed[i] for i in body.concept_ids],
        'unknowns':body.unsupported},body.audio_features)

class ConceptProposalRequest(BaseModel):
    text: str = Field(min_length=1,max_length=500)

@app.post('/concepts/propose')
def propose_concepts(body: ConceptProposalRequest):
    from .services.concept_planning import propose
    if not body.text.strip(): raise HTTPException(422,'Saisissez une phrase.')
    try: return propose(body.text,concept_vocabulary()['concepts'])
    except FileNotFoundError as exc: raise HTTPException(503,str(exc))
    except BlockingIOError as exc: raise HTTPException(429,str(exc))
    except subprocess.TimeoutExpired: raise HTTPException(504,'La proposition a dépassé le délai autorisé.')
    except RuntimeError as exc: raise HTTPException(503,str(exc))


@app.get('/avatar/model')
def avatar_model():
    path=capture_library.storage()/'avatar-candidates/Noa.glb'
    if not path.is_file(): raise HTTPException(503,'Le modèle 3D local est absent. Exécutez scripts/setup-avatar.py.')
    return FileResponse(path,media_type='model/gltf-binary')
