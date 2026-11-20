"""Local experimental lookup API; no translation or emotion model claims."""
import gzip
import json
import os
from pathlib import Path
import sqlite3
import uuid
import zlib
from datetime import datetime, timezone

import numpy as np
import subprocess
import tempfile
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Literal
from .schemas import GlossEntry, MotionAsset, TranslationDraft, AnimationPlan, AudioFeatures, ReviewRecord

from .services.text_lookup import lookup_text, mode
from .services import capture_library
from .services.lsf_pose_convert import engine_poses_to_motion
from .engines import lsf as lsf_engines

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
        # Legacy records require source-verified migration; absence is not evidence of fsl.
        imported = [row for row in capture_library.library() if row.get('sign_language') == 'fsl']
        gloss.extend(GlossEntry.model_validate(x['gloss']) for x in imported)
        motions.extend(MotionAsset.model_validate(x['motion']) for x in imported)
        if len({x.motion_id for x in motions}) != len(motions) or len({x.gloss_id for x in gloss}) != len(gloss):
            raise ValueError('Duplicate IDs')
        motion_glosses={m.motion_id:m.gloss_id for m in motions}
        for g in gloss:
            if motion_glosses.get(g.motion_id)!=g.gloss_id:
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
    name, path = local_avatar()
    engines={k:v['available'] for k,v in lsf_engines.status().items()}
    return {**capture_library.capabilities(), 'nlp': mode(), 'avatar': name if path else None, 'lsf_engines': engines}

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
    filename = (file.filename or '').lower()
    compressed_json = filename.endswith('.json.gz')
    is_json = compressed_json or filename.endswith('.json')
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
            if compressed_json:
                limit = 64 * 1024 * 1024
                with gzip.open(source, 'rb') as stream:
                    content = stream.read(limit + 1)
                if len(content) > limit:
                    source.unlink(missing_ok=True)
                    raise HTTPException(413, 'Export JSON limité à 64 Mio après décompression.')
            else:
                content = source.read_bytes()
            imported=convert_animics(json.loads(content))
        except (ValueError,KeyError,TypeError,OSError,EOFError,zlib.error) as exc:
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
    sign_language: Literal['fsl']
    concept: str = Field(min_length=1,max_length=100)
    aliases: list[str] = Field(default_factory=list,max_length=30)
    source: str = Field(min_length=1,max_length=500)
    license: str = Field(min_length=1,max_length=200)

@app.post('/captures/{job_id}/register')
def register_capture(job_id: str, body: RegisterCapture):
    if not body.concept.strip() or not body.source.strip() or not body.license.strip():
        raise HTTPException(422, 'Le concept, la source et les droits sont obligatoires.')
    try: return capture_library.register(job_id,body.concept,body.aliases,body.source,body.license,body.sign_language)
    except FileNotFoundError: raise HTTPException(404, 'Capture inconnue')
    except ValueError as exc: raise HTTPException(409,str(exc))

def preferred_glosses(entries):
    # Select a lexical source without unregistering older motion assets/plans.
    groups={};quality={r['gloss']['gloss_id']:r.get('quality',{}) for r in capture_library.library()}
    def priority(g):
        return 3 if g.source_id.startswith('DICTASIGN_PORTAL_') else 2 if g.motion_id.startswith('CAPTURE_') else 1
    for g in entries:groups.setdefault(g.concept_fr.casefold(),[]).append(g)
    selected=[]
    for rows in groups.values():
        rank=max(priority(g) for g in rows);rows=[g for g in rows if priority(g)==rank]
        if rank!=3:selected.extend(rows);continue
        variants={}
        for g in rows:
            def score(v):
                q=quality.get(v.gloss_id,{})
                return (q.get('pose_coverage',0)+q.get('left_hand_coverage',0)+q.get('right_hand_coverage',0),v.gloss_id)
            old=variants.get(g.gloss)
            if old is None or score(g)>score(old):variants[g.gloss]=g
        selected.extend(variants.values())
    return selected

@app.get('/glossary')
def glossary(include_synthetic: bool = Query(False)):
    _,entries,motions=registry()
    allowed={m.motion_id for m in motions if include_synthetic or not m.synthetic}
    selected=[g for g in entries if g.motion_id in allowed]
    return {'entries':selected if include_synthetic else preferred_glosses(selected)}

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
    return lookup_text(text, preferred_glosses([g for g in entries if g.motion_id in allowed]))

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
    # Play each sign's active core so bridges join signing hands, not two rest poses.
    if selected:
        from .services.sign_core import clip_windows
        root,_,_=registry()
        windows=clip_windows([asset(root,m) for m in selected])
        for segment,(start,end) in zip(segments,windows):
            segment.update(clip_start_ms=start,clip_end_ms=end)
        selected=[m.model_copy(update={'duration_ms':end-start}) for m,(start,end) in zip(selected,windows)]
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
async def audio(file: UploadFile = File(...), include_emotion: bool = Query(False), include_transcription: bool = Query(False)):
    from starlette.concurrency import run_in_threadpool
    from .services.audio_analysis import analyse
    raw = await file.read(10*1024*1024+1)
    if len(raw)>10*1024*1024: raise HTTPException(413,'Audio limité à 10 Mo.')
    if not raw: raise HTTPException(422,'Le fichier est vide.')
    try:
        return await run_in_threadpool(analyse,raw,file.filename or 'voice.wav',include_emotion,include_transcription)
    except ValueError as exc: raise HTTPException(422,str(exc))
    except subprocess.TimeoutExpired: raise HTTPException(504,'Le traitement audio a dépassé le délai autorisé.')
    except FileNotFoundError: raise HTTPException(503,'FFmpeg est nécessaire pour décoder les médias.')


@app.post('/analyze-audio/emotion-segments')
async def audio_emotion_segments(file: UploadFile = File(...), cues: str = Form(...)):
    """Film mode: emotion2vec scores of the voice for each subtitle line (one model load for all lines).
    cues: JSON list of {start_ms, end_ms}. Up to 30 min and 300 lines; the audio stays on this machine."""
    from starlette.concurrency import run_in_threadpool
    from .services.capture_library import emotion_model_dir, emotion_model_id, emotion_model_ready
    try:
        windows = [(int(c['start_ms']), int(c['end_ms'])) for c in json.loads(cues)][:300]
    except (ValueError, TypeError, KeyError):
        raise HTTPException(422, 'Liste de répliques invalide.')
    if not windows:
        raise HTTPException(422, 'Aucune réplique.')
    if not emotion_model_ready() or not capture_library.runtime_python('audio').is_file():
        raise HTTPException(503, 'Le modèle émotionnel local est absent.')
    raw = await file.read(500 * 1024 * 1024 + 1)
    if len(raw) > 500 * 1024 * 1024:
        raise HTTPException(413, 'Fichier limité à 500 Mo.')

    def work():
        with tempfile.TemporaryDirectory(prefix='signaculture-emotion-') as temp:
            temp = Path(temp)
            source = temp / ('film' + Path(file.filename or 'film.mp4').suffix.lower())
            source.write_bytes(raw)
            clips, index = [], []
            for k, (a, b) in enumerate(windows):
                if b - a < 300 or a > 30 * 60 * 1000:
                    continue
                wav = temp / f'cue{k:04d}.wav'
                done = subprocess.run(['ffmpeg', '-nostdin', '-hide_banner', '-loglevel', 'error', '-y', '-ss', f'{a / 1000:.3f}', '-to',
                                       f'{min(b, a + 30000) / 1000:.3f}', '-i', str(source), '-vn', '-ac', '1', '-ar', '16000',
                                       '-c:a', 'pcm_s16le', str(wav)], capture_output=True, timeout=60)
                if not done.returncode and wav.is_file() and wav.stat().st_size > 2000:
                    clips.append(str(wav)); index.append(k)
            if not clips:
                raise ValueError('Aucune piste audio exploitable dans les répliques.')
            listing, output = temp / 'clips.json', temp / 'emotion.json'
            listing.write_text(json.dumps(clips))
            result = subprocess.run([str(capture_library.runtime_python('audio')), str(capture_library.ROOT / 'backend/app/workers/emotion.py'),
                                     str(listing), str(emotion_model_dir()), str(output), '--batch'],
                                    capture_output=True, text=True, timeout=120 + 5 * len(clips), cwd=capture_library.ROOT)
            if result.returncode:
                (capture_library.storage() / 'emotion-error.log').write_text(result.stderr[-10000:])
                raise RuntimeError('Le modèle émotionnel a échoué.')
            items = json.loads(output.read_text())['items']
            scores = [None] * len(windows)
            for k, sc in zip(index, items):
                scores[k] = sc
            return {'items': scores, 'model': emotion_model_id(), 'validated': False,
                    'note': "Hypothèses du modèle sur la voix, pas une mesure de l'état intérieur."}
    try:
        return await run_in_threadpool(work)
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except FileNotFoundError:
        raise HTTPException(503, 'FFmpeg est nécessaire pour décoder les médias.')
    except subprocess.TimeoutExpired:
        raise HTTPException(504, 'Le traitement audio a dépassé le délai autorisé.')
    except RuntimeError as exc:
        raise HTTPException(503, str(exc))


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
    return {'concepts':[{'id':g.gloss_id,'concept':g.concept_fr} for g in preferred_glosses([g for g in glosses if g.motion_id in available])]}

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

# MetaHuman is opt-in (SIGNACULTURE_AVATAR=MetaHuman): local conversion, web-use licence unconfirmed.
AVATARS = ('ReadyEva', 'Noa', 'MetaHuman')

def local_avatar():
    configured = os.getenv('SIGNACULTURE_AVATAR')
    names = (configured,) if configured else ('ReadyEva', 'Noa')
    folder = Path(os.environ['SIGNACULTURE_AVATAR_DIR']) if os.getenv('SIGNACULTURE_AVATAR_DIR') else capture_library.storage() / 'avatar-candidates'
    for name in names:
        if name in AVATARS:
            path = folder / f'{name}.glb'
            if path.is_file():
                return name, path
    return configured, None


@app.get('/avatar/model')
def avatar_model():
    name, path = local_avatar()
    if path is None:
        if name and name not in AVATARS:
            raise HTTPException(503, f'Avatar inconnu : {name}')
        detail = f'Le modèle 3D local {name} est absent.' if name else 'Aucun personnage 3D local trouvé.'
        raise HTTPException(503, f'{detail} Exécutez scripts/setup-avatar.py.')
    # Revalidate on every load (ETag) so switching avatars is never masked by cache.
    return FileResponse(path,media_type='model/gltf-binary',headers={'Cache-Control':'no-cache'})


class LsfGenerateRequest(BaseModel):
    text: str = Field(min_length=1, max_length=400)
    engine: Literal['c0', 'b2'] = 'b2'
    seed: int = Field(default=0, ge=0, le=2**31 - 1)
    # B2 only: 'lsf' signs the glosses of the rule-based LSF word order (services/lsf_order.py), 'french' the text order.
    order: Literal['french', 'lsf'] = 'french'
    # B2 only: an edited gloss sequence (one list per clause) replaces the automatic order.
    glosses: list[list[str]] | None = Field(default=None, max_length=12)
    # Grammatical non-manual markers (questions: brows and head tilt; negation: head shake), from the French text.
    nonmanual: bool = True
    # Voice emotion (emotion2vec scores, e.g. AudioFeatures.emotion_hypothesis or one label at 1.0): changes the speed
    # and size of the manual signing only, never the face (services/lsf_emotion.py). Heuristic, not validated.
    emotion: dict[str, float] | None = Field(default=None, max_length=12)
    emotion_strength: float = Field(default=1.0, ge=0.0, le=1.0)


class LsfOrderRequest(BaseModel):
    text: str = Field(min_length=1, max_length=400)


@app.post('/lsf/order')
def lsf_order_preview(body: LsfOrderRequest):
    """French text -> LSF-ordered gloss sequence by simplified rules (time, place, subject, object, verb, negation,
    question word) with the clause markers. Editable draft for B2; not checked by an LSF reader."""
    from .services.lsf_order import order
    text = ' '.join(body.text.split())
    if not text:
        raise HTTPException(422, 'Saisissez une phrase en français.')
    return order(text)


def _nonmanual_spans(analysis: dict, meta: dict, n_frames: int, fps: float = 25.0) -> list[dict]:
    """Clause markers -> time spans. B2: from the clause's first to last sign; otherwise the clip is split
    between clauses in proportion to their text length (C0 has no sign boundaries)."""
    clauses = analysis.get('clauses') or []
    if not any(c['markers'] for c in clauses):
        return []
    total_ms = (n_frames - 1) / fps * 1000.0
    spans, signs = [], meta.get('sign_spans') or []
    if signs:
        for k, c in enumerate(clauses):
            mine = [sp for sp in signs if sp['clause'] == k]
            if mine and c['markers']:
                a, b = mine[0]['start_frame'] / fps * 1000.0, mine[-1]['end_frame'] / fps * 1000.0
                spans += [{'start_ms': round(a), 'end_ms': round(b), 'type': m} for m in c['markers']]
        return spans
    lengths = [max(1, len(c['text'])) for c in clauses]
    head, tail = min(300.0, 0.1 * total_ms), min(300.0, 0.1 * total_ms)
    usable, t = max(1.0, total_ms - head - tail), head
    for c, n in zip(clauses, lengths):
        d = usable * n / sum(lengths)
        spans += [{'start_ms': round(t), 'end_ms': round(t + d), 'type': m} for m in c['markers']]
        t += d
    return spans


@app.post('/lsf/generate')
def lsf_generate(body: LsfGenerateRequest):
    """French text -> generated MotionFile (C0 generator or B2 stitching). Automatic proxy output:
    never a validated LSF translation. Models are local, private files; nothing is downloaded here."""
    from .services.lsf_order import order
    text = ' '.join(body.text.split())
    if not text:
        raise HTTPException(422, 'Saisissez une phrase en français.')
    analysis = order(text)
    groups = None
    if body.engine == 'b2':
        if body.glosses:
            groups = [[g.strip()[:40] for g in clause if g.strip()][:40] for clause in body.glosses]
            # keep the text's markers when the edited clauses line up with them, else drop them
            if len(groups) != len(analysis['clauses']):
                analysis = {**analysis, 'clauses': [{'text': ' '.join(g), 'glosses': g, 'markers': [], 'rules': []} for g in groups]}
        elif body.order == 'lsf':
            groups = [c['glosses'] for c in analysis['clauses']]
    try:
        poses, meta = lsf_engines.generate(text, body.engine, seed=body.seed, groups=groups)
        from .services.lsf_emotion import apply_emotion, emotion_params
        feeling = emotion_params(body.emotion, body.emotion_strength)
        if feeling and (abs(feeling['speed'] - 1) > 1e-3 or abs(feeling['amplitude'] - 1) > 1e-3):
            poses = apply_emotion(poses, feeling)
            for sp in meta.get('sign_spans') or []:            # sign boundaries follow the retiming
                sp['start_frame'] = int(round(sp['start_frame'] / feeling['speed']))
                sp['end_frame'] = int(round(sp['end_frame'] / feeling['speed']))
        if feeling:
            meta['emotion'] = {**feeling, 'applies_to': 'speed and size of manual signing only', 'validated': False}
        spans = _nonmanual_spans(analysis, meta, len(poses)) if body.nonmanual else []
        motion = engine_poses_to_motion(poses, engine=body.engine, text=text, face_calibration=meta.pop('face_neutral', None),
                                        nonmanual=spans)
    except lsf_engines.EngineUnavailable as exc:
        raise HTTPException(503, str(exc))
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    meta.pop('sign_spans', None)
    meta['order'] = 'edited' if (body.engine == 'b2' and body.glosses) else (body.order if body.engine == 'b2' else 'french')
    meta['gloss_analysis'] = analysis
    motion['generation'] = {**meta, 'translation_validated': False, 'linguistic_review': 'pending'}
    return motion


@app.get('/captures/{job_id}/landmarks')
def capture_landmarks(job_id: str):
    try:
        path=capture_library.job_folder(job_id)/'animics-landmarks.json.gz'
        row=next((r for r in capture_library.library() if r.get('capture_id')==job_id),None)
        if row and row.get('raw_animics_path'):
            candidate=Path(row['raw_animics_path']).resolve()
            if candidate.is_relative_to(capture_library.storage()) and candidate.is_file():path=candidate
        if not path.is_file():raise FileNotFoundError()
        return FileResponse(path,media_type='application/gzip',filename='animics-landmarks.json.gz')
    except (ValueError,FileNotFoundError):raise HTTPException(404,'Landmarks bruts indisponibles pour cette capture.')

@app.get('/datasets/dictasign/lexicon')
def dictasign_lexicon():
    entries=[]
    for row in capture_library.library():
        if not row.get('source_id','').startswith('DICTASIGN_PORTAL_'):continue
        entries.append({'concept_id':row['source_id'],'concept_fr':row['gloss']['concept_fr'],
            'canonical_gloss':row['gloss']['gloss'],'gloss_id':row['gloss']['gloss_id'],
            'aliases':row['gloss']['aliases'],'language':'fsl','quality':row['quality'],
            'motion_url':'/assets/'+Path(row['motion']['path']).name,
            'raw_animics_url':'/captures/'+row['capture_id']+'/landmarks',
            'source_url':row['source'],'source_sha256':row.get('provenance',{}).get('source_sha256'),
            'rights_status':row['motion']['rights_status'],'linguistic_review':'pending'})
    return JSONResponse({'kind':'France-LSF lexical video/pose dataset, not sentence translation pairs',
        'source_url':'https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/concepts_fre.html',
        'entries':entries},headers={'Content-Disposition':'attachment; filename="dictasign-lexicon.json"'})

class LexicalPipelineRequest(BaseModel):
    text: str = Field(min_length=1,max_length=500)

@app.post('/pipeline/lexical')
def lexical_sequence(body: LexicalPipelineRequest):
    from .services.lexical_pipeline import run
    if not body.text.strip():raise HTTPException(422,'Saisissez des mots ou des glosses.')
    try:return run(body.text)
    except FileNotFoundError as exc:raise HTTPException(503,str(exc))
    except BlockingIOError as exc:raise HTTPException(429,str(exc))
    except ValueError as exc:raise HTTPException(422,str(exc))
    except subprocess.TimeoutExpired:raise HTTPException(504,'La construction des poses a dépassé le délai autorisé.')
    except RuntimeError as exc:raise HTTPException(503,str(exc))

@app.get('/pipeline/lexical/{run_id}/{kind}')
def lexical_artifact(run_id:str,kind:str):
    from .services.lexical_pipeline import artifact
    try:path,mime=artifact(run_id,kind)
    except FileNotFoundError:raise HTTPException(404,'Fichier de poses inconnu.')
    return FileResponse(path,media_type=mime,filename=path.name)
