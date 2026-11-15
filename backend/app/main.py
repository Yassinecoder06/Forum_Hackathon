"""Local experimental lookup API; no translation or emotion model claims."""
import io
import json
import os
from pathlib import Path
import re
import sqlite3
import uuid
import wave
from datetime import datetime, timezone

import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal
from .schemas import GlossEntry, MotionAsset, TranslationDraft, AnimationPlan, AudioFeatures, ReviewRecord

ROOT = Path(__file__).resolve().parents[2]
app = FastAPI(title='SignaCulture experimental API')
app.add_middleware(CORSMiddleware, allow_origins=os.getenv('SIGNACULTURE_CORS_ORIGINS', 'http://localhost:5173').split(','), allow_methods=['*'], allow_headers=['*'])

class TextRequest(BaseModel):
    text: str = Field(min_length=1, max_length=10000)
    audio_features: AudioFeatures | None = None

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
    response.headers['X-NLP-Backend'] = 'unicode-regex-fallback-no-lemmatizer'
    return response

def registry():
    root = Path(os.getenv('SIGNACULTURE_DATA_ROOT', str(ROOT)))
    try:
        gloss = [GlossEntry.model_validate(x) for x in json.loads((root/'data/glossary.json').read_text())]
        motions = [MotionAsset.model_validate(x) for x in json.loads((root/'data/motions_manifest.json').read_text())]
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
    if not p.is_relative_to((root/'data/approved_motion_assets').resolve()) or not p.is_file():
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
    except (ValueError, KeyError, AssertionError, TypeError):
        raise HTTPException(503, 'Invalid motion file')
    return p

def db():
    p = Path(os.getenv('SIGNACULTURE_DB_PATH', str(Path.home()/'.local/state/signaculture/plans.sqlite3')))
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p)
    c.execute('CREATE TABLE IF NOT EXISTS plans (id TEXT PRIMARY KEY, body TEXT)')
    c.execute('CREATE TABLE IF NOT EXISTS reviews (plan_id TEXT, body TEXT)')
    return c

TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
def words(s): return TOKEN.findall(s.casefold().replace('’', "'"))

@app.get('/health')
def health(): return {'status': 'ok'}

@app.get('/glossary')
def glossary(): return {'entries': registry()[1]}

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
    _, entries, _ = registry()
    tokens = TOKEN.findall(text)
    normalized = words(text)
    candidates = []
    for g in entries:
        for phrase in [g.concept_fr, *g.aliases]:
            w = words(phrase)
            if w: candidates.append((w, g))
    candidates.sort(key=lambda x: -len(x[0]))
    sequence, unknowns, i = [], [], 0
    while i < len(tokens):
        matches = [(w, g) for w, g in candidates if normalized[i:i+len(w)] == w]
        if matches:
            size = len(matches[0][0])
            best = {g.gloss_id: g for w, g in matches if len(w) == size}
            if len(best) == 1:
                sequence.append(next(iter(best.values())))
                i += size
                continue
        unknowns.append(tokens[i]); i += 1
    return dict(source_text=text, tokens=tokens, lemmas=normalized, gloss_sequence=sequence, unknowns=unknowns, linguistic_review='pending', translation_validated=False)

@app.post('/compose-animation', response_model=AnimationPlan)
def compose(body: TextRequest):
    draft = lookup(body)
    audio = body.audio_features
    segments, t = [], 0
    selected = []
    for g in draft['gloss_sequence']:
        m = motion(g.motion_id)
        selected.append(m)
        segments.append(dict(gloss_id=g.gloss_id, motion_id=m.motion_id, start_ms=t, duration_ms=m.duration_ms, asset_url='/assets/'+Path(m.path).name))
        t += m.duration_ms
    plan = AnimationPlan(plan_id=str(uuid.uuid4()), source_text=draft['source_text'], target_language='LSF', segments=segments, unknowns=draft['unknowns'], duration_ms=t, linguistic_review='pending', animation_review=('approved' if selected and all(m.animation_review == 'approved' for m in selected) else 'pending'), rights_status=('cleared' if selected and all(m.rights_status == 'cleared' for m in selected) else 'pending'), render_allowed_for_publication=False, audio_features=audio)
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
async def audio(file: UploadFile = File(...)):
    raw = await file.read(10*1024*1024+1)
    if len(raw) > 10*1024*1024: raise HTTPException(413, 'Maximum WAV size 10 MiB')
    try:
        with wave.open(io.BytesIO(raw)) as w:
            rate, channels, width, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
            if rate < 8000 or rate > 96000 or channels not in (1,2) or width not in (1,2,4) or not 0 < n/rate <= 60: raise ValueError('Unsupported WAV or duration (maximum 60 seconds)')
            data = w.readframes(n)
            if len(data) != n*channels*width: raise ValueError('Truncated WAV')
            x = np.frombuffer(data, dtype={1:'u1',2:'<i2',4:'<i4'}[width]).astype(float)
            x = (x-128)/128 if width == 1 else x/(2**(width*8-1))
            x = x.reshape(-1, channels).mean(axis=1)
    except (wave.Error, EOFError, ValueError) as e: raise HTTPException(422, str(e))
    frames = []
    step = int(rate*.04)
    for start in range(0,len(x),step):
        chunk = x[start:start+step]; energy = float(np.sqrt(np.mean(chunk**2))); pitch = None
        if energy >= .01 and len(chunk) >= rate/50:
            stride = max(1, rate//8000)
            pitch_rate = rate/stride
            y = chunk[::stride]-chunk[::stride].mean(); corr = np.correlate(y,y,mode='full')[len(y)-1:]
            lo, hi = int(pitch_rate/500), min(int(pitch_rate/60),len(corr)-1)
            lag = lo+int(np.argmax(corr[lo:hi+1]))
            if corr[0] > 0 and corr[lag]/corr[0] > .35: pitch = float(pitch_rate/lag)
        frames.append(dict(start_ms=round(start/rate*1000), end_ms=max(1,round((start+len(chunk))/rate*1000)), pitch_hz=pitch, energy=energy, is_pause=energy<.01))
    return dict(duration_ms=max(1,round(n/rate*1000)), frames=frames, emotion_hypothesis=None, model_version='numpy-autocorrelation-rms-v1', warnings=['Emotion unavailable: no real emotion adapter configured.', 'F0 is an autocorrelation estimate; noise and harmonics can cause errors.'])
