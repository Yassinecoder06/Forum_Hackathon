import io
import wave
import numpy as np
from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_partial_unknowns():
    text = 'Bonjour inconnu inconnu'
    r = client.post('/gloss-lookup', json={'text':text})
    assert r.status_code == 200
    assert r.json()['unknowns'] == ['inconnu','inconnu']
    assert r.json()['translation_validated'] is False

def test_review_cannot_bypass_animation(monkeypatch,tmp_path):
    monkeypatch.setenv('SIGNACULTURE_DB_PATH', str(tmp_path/'state.sqlite'))
    p = client.post('/compose-animation',json={'text':'bonjour'}).json()
    assert client.post('/review',json={'plan_id':p['plan_id'],'reviewer':'expert','status':'approved','notes':'test'}).status_code == 200
    assert client.post('/render',json={'plan_id':p['plan_id'],'purpose':'publication'}).status_code == 403
    assert client.post('/render',json={'plan_id':p['plan_id'],'purpose':'preview'}).json()['video_exported'] is False

def test_real_tone_and_invalid_wav():
    b = io.BytesIO()
    with wave.open(b,'wb') as w:
        w.setparams((1,2,16000,0,'NONE','NONE'))
        w.writeframes((np.sin(2*np.pi*200*np.arange(16000)/16000)*12000).astype('<i2').tobytes())
    r = client.post('/analyze-audio',files={'file':('tone.wav',b.getvalue(),'audio/wav')})
    assert r.status_code == 200
    assert abs(r.json()['frames'][0]['pitch_hz']-200)<5
    assert r.json()['emotion_hypothesis'] is None
    assert client.post('/analyze-audio',files={'file':('bad.wav',b'bad')}).status_code == 422

def test_missing_asset_and_empty_text():
    assert client.get('/assets/unregistered.json').status_code == 404
    assert client.post('/analyze-text',json={'text':''}).status_code == 422

def test_malformed_requests(monkeypatch):
    monkeypatch.setenv('SIGNACULTURE_SPACY_MODEL', '')
    assert client.post('/compose-animation',json={'text':'bonjour','audio_features':{'frames':[]}}).status_code == 422
    assert client.post('/render',json={'plan_id':[]}).status_code == 422
    assert client.post('/review',json={'plan_id':{},'reviewer':'x','status':'approved','notes':''}).status_code == 422
    assert client.post('/analyze-text',json={'text':'bonjour'}).headers['x-nlp-backend'] == 'unicode-regex-fallback-no-lemmatizer'

def test_high_rate_silence():
    b = io.BytesIO()
    with wave.open(b,'wb') as w:
        w.setparams((2,2,96000,0,'NONE','NONE'))
        w.writeframes(bytes(96000*4))
    result = client.post('/analyze-audio',files={'file':('silence.wav',b.getvalue())}).json()
    assert result['duration_ms'] == 1000
    assert result['frames'][-1]['end_ms'] == 1000
    assert all(f['is_pause'] and f['pitch_hz'] is None for f in result['frames'])

import json
import shutil
from pathlib import Path
import pytest
from backend.app import main
from backend.app.services import text_lookup

@pytest.fixture(autouse=True)
def isolated_runtime(tmp_path, monkeypatch):
    monkeypatch.setenv('SIGNACULTURE_STORAGE', str(tmp_path/'runtime'))
    monkeypatch.setenv('SIGNACULTURE_DB_PATH', str(tmp_path/'plans.sqlite'))

@pytest.fixture
def registry_copy(tmp_path, monkeypatch):
    shutil.copytree(main.ROOT/'data', tmp_path/'data')
    monkeypatch.setenv('SIGNACULTURE_DATA_ROOT', str(tmp_path))
    monkeypatch.setenv('SIGNACULTURE_SPACY_MODEL', '')
    return tmp_path

def test_longest_and_ambiguous_alias(registry_copy):
    path = registry_copy/'data/glossary.json'
    entries = json.loads(path.read_text())
    entries[0]['concept_fr'] = 'bon'
    entries[0]['aliases'] = ['ambigu']
    entries[1]['concept_fr'] = 'bon jour'
    entries[1]['aliases'] = ['ambigu']
    path.write_text(json.dumps(entries))
    r = client.post('/gloss-lookup',json={'text':'Bon jour ambigu Étrange Étrange','include_synthetic':True}).json()
    assert [g['gloss_id'] for g in r['gloss_sequence']] == [entries[1]['gloss_id']]
    assert r['unknowns'] == ['ambigu','Étrange','Étrange']

@pytest.mark.parametrize('damage',['missing_joint','negative_time','missing_file','escape','malformed'])
def test_invalid_assets(registry_copy, damage):
    manifest = registry_copy/'data/motions_manifest.json'
    motions = json.loads(manifest.read_text())
    m = motions[0]
    path = registry_copy/m['path']
    d = json.loads(path.read_text())
    if damage == 'missing_joint': del d['frames'][0]['joints'][d['joints'][0]]
    elif damage == 'negative_time': d['frames'][0]['time_ms'] = -1
    elif damage == 'missing_file': path.unlink()
    elif damage == 'escape':
        m['path'] = '../outside.json'
        manifest.write_text(json.dumps(motions))
    elif damage == 'malformed': path.write_text('{bad')
    if damage in ('missing_joint','negative_time'): path.write_text(json.dumps(d))
    assert client.get('/motions/'+m['motion_id']).status_code == 503

def test_optional_lemma_alignment(monkeypatch):
    # Adapter stub exercises token offset alignment without installing model weights.
    class Token:
        def __init__(self,text,idx): self.text,self.idx,self.lemma_=text,idx,{'chats':'chat'}.get(text,text)
    def nlp(text): return [Token(m.group(),m.start()) for m in text_lookup.TOKEN.finditer(text)]
    monkeypatch.setattr(text_lookup,'model',lambda:nlp)
    entry = main.registry()[1][0].model_copy(update={'concept_fr':'chat','aliases':[]})
    r = text_lookup.lookup_text('chats, Étrange chats!', [entry])
    assert len(r['gloss_sequence']) == 2
    assert r['tokens'] == ['chats','Étrange','chats']
    assert r['unknowns'] == ['Étrange']
    assert text_lookup.mode() == 'spacy-fr-lemmatizer'

def test_stereo_tone_then_silence():
    rate = 16000
    tone = (np.sin(2*np.pi*200*np.arange(rate//2)/rate)*10000).astype('<i2')
    mono = np.concatenate([tone,np.zeros(rate//2,dtype='<i2')])
    stereo = np.column_stack([mono,mono]).astype('<i2')
    b=io.BytesIO()
    with wave.open(b,'wb') as w:
        w.setparams((2,2,rate,0,'NONE','NONE')); w.writeframes(stereo.tobytes())
    r=client.post('/analyze-audio',files={'file':('stereo.wav',b.getvalue())}).json()
    assert abs(r['frames'][0]['pitch_hz']-200)<5
    assert r['frames'][0]['energy']>.1
    assert all(f['is_pause'] and f['energy']==0 for f in r['frames'] if f['start_ms']>=520)


def test_real_french_model(monkeypatch):
    monkeypatch.setenv('SIGNACULTURE_SPACY_MODEL', 'fr_core_news_sm')
    pytest.importorskip('spacy')
    import importlib.util
    if importlib.util.find_spec('fr_core_news_sm') is None:
        pytest.skip('Optional French spaCy model is not installed')
    assert text_lookup.model() is not None
    tokens, lemmas = text_lookup.tokenize('musées')
    assert tokens == ['musées']
    assert lemmas == ['musée']
    entry = main.registry()[1][0].model_copy(update={'concept_fr':'musée', 'aliases':[]})
    result = text_lookup.lookup_text('musées Inconnu', [entry])
    assert [g.gloss_id for g in result['gloss_sequence']] == [entry.gloss_id]
    assert result['unknowns'] == ['Inconnu']
    response = client.post('/analyze-text', json={'text':'musées'})
    assert response.headers['x-nlp-backend'] == 'spacy-fr-lemmatizer'


def test_synthetic_glosses_are_never_default_signs():
    response=client.post('/compose-animation',json={'text':'musée culture art'}).json()
    assert response['segments']==[]
    assert response['unknowns']==['musée','culture','art']
    assert len(client.get('/glossary').json()['entries'])==1
    assert len(client.get('/glossary?include_synthetic=true').json()['entries'])>1
