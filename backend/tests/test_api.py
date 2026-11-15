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

def test_malformed_requests():
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
