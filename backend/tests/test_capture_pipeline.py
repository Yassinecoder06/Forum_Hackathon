import json
from pathlib import Path
import uuid
import pytest
from fastapi.testclient import TestClient
from backend.app.main import app,ROOT
from backend.app.services import capture_library
from backend.app.services.landmark_import import convert_animics
from backend.app.services.animation_timing import segment_timing
from backend.app.schemas import AudioFeatures,MotionAsset

@pytest.fixture
def isolated(tmp_path,monkeypatch):
    monkeypatch.setenv('SIGNACULTURE_STORAGE',str(tmp_path/'runtime'))
    monkeypatch.setenv('SIGNACULTURE_DB_PATH',str(tmp_path/'plans.sqlite'))
    monkeypatch.setenv('SIGNACULTURE_SPACY_MODEL','')
    return TestClient(app)

def technical_motion():
    data=json.loads((ROOT/'data/approved_motion_assets/SYNTHETIC_001.json').read_text())
    data['synthetic']=True
    return data

def test_json_capture_lookup_and_persistence(isolated):
    client=isolated
    data=technical_motion()
    assert client.post('/captures',data={'rights_confirmed':'false'},files={'file':('motion.json',json.dumps(data))}).status_code==422
    response=client.post('/captures',data={'rights_confirmed':'true'},files={'file':('motion.json',json.dumps(data))})
    assert response.status_code==202
    job=response.json();assert job['status']=='ready'
    assert client.get(f"/captures/{job['job_id']}/motion").json()['synthetic'] is True
    record=client.post(f"/captures/{job['job_id']}/register",json={'concept':'bonjour','aliases':['salutation'],'source':'automated fixture test','license':'CC0-1.0'})
    assert record.status_code==200
    entry=record.json();assert entry['motion']['synthetic'] is True
    lookup=client.post('/gloss-lookup',json={'text':'bonjour salutation','include_synthetic':True}).json()
    assert len(lookup['gloss_sequence'])==2
    assert all(g['motion_id']==entry['motion']['motion_id'] for g in lookup['gloss_sequence'])
    plan=client.post('/compose-animation',json={'text':'bonjour','include_synthetic':True}).json()
    assert plan['segments'][0]['motion_id']==entry['motion']['motion_id']
    assert client.get('/assets/'+Path(entry['motion']['path']).name).status_code==200
    assert client.post('/render',json={'plan_id':plan['plan_id'],'purpose':'publication'}).status_code==403
    assert len(capture_library.library())==1

def test_bad_motion_and_unknown_job(isolated):
    bad=technical_motion();bad['frames'][0]['joints']['head'][0]=float('nan')
    response=isolated.post('/captures',data={'rights_confirmed':'true'},files={'file':('bad.json',json.dumps(bad))})
    assert response.status_code==422
    assert isolated.get('/captures/not-a-uuid').status_code==404
    assert isolated.get('/captures/'+str(uuid.uuid4())).status_code==404
    assert isolated.post('/captures',data={'rights_confirmed':'true'},files={'file':('empty.json',b'')}).status_code==422

def test_animics_import_preserves_fingers_and_sides():
    pose=[{'x':.5,'y':.5,'z':0} for _ in range(33)]
    pose[11]['x']=.3;pose[12]['x']=.7;pose[15]['x']=.2;pose[16]['x']=.8
    hand=[{'x':.2+i*.001,'y':.5-i*.001,'z':i*.001} for i in range(21)]
    converted=convert_animics([{'PLM':pose,'LLM':hand,'dt':0},{'PLM':pose,'LLM':hand,'dt':0}])
    frame=converted['frames'][0]['joints']
    assert frame['left_wrist'][0]<frame['right_wrist'][0]
    assert len([name for name in frame if name.startswith('left_')])==23
    assert frame['left_index_4'][2]<frame['left_wrist'][2]
    assert converted['frames'][-1]['time_ms']==converted['duration_ms']

def test_audio_timing_keeps_pause_boundaries():
    asset=MotionAsset(motion_id='M',gloss_id='G',path='test.json',rig_version='signaculture-v1',fps=10,duration_ms=1000,license='test',quality_status='test',rights_status='pending',animation_review='pending',synthetic=True)
    audio=AudioFeatures(duration_ms=3000,frames=[dict(start_ms=0,end_ms=1000,pitch_hz=200,energy=.2,is_pause=False),dict(start_ms=1000,end_ms=2000,pitch_hz=None,energy=0,is_pause=True),dict(start_ms=2000,end_ms=3000,pitch_hz=200,energy=.2,is_pause=False)],emotion_hypothesis=None,model_version='test',warnings=[])
    segments,duration,mode=segment_timing([asset,asset],audio)
    assert segments==[(0,2000),(2000,1000)]
    assert duration==3000 and mode=='audio_voiced'


def test_clip_transitions_have_dedicated_time():
    asset=MotionAsset(motion_id='M',gloss_id='G',path='test.json',rig_version='signaculture-v1',fps=10,duration_ms=1000,license='test',quality_status='test',rights_status='pending',animation_review='pending',synthetic=True)
    segments,duration,mode=segment_timing([asset,asset],None,333)
    assert segments==[(0,1000),(1333,1000)]
    assert duration==2333 and mode=='original'
