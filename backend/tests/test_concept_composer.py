import pytest
from fastapi.testclient import TestClient
from backend.app.main import app

@pytest.fixture
def client(tmp_path,monkeypatch):
    monkeypatch.setenv('SIGNACULTURE_STORAGE',str(tmp_path/'runtime'))
    monkeypatch.setenv('SIGNACULTURE_DB_PATH',str(tmp_path/'plans.sqlite'))
    return TestClient(app)

def test_selected_real_concepts_preserve_order_repetition_and_review_gate(client):
    vocabulary=client.get('/concepts/vocabulary').json()['concepts']
    assert len(vocabulary)==1 and vocabulary[0]['concept']=='bonjour'
    identifier=vocabulary[0]['id']
    response=client.post('/compose-concepts',json={'text':'Salutations répétées','concept_ids':[identifier,identifier],'unsupported':['question non représentée']})
    assert response.status_code==200
    plan=response.json()
    assert [s['gloss_id'] for s in plan['segments']]==[identifier,identifier]
    assert plan['segments'][1]['start_ms']==plan['segments'][0]['duration_ms']
    assert plan['unknowns']==['question non représentée']
    assert plan['linguistic_review']=='pending'
    assert plan['render_allowed_for_publication'] is False
    assert client.post('/render',json={'plan_id':plan['plan_id'],'purpose':'publication'}).status_code==403

@pytest.mark.parametrize('identifiers',[[],['invented'],['LSF_CONCEPT_002'],['LSF_CONCEPT_001']*41])
def test_invalid_or_synthetic_sequences_rejected(client,identifiers):
    assert client.post('/compose-concepts',json={'text':'Bonjour','concept_ids':identifiers}).status_code==422

def test_proposals_reject_unknown_model_ids_and_preserve_negation(client,monkeypatch):
    import subprocess
    from pathlib import Path
    from backend.app.services import concept_planning as service
    import json
    monkeypatch.setattr(service,'runtime',lambda:(Path('python'),Path('planner'),Path('weights')))
    identifier=client.get('/concepts/vocabulary').json()['concepts'][0]['id']
    def response(ids):
        return subprocess.CompletedProcess([],0,stdout=json.dumps({'concept_ids':ids,'unsupported':[],'translation_validated':True}))
    monkeypatch.setattr(service.subprocess,'run',lambda *a,**kw:response(['invented']))
    assert client.post('/concepts/propose',json={'text':'Bonjour'}).status_code==503
    monkeypatch.setattr(service.subprocess,'run',lambda *a,**kw:response([identifier]))
    output=client.post('/concepts/propose',json={'text':'Ne dis pas bonjour ?'}).json()
    assert output['concept_ids']==[identifier]
    assert output['translation_validated'] is False and output['linguistic_review']=='pending'
    assert any('Négation' in s for s in output['unsupported'])
    assert any('Question' in s for s in output['unsupported'])

def test_research_screen_keeps_concept_metrics_separate(tmp_path,monkeypatch):
    import json
    from backend.app.services import research_status
    for name,payload in [('translation',{'run_id':'translation','model_base':'t5-small','metrics':{'test':{}}}),('concepts',{'kind':'SOFTWARE_CONCEPT_PLANNING_NOT_LSF_TRANSLATION','metrics':{'model':{'examples':24}}})]:
        folder=tmp_path/'research/runs'/name
        folder.mkdir(parents=True)
        (folder/'metrics.json').write_text(json.dumps(payload))
    monkeypatch.setattr(research_status,'ROOT',tmp_path)
    assert [r['run_id'] for r in research_status.reports()]==['translation']
