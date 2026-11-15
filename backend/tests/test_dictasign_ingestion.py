from scripts.ingest_dictasign import parse_index,parse_lsf

def test_french_index_keeps_homonym_concepts_separate_and_merges_aliases():
    html='<a href="cs/cs_12.html">accueillir</a><a href="cs/cs_13.html">accueillir</a><a href="cs/cs_12.html">recevoir</a><a href="../index.html">Home</a>'
    rows=parse_index(html)
    assert len(rows)==2 and rows[0]['labels']==['accueillir','recevoir']
    assert rows[1]['concept_id']=='13'

def test_extracts_only_lsf_media_and_actual_canonical_gloss():
    def box(language,word,gloss,path):
        return f'<div class="cb-left"><div class="title-c">{word}</div><div class="title-l1">{language}&nbsp;</div><div class="nc"><source type="video/mp4" src="{path}"><div class="gloss">{gloss}</div><div class="hns">abc</div></div></div>'
    html=box('BSL','library','library(n)#1','../bsl/490.mp4')+box('LSF','bibliothèque','BIBLIOTHEQUE','../lsf/490.mp4')+box('DGS','Bibliothek','BIBLIOTHEK1','../dgs/490.mp4')
    rows=parse_lsf(html,'https://www.sign-lang.uni-hamburg.de/dicta-sign/portal/concepts/cs/cs_490.html')
    assert len(rows)==1 and rows[0]['canonical_gloss']=='BIBLIOTHEQUE'
    assert rows[0]['video_url'].endswith('/concepts/lsf/490.mp4')

def test_portal_preference_does_not_unregister_previous_motion(monkeypatch,tmp_path):
    from fastapi.testclient import TestClient
    from backend.app import main
    from backend.app.services import capture_library
    monkeypatch.setenv('SIGNACULTURE_STORAGE',str(tmp_path/'runtime'))
    _,glosses,motions=main.registry()
    base=glosses[0];asset=next(m for m in motions if m.motion_id==base.motion_id)
    records=[]
    for name,source in [('OLD','CAPTURE_OLD'),('NEW','DICTASIGN_PORTAL_380')]:
        g=base.model_copy(update={'gloss_id':name,'motion_id':'CAPTURE_'+name,'source_id':source})
        m=asset.model_copy(update={'motion_id':'CAPTURE_'+name,'gloss_id':name})
        records.append({'gloss':g.model_dump(),'motion':m.model_dump(),'quality':{},'source_id':source})
    monkeypatch.setattr(capture_library,'library',lambda:records)
    client=TestClient(main.app)
    result=client.post('/gloss-lookup',json={'text':'bonjour'}).json()
    assert result['gloss_sequence'][0]['gloss_id']=='NEW'
    assert client.get('/motions/CAPTURE_OLD').status_code==200
    assert client.get('/motions/'+asset.motion_id).status_code==200
