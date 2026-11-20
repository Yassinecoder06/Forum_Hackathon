import hashlib,json,sqlite3
from pathlib import Path
from backend.app.services import research_status

def setup(tmp_path,monkeypatch):
 monkeypatch.setattr(research_status,'ROOT',tmp_path);path=tmp_path/'catalog.sqlite';monkeypatch.setenv('SIGNACULTURE_RESEARCH_CATALOG',str(path));report=tmp_path/'.runtime/run/RESULT.json';report.parent.mkdir(parents=True);report.write_text(json.dumps({'status':'complete','validation':{'examples':23,'exact':12/23,'operation_wer':.4667,'private_french':'secret sentence'}}))
 db=sqlite3.connect(path);db.executescript('CREATE TABLE model_runs(run_id TEXT,model_base TEXT,hyperparameters_json TEXT,metrics_json TEXT);CREATE TABLE research_experiment_index(run_id TEXT,sign_language TEXT,representation TEXT,result_sha256 TEXT,report_path TEXT);')
 db.execute('INSERT INTO model_runs VALUES(?,?,?,?)',('example','google-t5/t5-base',json.dumps({'train':145,'language':'LSFB','checkpoint':'/private/weights'}),json.dumps({'examples':23,'exact':12/23,'operation_wer':.4667,'private_french':'secret sentence'})))
 db.execute('INSERT INTO research_experiment_index VALUES(?,?,?,?,?)',('example','sfb','lexical_projection',hashlib.sha256(report.read_bytes()).hexdigest(),str(report)));db.commit();db.close();return report,path

def test_recent_summary_excludes_private_paths_and_keeps_language(tmp_path,monkeypatch):
 report,path=setup(tmp_path,monkeypatch);before=hashlib.sha256(path.read_bytes()).hexdigest();result=research_status.recent_experiments();r=result['entries'][0]
 assert r['evidence_status']=='verified' and r['sign_language']=='sfb' and r['inference_ready'] is False
 assert r['metrics']['wer']==.4667 and r['train_examples']==145
 assert 'secret sentence' not in json.dumps(result) and '/private/weights' not in json.dumps(result) and str(report) not in json.dumps(result)
 assert hashlib.sha256(path.read_bytes()).hexdigest()==before

def test_changed_evidence_never_presented_as_verified(tmp_path,monkeypatch):
 report,_=setup(tmp_path,monkeypatch);report.write_text('{"changed":true}')
 r=research_status.recent_experiments()['entries'][0];assert r['evidence_status']=='changed' and r['metrics'] is None

def test_missing_catalog_does_not_create_database(tmp_path,monkeypatch):
 path=tmp_path/'missing.sqlite';monkeypatch.setenv('SIGNACULTURE_RESEARCH_CATALOG',str(path));assert research_status.recent_experiments()['available'] is False;assert not path.exists()

def test_external_report_is_not_read(tmp_path,monkeypatch):
 _,path=setup(tmp_path,monkeypatch)
 with sqlite3.connect(path) as db:db.execute('UPDATE research_experiment_index SET report_path=?',('/etc/passwd',))
 r=research_status.recent_experiments()['entries'][0];assert r['evidence_status']=='outside_allowed_roots' and r['metrics'] is None


def test_catalog_metrics_must_match_hashed_report(tmp_path,monkeypatch):
 _,path=setup(tmp_path,monkeypatch)
 with sqlite3.connect(path) as db:db.execute('UPDATE model_runs SET metrics_json=?',(json.dumps({'exact':1}),))
 r=research_status.recent_experiments()['entries'][0];assert r['evidence_status']=='metrics_mismatch' and r['metrics'] is None


def test_conflicting_index_language_is_not_trusted(tmp_path,monkeypatch):
 _,path=setup(tmp_path,monkeypatch)
 with sqlite3.connect(path) as db:db.execute("UPDATE research_experiment_index SET sign_language='fsl'")
 r=research_status.recent_experiments()['entries'][0];assert r['evidence_status']=='language_mismatch' and r['metrics'] is None
