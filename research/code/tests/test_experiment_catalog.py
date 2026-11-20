import sqlite3,json
import pytest
from training.register_recent_experiments import register

def setup_run(tmp_path,language='LSFB'):
 folder=tmp_path/'run';folder.mkdir();(folder/'RESULT.json').write_text(json.dumps({'status':'complete','validation':{'exact':0}}));(folder/'MANIFEST.json').write_text(json.dumps({'language':language,'data_hash':'original-recorded-digest'}));(tmp_path/'manifest.json').write_text('{}')
 db=sqlite3.connect(':memory:');db.execute('CREATE TABLE model_runs(run_id TEXT PRIMARY KEY,model_base TEXT,model_revision TEXT,dataset_sha256 TEXT,hyperparameters_json TEXT,metrics_json TEXT,checkpoint_uri TEXT)');return db

def test_idempotent_metadata_preserves_recorded_dataset_identity(tmp_path):
 db=setup_run(tmp_path);runs=[('x','sfb','lexical_projection','run','manifest.json')]
 register(db,tmp_path,runs);register(db,tmp_path,runs)
 assert db.execute('SELECT count(*) FROM model_runs').fetchone()[0]==1
 assert db.execute('SELECT dataset_sha256 FROM model_runs').fetchone()[0]=='original-recorded-digest'
 m=json.loads(db.execute('SELECT hyperparameters_json FROM model_runs').fetchone()[0]);assert m['catalog_manifest_matches_run'] is False
 assert db.execute('SELECT qualified_review FROM research_experiment_index').fetchone()[0]==0

def test_reject_language_mismatch_before_writing(tmp_path):
 db=setup_run(tmp_path,language='fsl')
 with pytest.raises(ValueError,match='language mismatch'):register(db,tmp_path,[('x','sfb','lexical_projection','run','manifest.json')])
 assert db.execute('SELECT count(*) FROM model_runs').fetchone()[0]==0
