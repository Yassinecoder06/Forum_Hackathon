"""Register measured recent runs without changing source pairs, permissions or reviews."""
import sqlite3,json,hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
RUNS=[
 ('lsfb-narrow-context-v1','sfb','lexical_projection','.runtime/lsfb-narrow-context-run-v1','.runtime/lsfb-narrow-context-data-v1/SCAN_REPORT.json'),
 ('lsfb-narrow-current-v1','sfb','lexical_projection','.runtime/lsfb-narrow-current-run-v1','.runtime/lsfb-narrow-context-data-v1/SCAN_REPORT.json'),
 ('lsfb-extracted-v1','sfb','lexical_projection','.runtime/lsfb-extracted-train-v1','/home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont-usable-v1/SCAN_REPORT.json'),
 ('geoquery-full-graph-v1','sfb','temporal_graph','.runtime/geoquery-full-graph-v1','.runtime/geoquery-graph-codec-v1/AUDIT.json'),
 ('fsl-mt5-small-comparison-v1','fsl','source_annotated_gloss','research/private/fsl-mt5-small-comparison-v1','data/parallel/manifest.json'),
 ('fsl-t5-base-comparison-v3','fsl','source_annotated_gloss','research/private/fsl-t5-base-comparison-v3','data/parallel/manifest.json'),
]

def register(db,root=ROOT,runs=RUNS):
 records=[]
 for run_id,lang,representation,folder,data_manifest in runs:
  folder=root/folder;path=folder/'RESULT.json'
  if not path.exists():continue
  result=json.loads(path.read_text())
  if result.get('status')!='complete':continue
  dataset_path=root/data_manifest
  if not dataset_path.is_file():raise ValueError('Dataset provenance manifest absent')
  config=next((p for p in [folder/'MANIFEST.json',folder/'CONFIG.json'] if p.is_file()),None)
  if config is None:raise ValueError('Run configuration absent')
  metadata=json.loads(config.read_text());declared=metadata.get('language');expected='LSFB' if lang=='sfb' else 'fsl'
  # Runs carry explicit operator mappings; never infer target language from French text.
  if representation=='lexical_projection' and declared!='LSFB':raise ValueError('Belgian run language mismatch')
  if representation=='source_annotated_gloss' and declared!='fsl':raise ValueError('France run language mismatch')
  if representation=='temporal_graph' and metadata.get('task')!='French machine-translated input to full LSFB graph; feasibility':raise ValueError('Graph provenance mismatch')
  model=metadata.get('model','google-t5/t5-base');revision=metadata.get('revision',metadata.get('model_revision'))
  if metadata.get('model_path'):model='google/mt5-small' if 'mt5-small' in metadata['model_path'] else 'google-t5/t5-base'
  current_hash=hashlib.sha256(dataset_path.read_bytes()).hexdigest();dataset_hash=metadata.get('data_hash',current_hash);metadata={**metadata,'catalog_current_manifest_sha256':current_hash,'catalog_manifest_matches_run':current_hash==dataset_hash};metrics=result.get('validation',result.get('metrics',result));checkpoint=next((folder/n for n in ['best_adapter','adapter'] if (folder/n).is_dir()),None)
  records.append({'id':run_id,'language':lang,'representation':representation,'dataset_sha256':dataset_hash,'model':model,'revision':revision,'metadata':metadata,'metrics':metrics,'checkpoint':str(checkpoint) if checkpoint else None,'result_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'report_path':str(path),'qualified_review':False})
 with db:
  db.execute('CREATE TABLE IF NOT EXISTS research_experiment_index(run_id TEXT PRIMARY KEY REFERENCES model_runs(run_id),sign_language TEXT NOT NULL CHECK(sign_language IN (\'fsl\',\'sfb\')),representation TEXT NOT NULL,result_sha256 TEXT NOT NULL,report_path TEXT NOT NULL,qualified_review INTEGER NOT NULL CHECK(qualified_review=0))')
  for r in records:
   old=db.execute('SELECT sign_language,representation FROM research_experiment_index WHERE run_id=?',(r['id'],)).fetchone()
   if old and old!=(r['language'],r['representation']):raise ValueError('Refusing cross-language/representation update')
   db.execute('INSERT INTO model_runs VALUES(?,?,?,?,?,?,?) ON CONFLICT(run_id) DO UPDATE SET model_base=excluded.model_base,model_revision=excluded.model_revision,dataset_sha256=excluded.dataset_sha256,hyperparameters_json=excluded.hyperparameters_json,metrics_json=excluded.metrics_json,checkpoint_uri=excluded.checkpoint_uri',(r['id'],r['model'],r['revision'],r['dataset_sha256'],json.dumps(r['metadata']),json.dumps(r['metrics']),r['checkpoint']))
   db.execute('INSERT INTO research_experiment_index VALUES(?,?,?,?,?,0) ON CONFLICT(run_id) DO UPDATE SET result_sha256=excluded.result_sha256,report_path=excluded.report_path',(r['id'],r['language'],r['representation'],r['result_sha256'],r['report_path']))
 return records

def main():
 path=ROOT/'data/catalog/catalog.sqlite';db=sqlite3.connect(path);db.execute('PRAGMA foreign_keys=ON');backup=ROOT/'.runtime/catalog-before-recent-experiments.sqlite'
 if not backup.exists():
  dest=sqlite3.connect(backup);db.backup(dest);dest.close()
 before={t:db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in ['pairs','sources','reviews','annotations']};records=register(db)
 assert before=={t:db.execute('SELECT count(*) FROM '+t).fetchone()[0] for t in before};assert db.execute('PRAGMA integrity_check').fetchone()==('ok',);assert not db.execute('PRAGMA foreign_key_check').fetchall()
 result={'registered_runs':[{'run_id':r['id'],'sign_language':r['language'],'representation':r['representation'],'result_sha256':r['result_sha256']} for r in records],'unchanged_source_table_counts':before,'catalog_integrity':'ok','qualified_reviews_added':0,'legacy_reviews_and_pairs_unchanged':True};(ROOT/'research/RECENT_EXPERIMENT_CATALOG.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));db.close()
if __name__=='__main__':main()
