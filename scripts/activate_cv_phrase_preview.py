"""Reversible local research-preview activation; never upgrades approval or rights."""
import json,sys,shutil,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.services import capture_library as store
wanted={w['source_id']:w for p in json.loads((ROOT/'data/source_inventory/cv_phrases15_review.json').read_text()) for w in p['words']};out=ROOT/'.runtime/cv-live-loop';out.mkdir(exist_ok=True);manifest=[]
# Cover the ordinary French-input route too (e.g. travaille -> TRAVAILLER).
by_capture={a['capture_id']:r['source_sha256'] for r in json.loads((ROOT/'.runtime/cv-v4-all/results.json').read_text())['results'] if r['status']=='complete' for a in r['aliases']}
by_gloss={r['gloss']['gloss_id']:r for r in store.library()}
for scenario in json.loads((ROOT/'data/source_inventory/cv_phrases15_review.json').read_text()):
 request=urllib.request.Request('http://127.0.0.1:8000/gloss-lookup',data=json.dumps({'text':scenario['phrase']}).encode(),headers={'Content-Type':'application/json'})
 for g in json.load(urllib.request.urlopen(request,timeout=30))['gloss_sequence']:
  row=by_gloss.get(g['gloss_id']);digest=by_capture.get(row.get('capture_id')) if row else None
  if digest:wanted.setdefault(g['source_id'],{'source_sha256':digest})
with store.library_lock():
 registry=store.library();snapshot=out/'library-before.json'
 if not snapshot.exists():shutil.copy2(store.storage()/'library.json',snapshot)
 for row in registry:
  word=wanted.get(row.get('source_id')) or wanted.get(row['gloss'].get('source_id'))
  if not word:continue
  sha=word['source_sha256'];folder=ROOT/'.runtime/cv-v5-focus'/sha
  if not (folder/'motion.json').exists():folder=ROOT/'.runtime/cv-v4-all/captures'/sha
  new=json.loads((folder/'motion.json').read_text());target=Path(row['motion']['path']);target=target if target.is_absolute() else ROOT/target;old=json.loads(target.read_text());backup=out/'backups'/target.name;backup.parent.mkdir(exist_ok=True)
  if not backup.exists():shutil.copy2(target,backup)
  new['attribution']=old.get('attribution',row.get('attribution'));new['cv_review']='research_preview_pending_accuracy_review';new['cv_source_sha256']=sha;store.write_json(target,new)
  row['quality']=new['quality'];row['motion'].update(fps=new['fps'],duration_ms=new['duration_ms'],quality_status='cv_corrected_research_preview_pending_review');row['raw_animics_path']=str(folder/'animics-landmarks.json.gz');row['cv_generation']=new['capture_method']
  manifest.append({'source_id':row['source_id'],'motion_id':row['motion']['motion_id'],'backup':str(backup),'capture_method':new['capture_method'],'rights_status':row['motion']['rights_status'],'linguistic_review':'pending'})
 store.write_json(store.storage()/'library.json',registry)
store.write_json(out/'activation.json',manifest);print('Activated',len(manifest),'source rows with backups and pending review.')
