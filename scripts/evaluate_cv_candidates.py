"""Run real candidate extraction on the fixed phrase source set, without publishing it."""
import argparse,hashlib,gzip,json,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
ROOT=Path(__file__).resolve().parents[1]

def run(row,output):
 folder=output/row['capture_id'];folder.mkdir(parents=True,exist_ok=True)
 source=ROOT/'.runtime/captures'/row['capture_id']/'source.mp4';raw=folder/'animics-landmarks.json.gz'
 expected=row.get('provenance',{}).get('source_sha256')
 if expected and hashlib.sha256(source.read_bytes()).hexdigest()!=expected:
  raise ValueError('Source video checksum mismatch: '+row['capture_id'])
 if not raw.exists():
  with (folder/'worker.log').open('w') as log:
   subprocess.run([str(ROOT/'.venv-motion/bin/python'),str(ROOT/'backend/app/workers/capture.py'),str(source),str(folder/'motion.json'),str(folder/'progress.json'),'--raw-animics',str(raw)],check=True,stdout=log,stderr=log)
 old=json.load(gzip.open(source.with_name(raw.name),'rt'));new=json.load(gzip.open(raw,'rt'));bytime={f['time_ms']:f for f in new};common=[(f,bytime[f['time_ms']]) for f in old if f['time_ms'] in bytime]
 sides={}
 for side,key in [('left','LLM'),('right','RLM')]:
  a=sum(bool(x.get(key)) for x,y in common);b=sum(bool(y.get(key)) for x,y in common)
  sides[side]={'before_observed':a,'candidate_observed':b,'common_frames':len(common),'lost':sum(bool(x.get(key)) and not y.get(key) for x,y in common),'recovered':sum(not x.get(key) and bool(y.get(key)) for x,y in common)}
 body_before=sum(bool(x.get('PLM')) for x,y in common);body_after=sum(bool(y.get('PLM')) for x,y in common)
 passed=len(common)>=len(old)*.95 and body_after>=body_before and all(s['lost']==0 for s in sides.values())
 result={'source_id':row['source_id'],'capture_id':row['capture_id'],'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'native_frames_before':len(old),'native_frames_candidate':len(new),'common_frames':len(common),'body_before':body_before,'body_candidate':body_after,'hands':sides,'engineering_gate':'no_observation_loss' if passed else 'review_regression_or_incomparable','linguistic_review':'pending','keypoint_accuracy_validated':False,'active_asset_replaced':False}
 (folder/'comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'.runtime/cv-v4');a=p.parse_args()
 ids=set()
 for f in (ROOT/'artifacts/phrase-review/after').glob('*-input.json'):
  ids.update(g['source_id'] for g in json.loads(f.read_text())['lookup']['gloss_sequence'] if g['source_id'].startswith('DICTASIGN_PORTAL_'))
 rows=[r for r in json.loads((ROOT/'.runtime/library.json').read_text()) if r.get('source_id') in ids]
 assert ids and len(rows)>=len(ids),'Missing real phrase source assets'
 with ThreadPoolExecutor(max_workers=2) as pool:results=list(pool.map(lambda r:run(r,a.output),rows))
 (a.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')
