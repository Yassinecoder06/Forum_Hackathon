"""Resumable local CV-v4 re-extraction, with exact-byte dedup and preserved originals."""
import argparse,concurrent.futures,gzip,hashlib,json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,ensure_ascii=False,indent=2)+'\n');tmp.replace(p)
def inventory():
 lib={r.get('capture_id'):r for r in json.loads((ROOT/'.runtime/library.json').read_text())};groups={};excluded=[]
 for source in sorted((ROOT/'.runtime/captures').glob('*/source.mp4')):
  meta=source.parent/'source_provenance.json';d=json.loads(meta.read_text()) if meta.exists() else {};r=lib.get(source.parent.name,{})
  if not (d.get('source')=='Dicta-Sign public lexical portal' or d.get('canonical_licence_source') or (r.get('source_id','').startswith('WMC_') and r.get('motion',{}).get('synthetic') is False)):
   excluded.append(source.parent.name);continue
  digest=sha(source);expected=d.get('source_sha256',d.get('sha256'))
  if expected and digest!=expected:raise ValueError('Source checksum mismatch '+str(source))
  item={'source':str(source),'capture_id':source.parent.name,'source_id':r.get('source_id') or ('DICTASIGN_PORTAL_'+d['concept_id'] if d.get('concept_id') else 'licensed_commons'),'source_url':d.get('video_url',d.get('canonical_licence_source',r.get('source',''))),'baseline_raw':str(source.with_name('animics-landmarks.json.gz'))}
  groups.setdefault(digest,[]).append(item)
 source=ROOT/'data/public_examples/bonjour.webm'
 if source.exists():groups.setdefault(sha(source),[]).append({'source':str(source),'capture_id':'curated-bonjour','source_id':'LSF_BONJOUR_COMMONS','source_url':'https://commons.wikimedia.org/wiki/File:LL-Q33302_(fsl)-Laura_Jauvert-Bonjour.webm','baseline_raw':None})
 return groups,excluded

def process(item,out,fingerprint):
 digest,aliases=item;folder=out/'captures'/digest;result=folder/'result.json';raw=folder/'animics-landmarks.json.gz';motion=folder/'motion.json';folder.mkdir(parents=True,exist_ok=True)
 if result.exists():
  cached=json.loads(result.read_text())
  if cached.get('worker_fingerprint')==fingerprint and cached.get('status')=='complete' and raw.exists() and motion.exists():return cached
 start=time.monotonic();source=Path(aliases[0]['source']);assert sha(source)==digest
 for attempt in range(2):
  with (folder/'worker.log').open('w') as log:
   try:
    run=subprocess.run([str(ROOT/'.venv-motion/bin/python'),str(ROOT/'backend/app/workers/capture.py'),str(source),str(motion),str(folder/'progress.json'),'--raw-animics',str(raw)],cwd=ROOT,stdout=log,stderr=log,timeout=240)
    if run.returncode==0:break
   except subprocess.TimeoutExpired:pass
 else:
  d={'source_sha256':digest,'aliases':aliases,'status':'failed','worker_fingerprint':fingerprint,'error':(folder/'worker.log').read_text()[-1500:]};save(result,d);return d
 new=json.load(gzip.open(raw,'rt'));bytime={f['time_ms']:f for f in new};comparisons=[]
 for alias in aliases:
  base=Path(alias['baseline_raw']) if alias['baseline_raw'] else None
  if not base or not base.exists():comparisons.append({'capture_id':alias['capture_id'],'gate':'review_no_raw_baseline'});continue
  old=json.load(gzip.open(base,'rt'));common=[(f,bytime[f['time_ms']]) for f in old if f['time_ms']in bytime];hands={}
  for s,k in [('left','LLM'),('right','RLM')]:hands[s]={'old_observed':sum(bool(x.get(k)) for x,y in common),'new_observed':sum(bool(y.get(k)) for x,y in common),'lost':sum(bool(x.get(k)) and not y.get(k) for x,y in common),'recovered':sum(not x.get(k) and bool(y.get(k)) for x,y in common)}
  body_lost=sum(bool(x.get('PLM')) and not y.get('PLM') for x,y in common)
  comparable=len(common)>=len(old)*.95
  gate='no_observation_loss' if comparable and body_lost==0 and all(s['lost']==0 for s in hands.values()) else 'review_regression_or_incomparable'
  comparisons.append({'capture_id':alias['capture_id'],'old_frames':len(old),'new_frames':len(new),'common_frames':len(common),'body_lost':body_lost,'hands':hands,'gate':gate})
 d={'source_sha256':digest,'aliases':aliases,'status':'complete','worker_fingerprint':fingerprint,'seconds':round(time.monotonic()-start,2),'quality':json.loads(motion.read_text())['quality'],'comparisons':comparisons,'keypoint_accuracy_validated':False,'active_assets_replaced':False};save(result,d);return d
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'.runtime/cv-v4-all');p.add_argument('--workers',type=int,default=3);a=p.parse_args();groups,excluded=inventory();fingerprint=hashlib.sha256(b''.join((ROOT/'backend/app/workers'/f).read_bytes() for f in ['capture.py','hand_tracking.py'])).hexdigest();save(a.output/'inventory.json',{'groups':groups,'excluded_unprovenanced_uploads':excluded,'worker_fingerprint':fingerprint});results=[];start=time.monotonic()
 with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(a.workers,4))) as pool:
  futures={pool.submit(process,item,a.output,fingerprint):item[0] for item in groups.items()}
  for f in concurrent.futures.as_completed(futures):
   try:r=f.result()
   except Exception as e:r={'source_sha256':futures[f],'status':'failed','error':str(e)}
   results.append(r);summary={'total_unique_videos':len(groups),'source_entries':sum(map(len,groups.values())),'processed':len(results),'complete':sum(x['status']=='complete' for x in results),'failed':sum(x['status']=='failed' for x in results),'seconds':round(time.monotonic()-start,1),'active_assets_replaced':False};save(a.output/'status.json',summary)
   if len(results)%20==0 or r['status']=='failed':print(json.dumps(summary),flush=True)
 save(a.output/'results.json',{'summary':summary,'results':results});print(json.dumps(summary),flush=True)
