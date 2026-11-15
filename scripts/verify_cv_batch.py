"""Check raw observation integrity without interpreting detector confidence as accuracy."""
import argparse,gzip,json,math
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,default=Path('.runtime/cv-v4-all'));a=p.parse_args();results=json.loads((a.batch/'results.json').read_text());checked=frames=0;errors=[]
for r in results['results']:
 if r['status']!='complete':continue
 try:
  raw=json.load(gzip.open(a.batch/'captures'/r['source_sha256']/'animics-landmarks.json.gz','rt'));assert raw,'Empty capture'
  last=-1
  for f in raw:
   assert f['time_ms']>last,'Non-monotonic timestamps';last=f['time_ms']
   for k,n in [('PLM',33),('PWLM',33),('FLM',468),('LLM',21),('RLM',21)]:
    pts=f.get(k)
    if pts is not None:
     assert len(pts)==n,(k,len(pts))
     assert all(math.isfinite(p[v]) for p in pts for v in ['x','y','z']),'Nonfinite coordinate'
   for side,k in [('left','LLM'),('right','RLM')]:assert f['hand_diagnostics'][side]['observed']==bool(f[k]),'Observation flag mismatch'
   assert f['LWLM'] is None and f['RWLM'] is None,'Fabricated independent world hands'
  frames+=len(raw);checked+=1
 except Exception as e:errors.append({'source_sha256':r['source_sha256'],'error':str(e)})
report={'checked_videos':checked,'checked_raw_frames':frames,'integrity_errors':errors,'keypoint_accuracy_validated':False};(a.batch/'integrity.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report));raise SystemExit(bool(errors))
