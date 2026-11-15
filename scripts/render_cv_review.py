"""Render observed 2D landmarks at exact common video timestamps; no imputation."""
import argparse,cv2,gzip,json,html
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--candidates',type=Path,default=Path('.runtime/cv-v4'));p.add_argument('--output',type=Path,default=Path('artifacts/cv-review/v4'));a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
rows=json.loads((a.candidates/'results.json').read_text());index='<meta charset="utf-8"><style>body{font:16px sans-serif}img{max-width:48%}article{padding:20px;border-bottom:1px solid #aaa}</style><h1>Observed source keypoints — before / candidate</h1><p>Exact shared timestamps. No smoothing, filled gaps or avatar geometry. Missing hands have no overlay.</p>'
for row in rows:
 cid=row['capture_id'];source=Path('.runtime/captures')/cid;old=json.load(gzip.open(source/'animics-landmarks.json.gz','rt'));new={f['time_ms']:f for f in json.load(gzip.open(a.candidates/cid/'animics-landmarks.json.gz','rt'))};common=[f for f in old if f['time_ms']in new];changed=[f for f in common if any(bool(f.get(k))!=bool(new[f['time_ms']].get(k)) for k in ['LLM','RLM'])];frames={f['time_ms']:f for f in changed+[common[int((len(common)-1)*u)] for u in [.2,.5,.8]]}
 cap=cv2.VideoCapture(str(source/'source.mp4'));index+=f'<article><h2>{html.escape(row["source_id"])}</h2><p>{row["engineering_gate"]} · accuracy unvalidated</p>'
 for t,f in frames.items():
  cap.set(cv2.CAP_PROP_POS_MSEC,t);ok,image=cap.read()
  if not ok:continue
  index+=f'<h3>{t} ms</h3>'
  for name,raw in [('before',f),('candidate',new[t])]:
   im=image.copy();h,w=im.shape[:2]
   for key,color in [('LLM',(0,220,0)),('RLM',(255,180,0))]:
    hand=raw.get(key)
    if hand:
     for i,pt in enumerate(hand):
      q=(round(pt['x']*w),round(pt['y']*h));cv2.circle(im,q,2,color,-1)
      if i:
       parent=0 if i in [1,5,9,13,17] else i-1;v=hand[parent];cv2.line(im,q,(round(v['x']*w),round(v['y']*h)),color,1)
   filename=f'{cid}-{t}-{name}.png';cv2.imwrite(str(a.output/filename),im);index+=f'<img alt="{name}" src="{filename}">'
 index+='</article>';cap.release()
(a.output/'index.html').write_text(index)
print('Reviewed-source gallery:',a.output/'index.html')
