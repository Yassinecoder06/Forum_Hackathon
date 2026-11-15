"""Audit original image landmarks, never smoothed avatar output."""
import argparse,gzip,json,math
from pathlib import Path

def audit(path):
 frames=json.load(gzip.open(path,'rt')); out={'frames':len(frames),'hands':{}}
 for side,key,wrist in [('left','LLM',15),('right','RLM',16)]:
  present=[];errors=[];jumps=[];last=None
  for f in frames:
   hand=f.get(key);body=f.get('PLM');present.append(bool(hand))
   if not hand:last=None;continue
   if body:
    errors.append(math.hypot(hand[0]['x']-body[wrist]['x'],hand[0]['y']-body[wrist]['y']))
   if last:
    # Translation-invariant fingertip jump relative to wrist; diagnostic only.
    jumps.append(max(math.hypot((hand[i]['x']-hand[0]['x'])-(last[i]['x']-last[0]['x']),(hand[i]['y']-hand[0]['y'])-(last[i]['y']-last[0]['y'])) for i in [4,8,12,16,20]))
   last=hand
  run=longest=0
  for v in present:run=0 if v else run+1;longest=max(run,longest)
  out['hands'][side]={'observed':sum(present),'coverage':sum(present)/len(frames),'longest_missing_frames':longest,'wrist_error_normalized_max':max(errors,default=None),'large_wrist_disagreement_frames':sum(e>.12 for e in errors),'relative_tip_jump_max':max(jumps,default=None)}
 return out
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.runtime/captures'));p.add_argument('--output',type=Path,required=True);a=p.parse_args();rows=[]
 for f in sorted(a.root.glob('*/animics-landmarks.json.gz')):
  try:rows.append({'capture_id':f.parent.name,**audit(f)})
  except Exception as e:rows.append({'capture_id':f.parent.name,'error':str(e)})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(rows,indent=2));print(json.dumps({'captures':len(rows),'errors':sum('error'in r for r in rows)}))
