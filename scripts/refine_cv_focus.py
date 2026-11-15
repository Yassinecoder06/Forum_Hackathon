"""Recover only missing observations using agreeing forward/reverse image trackers."""
import sys,json,gzip,copy,math,hashlib
from pathlib import Path
import cv2,mediapipe as mp
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.workers.hand_tracking import assign_hands,consistent_hand,agreed_candidates
from types import SimpleNamespace
from backend.app.workers.capture import pose_frame
FOCUS=['pain','livre','travail','chat','chercher'];OUT=ROOT/'.runtime/cv-v5-focus';OUT.mkdir(parents=True,exist_ok=True)
phrases=json.loads((ROOT/'data/source_inventory/cv_phrases15_review.json').read_text());words={w['concept']:w for p in phrases for w in p['words']};batch=json.loads((ROOT/'.runtime/cv-v4-all/results.json').read_text());sources={r['source_sha256']:r['aliases'][0]['source'] for r in batch['results'] if r['status']=='complete'}
def points(hand):return [{'x':float(p.x),'y':float(p.y),'z':float(p.z)} for p in hand.landmark]
summary=[]
for name in FOCUS:
 w=words[name];sha=w['source_sha256'];assert hashlib.sha256(Path(sources[sha]).read_bytes()).hexdigest()==sha;folder=OUT/sha;folder.mkdir(exist_ok=True);old=json.load(gzip.open(ROOT/'.runtime/cv-v4-all/captures'/sha/'animics-landmarks.json.gz','rt'));frames=copy.deepcopy(old);cap=cv2.VideoCapture(sources[sha]);images=[]
 for f in old:
  cap.set(cv2.CAP_PROP_POS_MSEC,f['time_ms']);ok,im=cap.read();assert ok
  if max(im.shape[:2])>960:im=cv2.resize(im,None,fx=960/max(im.shape[:2]),fy=960/max(im.shape[:2]))
  images.append(cv2.cvtColor(im,cv2.COLOR_BGR2RGB))
 cap.release();hypotheses=[]
 for backwards in [False,True]:
  found={}
  with mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=.35,min_tracking_confidence=.4) as detector:
   for i in (reversed(range(len(images))) if backwards else range(len(images))):
    detected=detector.process(images[i]);found[i]=list(detected.multi_hand_landmarks or [])
  hypotheses.append(found)
 recovered=[];rejected=0
 for i,f in enumerate(frames):
  body=f.get('PLM')
  if not body:continue
  h,width=images[i].shape[:2];candidates=[]
  candidates=agreed_candidates(hypotheses[0][i],hypotheses[1][i],width,h)
  rejected+=sum(not any(consistent_hand(a,b,width,h) for b in hypotheses[1][i]) for a in hypotheses[0][i])
  wrists={s:(body[k]['x'],body[k]['y']) for s,k in [('left',15),('right',16)]};existing={s:bool(f.get(k)) for s,k in [('left','LLM'),('right','RLM')]}
  chosen=assign_hands([(c.landmark[0].x,c.landmark[0].y) for c in candidates],wrists,existing,max_distance=.10)
  for side,j in chosen.items():
   wi=15 if side=='left' else 16
   if body[wi].get('visibility',0)<.5:continue
   key='LLM' if side=='left' else 'RLM';f[key]=points(candidates[j]);f['hand_diagnostics'][side]={'observed':True,'source':'forward_reverse_agreement','review_required':True,'wrist_disagreement_normalized':math.dist(wrists[side],(candidates[j].landmark[0].x,candidates[j].landmark[0].y))};recovered.append({'frame':i,'time_ms':f['time_ms'],'side':side})
 # Build the native pose from detector observations, preserving body/face data.
 baseline=json.loads((ROOT/'.runtime/cv-v4-all/captures'/sha/'motion.json').read_text());native=copy.deepcopy(baseline);native['frames']=[]
 def landmarks(pts,world=False):
  return SimpleNamespace(landmark=[SimpleNamespace(x=p['x'],y=-p['y'] if world else p['y'],z=-p['z'] if world else p['z']) for p in pts]) if pts else None
 for f in frames:
  if not f.get('PLM'):continue
  result=SimpleNamespace(pose_landmarks=landmarks(f['PLM']),pose_world_landmarks=landmarks(f.get('PWLM'),True),left_hand_landmarks=landmarks(f.get('LLM')),right_hand_landmarks=landmarks(f.get('RLM')),face_landmarks=landmarks(f.get('FLM')))
  joints,hands,face=pose_frame(result,f['image_width'],f['image_height']);native['frames'].append({'time_ms':f['time_ms'],'joints':joints,'face':face,'hand_tracking':f['hand_diagnostics']})
 if native['frames'][-1]['time_ms']<native['duration_ms']:
  last=copy.deepcopy(native['frames'][-1]);last['time_ms']=native['duration_ms'];native['frames'].append(last)
 native['capture_method']='roi-v4+forward-reverse-focused-v5';native['tracked_joints']=sorted(set().union(*(f['joints'] for f in native['frames'])))
 for side,key in [('left','LLM'),('right','RLM')]:native['quality'][side+'_hand_coverage']=sum(bool(f.get(key)) for f in frames)/len(frames)
 (folder/'motion.json').write_text(json.dumps(native,allow_nan=False))
 for a,b in zip(old,frames):
  for key in ['PLM','PWLM','FLM','LLM','RLM']:
   if a.get(key):assert a[key]==b[key], 'Existing observations were rewritten'
 with gzip.open(folder/'animics-landmarks.json.gz','wt') as out:json.dump(frames,out,allow_nan=False)
 r={'concept':name,'source_sha256':sha,'source':sources[sha],'frames':len(frames),'recovered':recovered,'lost_existing_observations':0,'disagreeing_hypotheses_rejected':rejected,'accuracy_validated':False,'active_assets_replaced':False};(folder/'result.json').write_text(json.dumps(r,indent=2));summary.append(r);print(json.dumps(r),flush=True)
(OUT/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
