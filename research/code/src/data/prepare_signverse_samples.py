"""Observed release adapter; body index availability is explicitly not confidence."""
import argparse,json,hashlib,re
from pathlib import Path
import numpy as np
from src.data.signverse import read_npz,select_track,preprocess,PARTS
from src.data.visualize import write_html

def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument('--raw',type=Path,default=Path('data/poses/private/raw'))
 parser.add_argument('--output',type=Path,default=Path('data/poses/private/windows-v1'))
 parser.add_argument('--audit',type=Path,default=Path('data/audit/sample_quality.json'))
 parser.add_argument('--preview',type=Path,default=Path('research/motion/fsl_pose_preview.html'))
 args=parser.parse_args()
 out=args.output;out.mkdir(parents=True,exist_ok=False);rows=[];windows={s:[] for s in ('train','val','test')};masks={s:[] for s in windows};groups={s:[] for s in windows}
 for folder in sorted(args.raw.iterdir()):
  m=json.loads((folder/'manifest.json').read_text());assert m['sign_language']=='fsl'
  z=read_npz(folder/'poses.npz');assert str(z['video_id'].item())==m['video_id'];assert int(z['fps'])==24
  frames=[];original_scores=[]
  for f in z['frame_payloads'].tolist():
   frame={};scores={}
   for i in range(int(f.get('num_persons',0))):
    key=f'person_{i:03}';person={}
    for part,(a,b) in PARTS.items():
     xy=np.asarray(f.get(key+'_'+part+'_keypoints',np.zeros((b-a,2))));score=np.asarray(f.get(key+'_'+part+'_scores',np.zeros(b-a)))
     scores[key+'_'+part]=score
     # Observed body field is an index/availability vector, not detector probability.
     valid=(score>=0).astype(np.float32) if part=='body' else np.where((score>=0)&(score<=1),score,0)
     person[part]=np.c_[xy,valid]
    frame[key]=person
   frames.append(frame);original_scores.append(scores)
  raw,ids,amb=select_track(frames);arrays,q=preprocess(raw,z['frame_indices'],24)
  arrays['mask'][amb]=False;arrays['normalized_xy'][amb]=0;arrays['smoothed_xy'][amb]=0
  arrays['original_body_scores']=np.array([f.get('person_000_body_scores',np.full(18,-1)) for f in z['frame_payloads'].tolist()])
  np.savez_compressed(out/(m['video_id']+'.npz'),**arrays)
  count=0
  for start in range(0,len(raw)-15,8):
   mask=arrays['mask'][start:start+16];xy=arrays['normalized_xy'][start:start+16]
   if mask[:,[2,5]].all() and mask[:,18:60].mean()>=.7 and mask[:,60:].mean()>=.7 and mask.mean()>=.7:
    windows[m['split']].append(xy);masks[m['split']].append(mask);count+=1
  if count:groups[m['split']].append(m['video_id'])
  vtts=list(folder.glob('*.vtt'));segments=sum(len(re.findall(r' --> ',x.read_text())) for x in vtts)
  rows.append({'sign_language':'fsl','video_id':m['video_id'],'split':m['split'],'frames':len(raw),'recording_duration_s':len(raw)/24,'subtitle_languages':[x.name.split('.')[-2] for x in vtts],'subtitle_segments':segments,'transcription_source':'released VTT; manual/automatic unknown','caption_json_present':False,'record_language_validation':'pose video_id joined to both metadata language tables; no embedded language field','windows':count,'quality':q,'body_score_semantics':'index availability, not probabilistic confidence','gloss_annotations':False})
  if count and m['split']=='val':write_html(args.preview,{k:v[:2400] if getattr(v,'ndim',0)>0 else v for k,v in arrays.items()})
 manifest={'sign_language':'fsl','fps':24,'groups':groups,'counts':{s:len(v) for s,v in windows.items()},'sha256':{},'limitations':['body availability substituted explicitly; no body confidence available','2D coordinates; unknown signer identity','convenience-selected records; official source-video splits retained','missing embedded signing-language labels; metadata joins checked']}
 for s,v in windows.items():
  if v:
   path=out/(s+'.npz');np.savez_compressed(path,xy=np.stack(v),mask=np.stack(masks[s]));manifest['sha256'][s]=hashlib.sha256(path.read_bytes()).hexdigest()
 (out/'manifest.json').write_text(json.dumps(manifest,indent=2));args.audit.write_text(json.dumps({'records':rows,'experiment':manifest},indent=2));print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
