"""Exact-frame source review of focused CV repairs, independent of avatar motion."""
import cv2,gzip,json,subprocess,html
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];BASE=ROOT/'.runtime/cv-v5-focus';OUT=ROOT/'artifacts/cv-review/focus-fixed';OUT.mkdir(parents=True,exist_ok=True);results=json.loads((BASE/'results.json').read_text());page='<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1000px;margin:auto;padding:20px}video{width:100%}img{max-width:100%}</style><h1>Focused CV repairs</h1><p>Left: CV v4. Right: agreeing forward/reverse estimates on actual source frames. Green: left hand; cyan: right. Occluded/ambiguous points remain unvalidated. No avatar or sign interpolation.</p>'
def draw(im,f):
 h,w=im.shape[:2]
 for k,color in [('LLM',(0,230,0)),('RLM',(255,180,0))]:
  pts=f.get(k)
  if pts:
   for i,p in enumerate(pts):
    q=(round(p['x']*w),round(p['y']*h));cv2.circle(im,q,2,color,-1)
    if i:
     a=pts[0 if i in [1,5,9,13,17] else i-1];cv2.line(im,q,(round(a['x']*w),round(a['y']*h)),color,1)
 return im
for r in results:
 sha=r['source_sha256'];old=json.load(gzip.open(ROOT/'.runtime/cv-v4-all/captures'/sha/'animics-landmarks.json.gz','rt'));new=json.load(gzip.open(BASE/sha/'animics-landmarks.json.gz','rt'));motion=json.loads((BASE/sha/'motion.json').read_text());cap=cv2.VideoCapture(r['source']);video=OUT/(r['concept']+'.mp4');recovered={x['frame'] for x in r['recovered']}
 process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','960x400','-framerate',str(motion['fps']),'-i','-','-an','-r','30','-c:v','libx264','-threads','2','-preset','fast','-crf','23','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
 stats={}
 for side,k,wi,ei in [('left','LLM',15,13),('right','RLM',16,14)]:
  def count(frames):return sum(not f.get(k) and bool(f.get('PLM')) and f['PLM'][wi].get('visibility',0)>=.75 and f['PLM'][ei].get('visibility',0)>=.75 and 0<f['PLM'][wi]['y']<1 and f['PLM'][wi]['y']<f['PLM'][ei]['y']-.01 for f in frames)
  stats[side]={'before_visible_raised_wrist_missing':count(old),'after_visible_raised_wrist_missing':count(new)}
 for i,(a,b) in enumerate(zip(old,new)):
  assert a['time_ms']==b['time_ms'];cap.set(cv2.CAP_PROP_POS_MSEC,a['time_ms']);ok,image=cap.read();assert ok
  canvas=np.full((400,960,3),235,np.uint8)
  for j,(label,f) in enumerate([('CV v4',a),('Focused recovery (review required)',b)]):
   im=draw(image.copy(),f);h,w=im.shape[:2];scale=min(460/w,300/h);im=cv2.resize(im,(round(w*scale),round(h*scale)));x=j*480+(480-im.shape[1])//2;canvas[40:40+im.shape[0],x:x+im.shape[1]]=im
   cv2.putText(canvas,label,(j*480+10,25),cv2.FONT_HERSHEY_SIMPLEX,.48,(20,20,20),1)
   text='L: '+('observed' if f.get('LLM') else 'MISSING')+' | R: '+('observed' if f.get('RLM') else 'MISSING');cv2.putText(canvas,text,(j*480+10,365),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
  cv2.putText(canvas,r['concept']+' | '+str(a['time_ms'])+' ms | source-frame estimates, accuracy unvalidated',(12,390),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
  process.stdin.write(canvas.tobytes())
  if i in recovered:cv2.imwrite(str(OUT/f'{r["concept"]}-{a["time_ms"]}.png'),canvas)
 process.stdin.close();assert process.wait()==0;cap.release();r['raised_wrist_review']=stats;page+=f'<h2>{html.escape(r["concept"])}</h2><p>New observations: {len(r["recovered"])}; no existing observations removed. Finger accuracy pending review.</p><video controls preload="none" src="{r["concept"]}.mp4"></video>'
(OUT/'results.json').write_text(json.dumps(results,indent=2)+'\n');(OUT/'index.html').write_text(page+'<script>document.querySelectorAll("video").forEach(v=>v.playbackRate=.5)</script>');print(json.dumps([{k:r[k] for k in ['concept','recovered','raised_wrist_review']} for r in results]))
