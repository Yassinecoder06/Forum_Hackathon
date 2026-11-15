"""Review the user's exact lexical sequences against real full-corpus CV outputs."""
import collections,cv2,gzip,html,json,subprocess,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'artifacts/cv-review/phrases15';OUT.mkdir(parents=True,exist_ok=True)
sequences=[['je','vouloir','boire','eau'],['je','vouloir','manger','pain'],['mon','frère','aimer','musique'],['je','aller','école','demain'],['mon','ami','habiter','Paris'],['je','vouloir','acheter','voiture'],['je','lire','livre'],['chat','dormir'],['je','regarder','football'],['je','vouloir','apprendre','langue','signe'],['je','chercher','mon','téléphone portable'],['je','aller','cinéma','demain'],['mon','père','travail',"aujourd'hui"],['je','vouloir','café'],['je','vouloir','abandonner']]
phrases=json.loads((ROOT/'e2e/phrases.json').read_text())+['Je veux abandonner.']
entries=json.load(urllib.request.urlopen('http://127.0.0.1:8000/glossary'))['entries'];lib={r['gloss']['gloss_id']:r for r in json.loads((ROOT/'.runtime/library.json').read_text())};batch=ROOT/'.runtime/cv-v4-all';data=json.loads((batch/'results.json').read_text());byid={a['capture_id']:(r,a) for r in data['results'] if r['status']=='complete' for a in r['aliases']};review={};results=[]

def draw(im,f):
 h,w=im.shape[:2]
 for key,color in [('LLM',(0,230,0)),('RLM',(255,180,0))]:
  hand=f.get(key)
  if hand:
   for i,p in enumerate(hand):
    q=(round(p['x']*w),round(p['y']*h));cv2.circle(im,q,2,color,-1)
    if i:
     a=hand[0 if i in [1,5,9,13,17] else i-1];cv2.line(im,q,(round(a['x']*w),round(a['y']*h)),color,1)
 for key,indexes,color in [('PLM',[0,11,12,13,14,15,16],(0,200,255)),('FLM',[13,14,33,263,105,334],(240,50,220))]:
  pts=f.get(key)
  if pts:
   for i in indexes:
    p=pts[i];cv2.circle(im,(round(p['x']*w),round(p['y']*h)),2,color,-1)
 return im

def render(g):
 r,alias=byid[lib[g['gloss_id']]['capture_id']];digest=r['source_sha256'];folder=batch/'captures'/digest;frames=json.load(gzip.open(folder/'animics-landmarks.json.gz','rt'));oldpath=Path(alias['baseline_raw']) if alias['baseline_raw'] else None;old={f['time_ms']:f for f in json.load(gzip.open(oldpath,'rt'))} if oldpath and oldpath.exists() else {};motion=json.loads((folder/'motion.json').read_text());source=cv2.VideoCapture(alias['source']);video=OUT/(digest+'.mp4')
 ready=video.exists() and subprocess.check_output(['ffprobe','-v','error','-select_streams','v:0','-show_entries','stream=r_frame_rate','-of','default=nw=1:nk=1',str(video)],text=True).strip()=='30/1'
 if not ready:
  process=subprocess.Popen(['ffmpeg','-v','error','-y','-f','rawvideo','-pixel_format','bgr24','-video_size','960x400','-framerate',str(motion['fps']),'-i','-','-an','-r','30','-c:v','libx264','-threads','2','-preset','fast','-crf','25','-pix_fmt','yuv420p','-movflags','+faststart',str(video)],stdin=subprocess.PIPE)
  import numpy as np
  for i,f in enumerate(frames):
   source.set(cv2.CAP_PROP_POS_MSEC,f['time_ms']);ok,image=source.read()
   if not ok:raise ValueError('Video decode failure '+alias['source'])
   canvas=np.full((400,960,3),235,np.uint8)
   for column,(label,raw) in enumerate([('Previous raw CV',old.get(f['time_ms'])),('CV v4 detector observations',f)]):
    im=draw(image.copy(),raw) if raw else image.copy();h,w=im.shape[:2];scale=min(460/w,300/h);im=cv2.resize(im,(round(w*scale),round(h*scale)));x=column*480+(480-im.shape[1])//2;canvas[40:40+im.shape[0],x:x+im.shape[1]]=im
    cv2.putText(canvas,label,(column*480+12,25),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,20),1)
    status='No older raw sample at this timestamp' if raw is None else f"Left: {'observed' if raw.get('LLM') else 'MISSING'} / right: {'observed' if raw.get('RLM') else 'MISSING'}"
    cv2.putText(canvas,status,(column*480+12,365),cv2.FONT_HERSHEY_SIMPLEX,.42,(20,20,20),1)
   cv2.putText(canvas,g['gloss']+' | '+str(f['time_ms'])+' ms | estimates, accuracy unvalidated',(12,390),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
   process.stdin.write(canvas.tobytes())
   if i in [len(frames)//4,len(frames)//2,3*len(frames)//4]:cv2.imwrite(str(OUT/f'{digest}-{i}.png'),canvas)
  process.stdin.close();assert process.wait()==0
 source.release();hands={}
 for side,key in [('left','LLM'),('right','RLM')]:
  gaps=[];start=None
  for f in frames:
   if not f.get(key) and start is None:start=f['time_ms']
   if f.get(key) and start is not None:gaps.append(f['time_ms']-start);start=None
  if start is not None:gaps.append(frames[-1]['time_ms']-start+1000/motion['fps'])
  hands[side]={'coverage':sum(bool(f.get(key)) for f in frames)/len(frames),'longest_missing_ms':round(max(gaps,default=0)),'missing_with_raised_visible_wrist':sum(not f.get(key) and bool(f.get('PLM')) and f['PLM'][15 if side=='left' else 16].get('visibility',0)>=.75 and f['PLM'][13 if side=='left' else 14].get('visibility',0)>=.75 and 0<f['PLM'][15 if side=='left' else 16]['y']<1 and f['PLM'][15 if side=='left' else 16]['y']<f['PLM'][13 if side=='left' else 14]['y']-.01 for f in frames),'roi_review_frames':sum(f.get('hand_diagnostics',{}).get(side,{}).get('source')=='hands_roi' for f in frames)}
 comparison=next(c for c in r['comparisons'] if c['capture_id']==alias['capture_id'])
 return {'concept':g['concept_fr'],'gloss':g['gloss'],'source_id':g['source_id'],'source_sha256':digest,'frames':len(frames),'fps':motion['fps'],'hands':hands,'gate':comparison['gate'],'video':video.name,'source_url':alias['source_url'],'accuracy_validated':False}

for i,words in enumerate(sequences,1):
 selected=[];missing=[]
 for word in words:
  matches=[g for g in entries if g['concept_fr'].casefold()==word.casefold()]
  if len(matches)!=1:missing.append({'word':word,'matches':len(matches)});continue
  g=matches[0]
  if g['gloss_id'] not in review:review[g['gloss_id']]=render(g);print('Rendered',word,flush=True)
  selected.append(review[g['gloss_id']])
 results.append({'id':i,'phrase':phrases[i-1],'requested_concepts':words,'missing_or_ambiguous':missing,'words':selected})
 if not missing:
  concat=OUT/f'phrase-{i:02}.txt';concat.write_text(''.join("file '"+x['video']+"'\n" for x in selected))
  subprocess.run(['ffmpeg','-v','error','-y','-f','concat','-safe','0','-i',str(concat),'-c','copy','-movflags','+faststart',str(OUT/f'phrase-{i:02}.mp4')],check=True)
page='<meta charset="utf-8"><style>body{font:16px system-ui;max-width:1100px;margin:auto;padding:20px}video{width:100%}article{border-bottom:1px solid #aaa;padding:20px}small{display:block}</style><h1>15 dictionary sequences — source CV review</h1><p>Source recordings with raw detector overlays. No avatar, sign interpolation or filled missing hands. Left green, right cyan; body yellow, selected face points pink. Detector presence is not ground truth.</p>'
for r in results:
 page+=f'<article><h2>{r["id"]}. {html.escape(r["phrase"])}</h2><p>{html.escape(" · ".join(r["requested_concepts"]))}</p>'
 if not r['missing_or_ambiguous']:page+=f'<video controls preload="none" src="phrase-{r["id"]:02}.mp4"></video>'
 for w in r['words']:page+=f'<p>{html.escape(w["concept"])}: L {w["hands"]["left"]["coverage"]:.0%}, R {w["hands"]["right"]["coverage"]:.0%} — {w["gate"]}</p>'
 page+='</article>'
page+='<script>document.querySelectorAll("video").forEach(v=>v.playbackRate=.5)</script>';(OUT/'index.html').write_text(page);(OUT/'results.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n');print('DONE',len(results),'phrases',len(review),'unique clips',flush=True)
