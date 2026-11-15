#!/usr/bin/env python3
"""Generate editable vector pipeline diagrams and print-ready HTML (stdlib only)."""
from pathlib import Path
from html import escape
import textwrap
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'docs/pipeline';OUT.mkdir(parents=True,exist_ok=True)
W,H=1680,1188
C={'ink':'#193b34','muted':'#60746d','green':'#23684f','blue':'#336a91','amber':'#996522','purple':'#74578d'}
class Page:
 def __init__(self,n,title,subtitle):
  self.n=n; self.parts=[f'''<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}"><defs>'''+''.join(f'<marker id="{k}" markerWidth="10" markerHeight="10" refX="8" refY="5" orient="auto"><path d="M0,1 L8,5 L0,9" fill="none" stroke="{v}" stroke-width="1.5"/></marker>' for k,v in C.items())+'''</defs><rect width="1680" height="1188" fill="#f7f8f3"/>''']
  self.text(40,47,'SIGNACULTURE  /  SYSTEM MAP',17,'green',weight=700)
  self.text(40,99,title,39,weight=700)
  self.text(40,137,subtitle,20,'muted')
 def text(self,x,y,t,size=20,color='ink',weight=400):
  self.parts.append(f'<text x="{x}" y="{y}" fill="{C.get(color,color)}" font-size="{size}" font-weight="{weight}" font-family="Arial,DejaVu Sans,sans-serif">{escape(t)}</text>')
 def box(self,x,y,w,h,title,lines=(),color='green',tag=None):
  bg={'green':'#edf3e9','blue':'#edf3f7','amber':'#faf2e5','purple':'#f2eef7'}[color]
  self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="17" fill="{bg}" stroke="{C[color]}" stroke-width="1.4"/>')
  for font in [18,17,16,15,14]:
   title_font=23 if font>=17 else 21
   tl=textwrap.wrap(title,width=int((w-40)/(title_font*.53)))
   body=[sub for line in lines for sub in textwrap.wrap(line,width=int((w-40)/(font*.49)))]
   title_y=y+(55 if tag else 32)
   body_y=title_y+(len(tl)-1)*27+31
   step=font+5
   last=body_y+max(0,len(body)-1)*step
   if last+5<=y+h-12:break
  else:raise ValueError('Box cannot fit: '+title)
  if tag:self.text(x+20,y+24,tag.upper(),12,color,700)
  for i,line in enumerate(tl):self.text(x+20,title_y+27*i,line,title_font,color,700)
  for i,line in enumerate(body):self.text(x+20,body_y+step*i,line,font,'ink')
 def arrow(self,points,color='green',dashed=False,label=None,lx=None,ly=None):
  d='M'+' L'.join(f'{x},{y}' for x,y in points)
  self.parts.append(f'<path d="{d}" fill="none" stroke="{C[color]}" stroke-width="2.4" stroke-linejoin="round" marker-end="url(#{color})"'+(' stroke-dasharray="9 7"' if dashed else '')+'/>')
  if label:self.text(lx,ly,label,16,color,700)
 def note(self,x,y,text,width=100,color='muted',size=18):
  for i,line in enumerate(textwrap.wrap(text,width=width)):self.text(x,y+25*i,line,size,color)
 def finish(self,name):
  self.parts.append('<path d="M40 1134 H1640" stroke="#ced8cc"/>')
  self.text(40,1164,'Implementation snapshot · 15 November 2026 · Local research prototype · Human LSF review pending',16,'muted')
  self.text(1490,1164,f'{self.n} / 5',18,'green',700)
  self.parts.append('</svg>');path=OUT/name;path.write_text('\n'.join(self.parts));return path

p=Page(1,'From French input to a 3D signing preview','Current application flow. Solid arrows = default path; dashed arrows = optional assistance or audio controls.')
p.box(40,180,590,160,'What the application produces',[
 'An attributed motion preview and a downloadable WebM.',
 'Concept selection is not a validated French-to-LSF translation.',
 'Audio affects timing / expressivity; it is not transcribed into text.' ],tag='Scope')
p.box(700,180,260,170,'Real sign library',['1,447 real motion entries','1,007 Dicta-Sign captures','Source pipeline: pages 2 + 5'],tag='Shared resource')
p.arrow([(700,265),(665,265),(665,390),(485,390),(485,430)],label='Known concepts',lx=490,ly=380)
p.arrow([(830,350),(830,430)],label='Verified IDs',lx=845,ly=395)
p.arrow([(960,265),(1160,265),(1160,430)],label='Motion assets',lx=1030,ly=253)
p.box(40,430,250,185,'French text',['Typed or pasted in Studio','No speech-to-text stage'],tag='Input')
p.box(340,430,290,185,'Concept lookup',['spaCy FR lemmatization','Longest phrase matching','Unknown words stay visible'],tag='Default')
p.box(700,430,260,185,'Concept sequence',['Known gloss / concept IDs','No LSF grammar guarantee','Unrepresented meaning shown'],tag='Intermediate')
p.box(1030,430,260,185,'Motion composition',['Resolve captured clips','333 ms quintic bridges','Plan timestamps + credits'],tag='Animation plan')
p.box(1360,430,280,185,'3D avatar + preview',['Rigged mesh + arm / hand IK','Fingers + captured face','Timeline / seek / speed'],tag='Player')
for a,b in [(290,340),(630,700),(960,1030),(1290,1360)]:p.arrow([(a,520),(b,520)])
p.box(340,705,290,165,'Dictionary search',['Existing French concepts','User chooses valid IDs','No language-model inference'],color='blue',tag='Manual selection')
p.box(700,705,260,165,'Editable draft',['User adds / removes signs','User changes their order','Review-sheet JSON export'],color='blue',tag='Human control')
p.arrow([(165,615),(165,665),(485,665),(485,705)],'blue',True)
p.arrow([(630,785),(700,785)],'blue',True)
p.arrow([(830,705),(830,615)],'blue',True,label='Review & preview',lx=845,ly=675)
p.box(1030,705,260,165,'Audio controls',['Voiced-region timing','Hold through audio pauses','Optional facial modulation'],color='blue',tag='Optional')
p.arrow([(1160,705),(1160,615)],'blue',True)
p.arrow([(1290,785),(1325,785),(1325,660),(1500,660),(1500,615)],'blue',True)
p.box(1360,705,280,165,'Preview / publication',['WebM: browser recording','Watermark + source credits','Publication API stays gated'],color='amber',tag='Output & review')
p.arrow([(1640,520),(1660,520),(1660,680),(1500,680),(1500,705)],label=None)
p.box(40,930,250,170,'Audio or video',['Optional uploaded media','FFmpeg → mono 16 kHz PCM'],color='blue',tag='Audio input')
p.box(340,930,620,170,'Acoustic analysis + emotion hypothesis',['DSP: estimated pitch (F0), RMS energy and pauses.', 'Local emotion2vec+ seed: experimental emotion scores.', 'Captured facial signals take priority over audio expression.'],color='blue',tag='Separate from text understanding')
p.arrow([(290,1010),(340,1010)],'blue',True)
p.arrow([(960,1010),(995,1010),(995,835),(1030,835)],'blue',True)
p.box(1030,930,610,170,'T5 / mT5: research branch only',['Trained and evaluated on source-annotated utterances.', 'Available in Research; never automatically drives playback.', 'Data, experiments and measured limitations: page 3.'],color='purple',tag='Outside the normal animation path')
files=[p.finish('01-application.svg')]

p=Page(2,'How video becomes a reusable motion','Lexical video ingestion and sentence annotations are separate data streams.')
p.box(40,190,340,180,'Public lexical videos',['1,025 Dicta-Sign LSF videos','Earlier Commons: 447 source concepts','Source rights recorded separately'],tag='Public corpus')
p.box(460,190,340,180,'Source verification',['Rights / licence checks','Source URL + provenance','Checksums verify retrieved files'],tag='Acquisition')
p.box(880,190,340,180,'MediaPipe extraction',['Holistic body + face','Hand detector fallback','Video sampling ≤ 20 fps'],tag='Computer vision')
p.box(1300,190,340,180,'Technical quality gate',['Pose coverage ≥ 80%','At least one hand ≥ 35%','Threshold is not LSF validation'],color='amber',tag='Automatic screening')
for a,b in [(380,460),(800,880),(1220,1300)]:p.arrow([(a,280),(b,280)])
p.box(40,490,340,175,'Authorized local input',['Uploaded signer video','Native motion JSON','Animics landmark JSON'],color='blue',tag='Alternative input')
p.box(460,490,340,175,'Motion normalization',['signaculture-v1 format','Body + sparse finger joints','Mouth / brow measurements'],tag='Common representation')
p.arrow([(380,570),(460,570)],'blue',True,label='JSON',lx=389,ly=555)
p.arrow([(210,490),(210,400),(860,400),(860,280),(880,280)],'blue',True,label='Local video uses the same CV extraction',lx=230,ly=390)
p.arrow([(1050,370),(1050,425),(630,425),(630,490)])
p.box(880,490,340,175,'Tracking review',['Dicta: 1,007 accepted, 18 flagged','Commons: 439 imports, 7 flagged','Review hands against source video'],color='amber',tag='Observed extraction outcome')
p.arrow([(1470,370),(1470,445),(1050,445),(1050,490)],'amber')
p.box(1300,490,340,175,'Registered sign library',['1,007 Dicta + previous 440 entries','1,447 real motion entries','Canonical gloss + source + rights'],tag='Playback data')
p.arrow([(800,590),(840,590),(840,720),(1470,720),(1470,665)])
p.arrow([(1220,565),(1300,565)])
p.box(40,795,760,220,'What the capture preserves',['Hip-relative body coordinates; wrist-anchored finger trajectories.', 'Sparse tracking and face measurements are kept with coverage statistics.', 'Occluded or untracked fingers are not fabricated.', 'Depth and avatar retargeting remain approximate; review is required.'],tag='Motion characteristics')
p.box(880,795,760,220,'Traceability and review',['Local captures, original files and derived motions: .runtime/', 'Bundled attributed Bonjour example: data/public_examples/', 'Inventory and ingestion report: data/source_inventory/', 'Pending linguistic / animation review remains separate from rights clearance.'],tag='Persistence')
p.note(40,1080,'Boundary: isolated-sign videos are lexical motion data, not sentence-level French–gloss training pairs.',120,'amber',21)
files.append(p.finish('02-motion-data.svg'))

p=Page(3,'Models, evaluation and paper coverage','T5/mT5 and the new within-sign CGAN are research-only. The supplied paper is compared on page 4.')
p.box(40,185,340,180,'UD / SUD STKAutogramm',['Published gloss annotations','French free translations','Versions + annotation lineage'],color='purple',tag='Sentence research data')
p.box(460,185,340,180,'Audit and derive',['Deduplicate overlap','Preserve source annotations','0 local linguistic approvals'],color='purple',tag='Catalog / eligibility')
p.box(880,185,340,180,'Frozen dataset splits',['128 research pairs (v002)','61 train / 28 validation / 39 test','Grouped by recordings'],color='purple',tag='Leakage controls')
p.box(1300,185,340,180,'Train / select / evaluate',['LoRA on small seq2seq models','mT5: RTX 4060, BF16','Select with validation loss'],color='purple',tag='Experiment')
for a,b in [(380,460),(800,880),(1220,1300)]:p.arrow([(a,275),(b,275)],'purple')
p.box(40,455,760,220,'Measured translation results',['T5-small CPU (v001; 39 train): exact 0%; WER 207.28%; BLEU 0.', 'mT5-small, 8 epochs: exact match 0%; WER 100%; BLEU 0.', 'Longer mT5: exact match 0%; WER 517.96%; BLEU 0.042.', 'Expanded-data retrieval baseline: WER 120.87%; BLEU 0.301.'],color='purple',tag='39-example reused engineering test')
p.box(880,455,760,220,'Research inference only',['Reports, checkpoints and raw predictions are retained.', 'Research tab can query the failed T5 / mT5 adapters.', 'Their generated glosses do not automatically select avatar motions.', 'A five-point CGAN is now trained for within-sign tests only.'],color='amber',tag='Integration decision')
p.arrow([(1470,365),(1470,410),(420,410),(420,455)],'purple')
p.arrow([(800,565),(880,565)],'purple')
p.box(40,765,500,250,'Retired concept-planner experiment',['Archived local-model trial; removed from application.', 'Fixed engineering check: 24 French inputs.', '13/24 exact concept sequences vs 12/24 literal lookup.', 'Mean latency 7.8 s; GPU peak ≈ 5.2 GB.'],color='blue',tag='Separate optional experiment')
p.box(590,765,500,250,'Correction and limitations',['Name detection falsely removed valid concepts.', 'The runtime was repaired after evaluation.', 'Original scores stay frozen; repaired runtime is unscored.', 'NER is advisory; editable human review is always required.'],color='blue',tag='Measured version ≠ current runtime')
p.box(1140,765,500,250,'What is still needed',['Qualified LSF sentence-level review.', 'Larger, aligned France-LSF phrase data.', 'Validation of nonmanuals and intelligibility.', 'Institutionally restricted datasets remain metadata-only.'],color='amber',tag='Toward reliable translation')
p.note(40,1071,'Repository evidence: research/IMPLEMENTATION_STATUS.md · research/CONCEPT_PLANNER_REPORT.md',145,size=17)
p.note(40,1097,'Verified paper comparison: research/PAPER_B_VERIFIED_REVIEW.md · Paper results are not reproduced here.',145,size=17)
files.append(p.finish('03-models-research.svg'))
p=Page(4,'What the supplied T5–GAN paper actually does','Source: research_paper.pdf · Muanji et al., Procedia Computer Science 290 (2026), pp. 578–584.')
p.box(40,190,470,180,'Voice or typed French',['Voice → Google ASR API.', 'Typed French bypasses ASR.', 'This stage is different from emotion analysis.'],color='purple',tag='1 · Input / speech recognition')
p.box(605,190,470,180,'Fine-tuned T5-base',['220M parameters; 319 added gloss tokens.', '2,500 French–gloss training corpus pairs.', 'AdamW; 20 epochs; LR 2e-5; batch 4.'],color='purple',tag='2 · Neural translation')
p.box(1170,190,470,180,'Hybrid post-correction',['47 expert canonicalization rules.', 'RapidFuzz matching, threshold 85%.', 'Rule-based reordering; validate glosses.'],color='purple',tag='3 · Canonical gloss sequence')
p.arrow([(510,280),(605,280)],'purple')
p.arrow([(1075,280),(1170,280)],'purple')
p.box(1170,485,470,195,'Retrieve sign landmarks',['319 isolated signs; MediaPipe Holistic.', 'Landmark sequences retrieved from CSV.', 'Adjacent sign endpoints condition transitions.'],color='purple',tag='4 · Gloss-to-motion lookup')
p.arrow([(1405,370),(1405,485)],'purple')
p.box(605,485,470,195,'Conditional GAN transitions',['MLP generator + discriminator.', '5 selected landmarks → 10 features.', '10 frames at 30 fps; adversarial + L1, λ=1000.'],color='purple',tag='5 · Co-articulation between signs')
p.box(40,485,470,195,'Pygame animation',['Stylized 2D avatar.', 'Full 3D and richer hands are future work.', 'No implemented emotion2vec branch in this paper.'],color='purple',tag='6 · Rendering')
p.arrow([(1170,580),(1075,580)],'purple')
p.arrow([(605,580),(510,580)],'purple')
p.box(40,795,760,235,'What SignaCulture already has',['Related MediaPipe extraction and a real motion library.', 'Controlled lookup and manually edited concept sequences.', '3D avatar, video preview export, audio DSP and emotion2vec.', 'T5/mT5 and a five-point CGAN experiment; different data from paper.'],tag='Implemented, with important differences')
p.box(880,795,760,235,'Paper modules still missing',['No matching T5-base + 319-token reproduction or 47-rule correction.', 'CGAN: within-sign tests only; no validated inter-sign model.', 'No Google ASR path and no comparable interpreter evaluation.', 'Original corpus / code: paper says available upon request.'],color='amber',tag='Not a full reproduction')
p.note(40,1071,'Paper-reported data: 379 real + 1,000 synthetic transition pairs. These are not our extracted isolated clips.',145,size=17)
p.note(40,1097,'Read all 7 supplied pages. Exact comparison and reproducibility caveats: research/PAPER_B_VERIFIED_REVIEW.md',145,size=17)
files.append(p.finish('04-paper-architecture.svg'))

p=Page(5,'Completed Dicta-Sign → gloss → pose baseline','All 1,025 public LSF clips downloaded and captured. Actual upstream code is connected to the Studio.')
p.box(40,190,340,190,'French lexical index',['1,046 concept pages inspected','1,025 public France-LSF MP4s','21 pages have no LSF video'],tag='Hamburg Dicta-Sign portal')
p.box(460,190,340,190,'Cache + provenance',['Original MP4 + SHA256 verified','Actual canonical source gloss','French concept label + HamNoSys'],tag='No invented sentence pairs')
p.box(880,190,340,190,'Computer vision',['MediaPipe Holistic + Hands','Animics-compatible raw channels','33 body + 468 face + 21 per hand'],tag='Observed landmarks')
p.box(1300,190,340,190,'Tracking screening',['1,007 accepted into lookup','18 retained for tracking review','Missing arrays stay missing'],color='amber',tag='Not linguistic approval')
for a,b in [(380,460),(800,880),(1220,1300)]:p.arrow([(a,285),(b,285)])
p.box(40,505,340,190,'French text / glosses',['Example: Bonjour merci','French lemmatization','Longest expression lookup'],tag='Studio input')
p.box(460,505,340,190,'Resolve real source clips',['BONJOUR → Dicta cs380','MERCI → Dicta cs905','Unknown / ambiguity blocks output'],tag='France LSF only')
p.box(880,505,340,190,'Actual upstream pipeline',['PoseLookup.lookup_sequence','normalize_pose + join_poses','Masks preserved at 25 fps'],tag='Pinned spoken-to-signed code')
p.box(1300,505,340,190,'Downloadable outputs',['543-point standard .pose','512×512 MP4 + GIF skeleton','80 frames / 25 fps / 3.2 s example'],tag='Diagnostic renderer')
for a,b in [(380,460),(800,880),(1220,1300)]:p.arrow([(a,600),(b,600)])
p.arrow([(1470,380),(1470,440),(630,440),(630,505)])
p.box(460,815,760,195,'Existing rigged 3D player',['Uses the same registered Dicta-Sign motion clips.', 'Separate arm / hand IK and 333 ms transition bridges.', 'Browser verified: playback, seeking and actual WebM download.'],color='blue',tag='Alternative rendering branch')
p.arrow([(630,695),(630,815)],'blue',True)
p.box(1300,815,340,195,'Scope and rights',['Lexical prototype; no LSF grammar claim','T5 / GAN training not required','Local previews; redistribution pending'],color='amber',tag='Explicit limitations')
p.note(40,1080,'Evidence: docs/DICTASIGN_PIPELINE.md · data/source_inventory/dictasign_ingestion_report.json',145,size=18)
files.append(p.finish('05-dictasign-upstream.svg'))

html='''<!doctype html><html lang="en"><meta charset="utf-8"><title>SignaCulture — Full Pipeline</title><style>@page{size:A3 landscape;margin:0}*{box-sizing:border-box}html,body{margin:0;background:#f7f8f3}.page{width:420mm;height:297mm;break-after:page;overflow:hidden}.page:last-child{break-after:auto}.page svg{display:block;width:100%;height:100%}</style>'''+''.join('<section class="page">'+f.read_text()+'</section>' for f in files)+'</html>'
(OUT/'pipeline.html').write_text(html)
print('Generated five SVG pages and docs/pipeline/pipeline.html')
