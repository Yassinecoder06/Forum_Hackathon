"""Render exact-timestamp source overlays for held CV regressions; no approvals."""
import gzip
import html
import json
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/cv-review/corpus-holds'
OUT.mkdir(parents=True, exist_ok=True)
report = json.loads((ROOT / '.runtime/cv-corpus-activation/applied.json').read_text())
results = json.loads((ROOT / '.runtime/cv-v4-all/results.json').read_text())['results']
by_capture = {a['capture_id']:(r,a) for r in results for a in r.get('aliases',[])}
page = ['<meta charset="utf-8"><title>Held CV source review</title><style>body{font:16px system-ui;max-width:1100px;margin:auto}img{max-width:49%}article{padding:15px;border-bottom:1px solid #aaa}</style><h1>Held CV regressions</h1><p>Previous extraction remains active. Same source frame: previous raw points on left, candidate on right. Green/cyan = estimated left/right hands. Missing observations are not filled. These images require independent human review; no correctness or approval is inferred.</p>']
count = images = 0
for entry in report['entries']:
    if not entry['decision'].startswith('hold_'):
        continue
    result, alias = by_capture[entry['capture_id']]
    old = {f['time_ms']:f for f in json.load(gzip.open(alias['baseline_raw'],'rt'))}
    new = {f['time_ms']:f for f in json.load(gzip.open(ROOT / '.runtime/cv-v4-all/captures' / result['source_sha256'] / 'animics-landmarks.json.gz','rt'))}
    common = sorted(old.keys() & new.keys())
    lost = [t for t in common if any(old[t].get(k) and not new[t].get(k) for k in ['PLM','LLM','RLM'])]
    selected = lost[:2] or common[len(common)//2:len(common)//2+1]
    page.append(f'<article><h2>{html.escape(str(entry["source_id"]))}</h2><p>{html.escape(entry["decision"])} · {len(lost)} shared timestamps with lost observations</p>')
    cap = cv2.VideoCapture(alias['source'])
    for t in selected:
        cap.set(cv2.CAP_PROP_POS_MSEC,t);ok,image=cap.read()
        if not ok:
            raise ValueError('Cannot decode held source '+alias['source'])
        for label,frame in [('before',old[t]),('candidate',new[t])]:
            im=image.copy();h,w=im.shape[:2]
            for key,color in [('LLM',(0,220,0)),('RLM',(255,180,0))]:
                points=frame.get(key)
                if points:
                    for i,p in enumerate(points):
                        q=(round(p['x']*w),round(p['y']*h));cv2.circle(im,q,2,color,-1)
                        if i:
                            parent=points[0 if i in [1,5,9,13,17] else i-1]
                            cv2.line(im,q,(round(parent['x']*w),round(parent['y']*h)),color,1)
            cv2.putText(im,f'{label} {t} ms',(5,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(20,20,240),1)
            name=f'{entry["capture_id"]}-{t}-{label}.jpg'
            assert cv2.imwrite(str(OUT/name),im)
            page.append(f'<img loading="lazy" alt="{label} {t} ms" src="{name}">');images+=1
    cap.release();page.append('</article>');count+=1
(OUT/'index.html').write_text('\n'.join(page))
print(json.dumps({'held_clips':count,'source_overlay_images':images,'accuracy_validated':False,'gallery':str(OUT/'index.html')}))
