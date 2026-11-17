"""Audit every registered source motion and create honest per-sign skeleton previews.

Engineering flags are review candidates, not linguistic judgements or approval.
The original files/coordinates are never deleted or rewritten by this audit.
"""
import hashlib,html,json,math,statistics
from pathlib import Path
from collections import Counter
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.services.sign_core import core_window
BODY=[('head','neck'),('neck','left_shoulder'),('neck','right_shoulder'),('left_shoulder','left_elbow'),('left_elbow','left_wrist'),('right_shoulder','right_elbow'),('right_elbow','right_wrist')]
FINGERS=['thumb','index','middle','ring','pinky']


def audit(motion):
    frames=motion['frames'];events=[]
    for side in ['left','right']:
        for i in range(1,len(frames)-1):
            a,b,c=frames[i-1:i+2];dt=c['time_ms']-a['time_ms']
            if not 0<dt<=180:continue
            for name,threshold in [(side+'_wrist',.075),(side+'_elbow',.09)]:
                if not all(name in f['joints'] for f in [a,b,c]):continue
                x,y,z=[f['joints'][name] for f in [a,b,c]]
                if math.dist(x,z)<.03 and min(math.dist(y,x),math.dist(y,z))>threshold:
                    events.append({'time_ms':b['time_ms'],'side':side,'kind':'isolated_arm_reversal','joint':name})
            bad=[]
            for finger in FINGERS:
                for n in range(1,4):
                    keys=[f'{side}_{finger}_{n}',f'{side}_{finger}_{n+1}']
                    if not all(all(k in f['joints'] for k in keys) for f in [a,b,c]):continue
                    lengths=[math.dist(f['joints'][keys[0]],f['joints'][keys[1]]) for f in [a,b,c]]
                    reference=(lengths[0]+lengths[2])/2
                    if reference>.005 and abs(lengths[0]-lengths[2])<.004 and abs(lengths[1]-reference)>.012 and abs(lengths[1]-reference)>reference*.65:bad.append(keys[0])
            if len(bad)>=3:events.append({'time_ms':b['time_ms'],'side':side,'kind':'isolated_finger_length_burst','segments':bad})
    return events


def preview(motion,start,end):
    fs=[f for f in motion['frames'] if start<=f['time_ms']<=end] or motion['frames']
    selected=[fs[round((len(fs)-1)*u)] for u in [0,.5,1]]
    svg=['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 720 280"><rect width="720" height="280" fill="#f9f4ed"/>']
    for col,f in enumerate(selected):
        js=f['joints'];neck=js.get('neck',[0,1.5,0]);x0=120+col*240
        def point(p):return x0+(p[0]-neck[0])*180,115-(p[1]-neck[1])*180
        edges=[(a,b,'#83644f') for a,b in BODY]
        for side,color in [('left','#27815a'),('right','#bd7418')]:
            for finger in FINGERS:
                keys=[side+'_wrist']+[f'{side}_{finger}_{n}' for n in range(1,5)]
                edges += [(a,b,color) for a,b in zip(keys,keys[1:])]
        for a,b,color in edges:
            if a not in js or b not in js:continue
            x,y=point(js[a]);xx,yy=point(js[b]);svg.append(f'<line x1="{x:.2f}" y1="{y:.2f}" x2="{xx:.2f}" y2="{yy:.2f}" stroke="{color}" stroke-width="2"/>')
        svg.append(f'<text x="{col*240+15}" y="260" font-size="14">{f["time_ms"]} ms · source pose</text>')
    return ''.join(svg)+'</svg>'


def main():
    rows=json.loads((ROOT/'.runtime/library.json').read_text());out=ROOT/'.runtime/motion-quality';gallery=ROOT/'site/media/library-review';gallery.mkdir(parents=True,exist_ok=True);out.mkdir(exist_ok=True)
    report=[];counts=Counter()
    for row in rows:
        p=Path(row['motion']['path']);p=p if p.is_absolute() else ROOT/p
        motion=json.loads(p.read_text());events=audit(motion);start,end=core_window(motion);identifier=row['motion']['motion_id']
        (gallery/(identifier+'.svg')).write_text(preview(motion,start,end))
        item={'id':identifier,'concept':row['gloss']['concept_fr'],'source_id':row.get('source_id'),'source_sha256':motion.get('cv_source_sha256'),'native_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'frames':len(motion['frames']),'duration_ms':motion['duration_ms'],'core_start_ms':start,'core_end_ms':end,'events':events,'source_video':'/api/captures/'+row['capture_id']+'/source','source_url':row.get('attribution',{}).get('source_url'),'review':'pending','accuracy_validated':False};report.append(item)
        counts.update(e['kind'] for e in events)
    value={'clips_audited':len(report),'clips_with_flags':sum(bool(r['events']) for r in report),'events':dict(counts),'annotations':'Source-coordinate engineering triage only. Fists and repeated movements are not automatically wrong. No captures removed or approved.','records':report}
    (out/'audit.json').write_text(json.dumps(value,ensure_ascii=False,indent=2));(gallery/'audit.json').write_text(json.dumps(value,ensure_ascii=False))
    page=['<meta charset="utf-8"><title>Library motion review</title><style>body{font:16px system-ui;max-width:1000px;margin:auto;background:#faf6ef}article{padding:12px;border-bottom:1px solid #cdbcae}img{width:100%}video{max-width:500px}</style><h1>All source signs — engineering review</h1><p>Actual captured poses at core start/middle/end. Hidden fingers stay absent. Flags require review; this is not linguistic approval.</p>']
    for r in sorted(report,key=lambda r:(-len(r['events']),r['concept'])):
        page.append(f'<article><h2>{html.escape(r["concept"])} · {len(r["events"])} flags</h2><img loading="lazy" src="{r["id"]}.svg"><details><summary>Original source video</summary><video controls preload="none" src="{r["source_video"]}"></video></details></article>')
    (gallery/'index.html').write_text('\n'.join(page));print(json.dumps({k:v for k,v in value.items() if k!='records'}))


if __name__=='__main__':main()
