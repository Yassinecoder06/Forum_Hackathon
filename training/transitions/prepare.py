"""Split source clips FIRST, then mask complete native-rate12frame windows."""
import hashlib,json,unicodedata,urllib.parse
from collections import Counter
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
from training.transitions.common import ROOT,MAIN,ARTIFACTS,JOINTS,FEATURES,CLAIM,sha,dump
SEED=42

def source_key(url):
    s=urllib.parse.unquote(url).replace('_',' ')
    return unicodedata.normalize('NFKC',s).casefold().split('?')[0]
def main():
    ARTIFACTS.mkdir(exist_ok=True)
    manifest_path=ROOT/'research/TRANSITION_DATA_MANIFEST.json'
    frozen=json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    library=MAIN/'.runtime/library.json';lib=json.loads(library.read_text());records=[]
    for r in lib:
        m=r['motion']
        if m.get('synthetic') or m.get('rights_status')!='cleared':continue
        records.append(dict(clip_id=m['motion_id'],concept=r['gloss']['concept_fr'],path=m['path'],source_url=r.get('attribution',{}).get('source_url',r['source']),source_id=r.get('source_id'),license=m['license'],credit=r.get('attribution',{}).get('credit'),linguistic_review='PENDING'))
    mg={r['motion_id']:r for r in json.loads((MAIN/'data/motions_manifest.json').read_text())}
    for g in json.loads((MAIN/'data/glossary.json').read_text()):
        m=mg.get(g['motion_id'],{})
        if g['concept_fr']=='bonjour' and m.get('synthetic') is False and m.get('rights_status')=='cleared':
            handoff=json.loads((MAIN/'research/VIDEO_HANDOFF.json').read_text())
            records.append(dict(clip_id=m['motion_id'],concept='bonjour',path=str(MAIN/m['path']),source_url=handoff['source_url'],source_id=g['source_id'],license=m['license'],credit=handoff['attribution'],linguistic_review='PENDING'))
    assert len(records)==440
    loaded={};duplicates=[];source_seen={};content_seen={};unique=[]
    for r in sorted(records,key=lambda x:x['clip_id']):
        obj=json.loads(Path(r['path']).read_text());frames=obj['frames'];r['input_sha256']=sha(r['path']);r['fps']=obj['fps'];r['frame_count']=len(frames)
        key=source_key(r['source_url'])
        # Geometry+timestamps hash catches duplicated captures even when metadata differs.
        reduced=[dict(time_ms=f['time_ms'],points={k:f['joints'].get(k) for k in JOINTS}) for f in frames]
        content=hashlib.sha256(json.dumps(reduced,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        duplicate=source_seen.get(key) or content_seen.get(content)
        if duplicate:duplicates.append(dict(clip_id=r['clip_id'],same_source_as=duplicate));continue
        source_seen[key]=r['clip_id'];content_seen[content]=r['clip_id'];r['group_id']='source:'+hashlib.sha256(key.encode()).hexdigest()[:20]
        r['geometry_sha256']=content;unique.append(r);loaded[r['clip_id']]=obj
    # Group assignment uses source identity only BEFORE inspecting/constructing windows.
    ordered=sorted(unique,key=lambda r:hashlib.sha256((str(SEED)+r['group_id']).encode()).hexdigest())
    n=len(ordered);ntrain=int(.70*n);nval=int(.15*n)
    for i,r in enumerate(ordered):r['split']='train' if i<ntrain else 'val' if i<ntrain+nval else 'test'
    results={s:[] for s in ['train','val','test']};metadata={s:[] for s in results};excluded=Counter()
    for r in unique:
        f=loaded[r['clip_id']]['frames'];r['windows']=0
        for i in range(0,len(f)-11,3):
            w=f[i:i+12]
            if any(any(k not in a['joints'] or len(a['joints'][k])<2 for k in JOINTS) for a in w):excluded['missing_five_points']+=1;continue
            x=np.array([[v for k in JOINTS for v in a['joints'][k][:2]] for a in w],dtype=np.float32)
            if not np.isfinite(x).all():excluded['nonfinite']+=1;continue
            times=np.array([a['time_ms'] for a in w],dtype=np.float64);dt=np.diff(times)
            if np.any(dt<=0) or np.max(np.abs(dt-dt.mean()))>1.01:excluded['irregular_native_timing']+=1;continue
            results[r['split']].append(x);r['windows']+=1
            metadata[r['split']].append(dict(window_id=r['clip_id']+':'+str(i),clip_id=r['clip_id'],group_id=r['group_id'],source_url=r['source_url'],source_id=r['source_id'],concept=r['concept'],input_sha256=r['input_sha256'],first_frame=i,times_ms=times.tolist(),interior_duration_ms=float(times[-2]-times[1]),anchor_duration_ms=float(times[-1]-times[0]),license=r['license'],credit=r['credit']))
    arrays={s:np.stack(a) for s,a in results.items()}
    flat=arrays['train'].reshape(-1,10).astype(np.float64);mean=flat.mean(axis=0);std=np.maximum(flat.std(axis=0),1e-4)
    norm=dict(mean=mean.tolist(),std=std.tolist(),fit_split='train only',fit_samples=int(flat.shape[0]),definition='Per-feature z-score on complete training windows, including supplied anchors; each method uses identical stats',feature_order=FEATURES)
    dump(ARTIFACTS/'normalization.json',norm)
    for s,x in arrays.items():np.savez_compressed(ARTIFACTS/(s+'.npz'),windows=x)
    data=dict(kind=CLAIM,seed=SEED,library_sha256=sha(library),feature_order=FEATURES,head_note='Captured head proxy (MediaPipe pose nose/head position), x/y in signaculture-v1 rig coordinates; z and other fingers omitted',window_policy='12consecutive native captured frames; endpoints at0,11; known middle10frames masked; stride3; no missing-point imputation or30fps resampling',grouping_policy='One canonical source-video group per unique isolated clip; source URL or full reduced geometry/timestamp duplicates removed before hash-seeded70/15/15source split; windows extracted only afterward',clips_before_dedup=len(records),unique_clips=len(unique),duplicates_removed=duplicates,source_clip_counts=dict(Counter(r['split'] for r in unique)),window_counts={s:len(a) for s,a in arrays.items()},eligible_group_counts={s:len({r['group_id'] for r in metadata[s]}) for s in arrays},exclusions=dict(excluded),normalization=norm,files={s+'.npz':sha(ARTIFACTS/(s+'.npz')) for s in arrays},clips=unique,windows=metadata,limitations=['INTRA-SIGN reconstruction only, not genuine inter-sign coarticulation','No379sentence transitions or synthetic transition pairs from paper','Captures/retargeting are approximate and pending linguistic/animation review','Hand visibility selects the eligible subset','Same original signer may appear throughout; identity grouping unavailable; no signer-disjoint claim','Neighboring windows overlap within clips; clips are disjoint across splits','Native captured fps/timing differs from paper30fps; no paper-score comparison'])
    if frozen is not None:
        if data!=frozen:raise ValueError('Inputs or preprocessing differ from frozen manifest; do not overwrite')
    else:dump(manifest_path,data)
    print(json.dumps({k:data[k] for k in ['clips_before_dedup','unique_clips','source_clip_counts','window_counts','eligible_group_counts','exclusions']},indent=2))
if __name__=='__main__':main()
