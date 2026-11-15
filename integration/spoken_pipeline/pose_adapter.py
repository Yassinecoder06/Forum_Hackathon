"""Observed-landmark export for France LSF lexical poses. No synthetic fallback."""
import copy,csv,gzip,hashlib,json,math,re,time,unicodedata
from collections import Counter
from pathlib import Path
import numpy as np
import cv2
from pose_format import Pose
from pose_format.numpy import NumPyPoseBody
from pose_format.pose_header import PoseHeader,PoseHeaderComponent,PoseHeaderDimensions

UPSTREAM_REV='fec7ac393989f4b6541ab5f860819262a54f27c7'
BASE=Path(__file__).resolve().parent
DEFAULT_MAIN=BASE.parents[1]
BODY=['NOSE','LEFT_EYE_INNER','LEFT_EYE','LEFT_EYE_OUTER','RIGHT_EYE_INNER','RIGHT_EYE','RIGHT_EYE_OUTER','LEFT_EAR','RIGHT_EAR','MOUTH_LEFT','MOUTH_RIGHT','LEFT_SHOULDER','RIGHT_SHOULDER','LEFT_ELBOW','RIGHT_ELBOW','LEFT_WRIST','RIGHT_WRIST','LEFT_PINKY','RIGHT_PINKY','LEFT_INDEX','RIGHT_INDEX','LEFT_THUMB','RIGHT_THUMB','LEFT_HIP','RIGHT_HIP','LEFT_KNEE','RIGHT_KNEE','LEFT_ANKLE','RIGHT_ANKLE','LEFT_HEEL','RIGHT_HEEL','LEFT_FOOT_INDEX','RIGHT_FOOT_INDEX']
HAND=['WRIST','THUMB_CMC','THUMB_MCP','THUMB_IP','THUMB_TIP']+[f'{f}_{j}' for f in ['INDEX_FINGER','MIDDLE_FINGER','RING_FINGER','PINKY'] for j in ['MCP','PIP','DIP','TIP']]
BODY_LIMBS=[(0,1),(1,2),(2,3),(3,7),(0,4),(4,5),(5,6),(6,8),(9,10),(11,12),(11,13),(13,15),(12,14),(14,16),(11,23),(12,24),(23,24),(23,25),(24,26),(25,27),(26,28),(27,29),(28,30),(29,31),(30,32)]
HAND_LIMBS=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(0,17),(17,18),(18,19),(19,20)]
TOTAL=543

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
def fold(s):return unicodedata.normalize('NFKC',s).casefold().strip()
def make_header(width=640,height=640):
    c=lambda name,points,limbs,color:PoseHeaderComponent(name,points,limbs,[color],'XYZC')
    return PoseHeader(.2,PoseHeaderDimensions(width=width,height=height,depth=width),[
        c('POSE_LANDMARKS',BODY,BODY_LIMBS,(25,90,190)),
        c('FACE_LANDMARKS',[str(i) for i in range(468)],[],(130,130,130)),
        c('LEFT_HAND_LANDMARKS',HAND,HAND_LIMBS,(210,50,50)),
        c('RIGHT_HAND_LANDMARKS',HAND,HAND_LIMBS,(40,150,60))])
def mask_missing(pose):
    values=np.asarray(np.ma.getdata(pose.body.data),dtype=np.float32).copy();confidence=np.asarray(pose.body.confidence,dtype=np.float32).copy()
    valid=np.isfinite(values).all(-1)&np.isfinite(confidence)&(confidence>0)
    confidence=np.where(valid,np.clip(confidence,0,1),0);values[~valid]=0
    pose.body.data=np.ma.array(values,mask=np.repeat((~valid)[...,None],3,axis=-1));pose.body.confidence=confidence
    return pose

def _store(data,conf,frame,offset,points,scale):
    if points is None:return
    for j,p in enumerate(points[:scale[3]]):
        if p is None:continue
        vals=[p.get(k) for k in ['x','y','z']]
        if any(not isinstance(x,(int,float)) or not math.isfinite(x) for x in vals):continue
        confidence=p.get('visibility',1.)
        if not isinstance(confidence,(int,float)) or not math.isfinite(confidence) or confidence<=0:continue
        data[frame,0,offset+j]=[vals[0]*scale[0],vals[1]*scale[1],vals[2]*scale[2]]
        conf[frame,0,offset+j]=min(1.,confidence)

def raw_to_pose(path):
    frames=json.load(gzip.open(path,'rt',encoding='utf-8'))
    if not frames:raise ValueError('Empty raw Animics capture')
    source=Path(path).with_name('source.mp4')
    width=frames[0].get('image_width');height=frames[0].get('image_height')
    if not width or not height:
        video=cv2.VideoCapture(str(source))
        try:width=int(video.get(cv2.CAP_PROP_FRAME_WIDTH));height=int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        finally:video.release()
    if not width or not height:raise ValueError('Source image dimensions are required for undistorted landmarks')
    data=np.zeros((len(frames),1,TOTAL,3),np.float32);conf=np.zeros((len(frames),1,TOTAL),np.float32)
    times=np.array([r['time_ms'] for r in frames],float)
    if len(times)<2 or np.any(np.diff(times)<=0):raise ValueError('Invalid raw timestamps')
    dt=[r.get('dt') for r in frames]
    valid_dt=all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in dt[1:])
    fps=(len(frames)-1)/sum(dt[1:]) if valid_dt else (len(times)-1)*1000./(times[-1]-times[0])
    for i,r in enumerate(frames):
        _store(data,conf,i,0,r.get('PLM'),(width,height,width,33))
        _store(data,conf,i,33,r.get('FLM'),(width,height,width,468))
        _store(data,conf,i,501,r.get('LLM'),(width,height,width,21))
        _store(data,conf,i,522,r.get('RLM'),(width,height,width,21))
    pose=mask_missing(Pose(make_header(width,height),NumPyPoseBody(fps,data,conf)))
    return pose,dict(format='raw-animics-image-landmarks',source_sha256=sha(path),time_ms=times.tolist(),dt_seconds=dt,image_width=width,image_height=height,image_transform='Normalized x/z multiplied by source width; y by source height, preserving image aspect ratio; no image-axis flip',unused_channels='PWLM has different world axis/unit conventions; LWLM/RWLM unavailable, never fabricated',confidence='Original body visibility; presence1for supplied face/hands, absent0')

def motion_to_pose(path):
    source=json.loads(Path(path).read_text())
    if source.get('synthetic') is True:raise ValueError('Synthetic motion is not eligible')
    frames=source['frames'];data=np.zeros((len(frames),1,TOTAL,3),np.float32);conf=np.zeros((len(frames),1,TOTAL),np.float32)
    bodymap={'NOSE':'head','LEFT_SHOULDER':'left_shoulder','RIGHT_SHOULDER':'right_shoulder','LEFT_ELBOW':'left_elbow','RIGHT_ELBOW':'right_elbow','LEFT_WRIST':'left_wrist','RIGHT_WRIST':'right_wrist'}
    def point(i,k,values):
        if values is None or len(values)!=3 or not np.isfinite(values).all():return
        x,y,z=values;data[i,0,k]=[320+240*x,560-240*y,-240*z];conf[i,0,k]=1.
    for i,f in enumerate(frames):
        joints=f['joints']
        for name,key in bodymap.items():point(i,BODY.index(name),joints.get(key))
        for side,offset in [('left',501),('right',522)]:
            hand_present=any(f'{side}_{finger}_{j}' in joints for finger in ['thumb','index','middle','ring','pinky'] for j in range(1,5))
            if hand_present:point(i,offset,joints.get(side+'_wrist'))
            for finger_idx,finger in enumerate(['thumb','index','middle','ring','pinky']):
                for j in range(1,5):point(i,offset+1+finger_idx*4+j-1,joints.get(f'{side}_{finger}_{j}'))
    pose=mask_missing(Pose(make_header(),NumPyPoseBody(float(source['fps']),data,conf)))
    return pose,dict(format='stored-retargeted-motion',source_sha256=sha(path),time_ms=[f['time_ms'] for f in frames],image_transform='x_pixel=320+240x; y_pixel=560-240y; z_pixel=-240z (reversible affine)',confidence='Presence flags, not calibrated detector confidence; coordinates are stored captured/retargeted estimates',absent_channels='No face mesh or individual hips/legs/eyes/ears inferred from ratios/rig roots; all masked',hand_root='Observed body wrist reused as hand root only while actual finger joints exist')

def read_entries(main=DEFAULT_MAIN,allow_local_preview=False):
    main=Path(main);library_bytes=(main/'.runtime/library.json').read_bytes();library=json.loads(library_bytes);entries=[];blocked=[]
    for r in library:
        g,m=r['gloss'],r['motion']
        if m.get('synthetic'):continue
        pending=m.get('rights_status')!='cleared'
        if pending and not allow_local_preview:blocked.append(g['gloss_id']);continue
        if pending and not str(r.get('source_id','')).startswith('DICTASIGN_PORTAL_'):blocked.append(g['gloss_id']);continue
        entries.append(dict(id=g['gloss_id'],concept=g['concept_fr'],gloss=r.get('canonical_gloss',g['gloss']),aliases=g.get('aliases',[]),source_id=r.get('source_id',g['source_id']),motion_path=m['path'],capture_id=r.get('capture_id'),license=m.get('license'),rights_status=m.get('rights_status'),source_url=r.get('attribution',{}).get('source_url',r.get('source')),credit=r.get('attribution',{}).get('credit'),priority=0 if str(r.get('source_id','')).startswith('DICTASIGN_PORTAL_') else 10,local_preview_only=pending))
    motions={x['motion_id']:x for x in json.loads((main/'data/motions_manifest.json').read_text())}
    for g in json.loads((main/'data/glossary.json').read_text()):
        m=motions.get(g['motion_id'],{})
        if g['concept_fr']=='bonjour' and m.get('synthetic') is False and m.get('rights_status')=='cleared':
            path=main/m['path'];obj=json.loads(path.read_text());a=obj.get('attribution',{})
            entries.append(dict(id=g['gloss_id'],concept=g['concept_fr'],gloss=g['gloss'],aliases=g.get('aliases',[]),source_id=g['source_id'],motion_path=str(path),capture_id=None,license=m['license'],rights_status='cleared',source_url=a.get('source_url'),credit=a.get('credit'),priority=10,local_preview_only=False))
    return entries,blocked,hashlib.sha256(library_bytes).hexdigest()

def manifest_fingerprint(manifest):
    payload={k:manifest[k] for k in ['upstream_revision','language','spoken_language','allow_local_preview','source_library_sha256','records','blocked_pending_ids','export_errors']}
    return hashlib.sha256(json.dumps(payload,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def export_lexicon(out,main=DEFAULT_MAIN,allow_local_preview=False):
    start=time.monotonic();out=Path(out);(out/'poses').mkdir(parents=True,exist_ok=True)
    entries,blocked,library_sha=read_entries(main,allow_local_preview);rows=[];records=[];errors=[]
    for e in entries:
        try:
            raw=Path(main)/'.runtime/captures'/str(e['capture_id'])/'animics-landmarks.json.gz'
            pose,origin=raw_to_pose(raw) if e['capture_id'] and raw.exists() else motion_to_pose(e['motion_path'])
            filename='poses/'+e['id']+'.pose'
            with (out/filename).open('wb') as f:pose.write(f)
            record=dict(**e,pose_path=filename,frames=len(pose.body.data),fps=float(pose.body.fps),pose_sha256=sha(out/filename),origin=origin,tracked_samples=int((pose.body.confidence>0).sum()),missing_samples=int((pose.body.confidence==0).sum()),language='fsl',review='pending')
            records.append(record)
            duration_ms=int(np.ceil(1000*len(pose.body.data)/pose.body.fps))
            for label in dict.fromkeys([e['concept'],*e['aliases']]):
                rows.append(dict(words=label,glosses=e['gloss'],spoken_language='fr',signed_language='fsl',path=filename,start=0,end=duration_ms,priority=e['priority']))
        except Exception as ex:errors.append(dict(id=e['id'],error=str(ex)))
    with (out/'index.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=['words','glosses','spoken_language','signed_language','path','start','end','priority']);writer.writeheader();writer.writerows(rows)
    manifest=dict(upstream_revision=UPSTREAM_REV,language='fsl',spoken_language='fr',allow_local_preview=allow_local_preview,redistribution_allowed=False if any(e['local_preview_only'] for e in records) else 'Per-source CC BY attribution terms; no linguistic approval',source_library_sha256=library_sha,records=records,blocked_pending_ids=blocked,export_errors=errors,counts=dict(entries=len(records),csv_rows=len(rows),raw_animics=sum(r['origin']['format']=='raw-animics-image-landmarks' for r in records),retargeted=sum(r['origin']['format']=='stored-retargeted-motion' for r in records)),seconds=time.monotonic()-start)
    manifest['fingerprint']=manifest_fingerprint(manifest)
    manifest['fingerprint_algorithm']='sha256canonicalJSON(upstream/languages/preview-gate/source-registry/records/errors), excluding elapsed seconds'
    dump(out/'manifest.json',manifest);return manifest
