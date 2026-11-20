"""Bounded SignVerse reader. XY+confidence; never interprets score as depth."""
import io,json,pickle,tarfile,zipfile
from pathlib import Path, PurePosixPath
import numpy as np

PARTS={'body':(0,18),'left_hand':(18,39),'right_hand':(39,60),'face':(60,128)}
BONES=[(1,2),(2,3),(3,4),(1,5),(5,6),(6,7),(1,8),(8,9),(9,10),(1,11),(11,12),(12,13)]
for start in (18,39):
    for finger in range(5):
        chain=[start]+list(range(start+1+finger*4,start+5+finger*4));BONES.extend(zip(chain[:-1],chain[1:]))
class RestrictedNumpyUnpickler(pickle.Unpickler):
    def find_class(self,module,name):
        allowed={('numpy','ndarray'):np.ndarray,('numpy','dtype'):np.dtype}
        for prefix in ('numpy.core.multiarray','numpy._core.multiarray'):
            allowed[(prefix,'_reconstruct')]=np._core.multiarray._reconstruct
            allowed[(prefix,'scalar')]=np._core.multiarray.scalar
        if (module,name) not in allowed:raise ValueError(f'Unsupported pickle global {module}.{name}')
        return allowed[module,name]

def read_npz(path,max_bytes=256*1024**2):
    """Allow only NumPy containers, not arbitrary pickle code. Bound decompressed size."""
    out={}
    with zipfile.ZipFile(path) as z:
        if sum(x.file_size for x in z.infolist())>max_bytes:raise ValueError('NPZ exceeds decompressed budget')
        for member in z.infolist():
            if not member.filename.endswith('.npy') or '/' in member.filename:raise ValueError('Unexpected NPZ member')
            raw=z.read(member);fp=io.BytesIO(raw);version=np.lib.format.read_magic(fp)
            if version==(1,0):shape,fortran,dtype=np.lib.format.read_array_header_1_0(fp)
            elif version==(2,0):shape,fortran,dtype=np.lib.format.read_array_header_2_0(fp)
            else:raise ValueError('Unsupported NPY header version')
            if np.prod(shape,dtype=np.float64)>30_000_000:raise ValueError('Array shape exceeds budget')
            if dtype.hasobject:
                array=RestrictedNumpyUnpickler(fp).load()
            else:array=np.load(io.BytesIO(raw),allow_pickle=False)
            out[member.filename[:-4]]=array
    return out

def safe_members(archive,video_ids,max_bytes=256*1024**2):
    """Yield approved bytes without extractall; reject links and traversal."""
    used=0
    with tarfile.open(archive,'r:*') as tar:
        for m in tar:
            p=PurePosixPath(m.name)
            if p.is_absolute() or '..' in p.parts or m.issym() or m.islnk():raise ValueError('Unsafe archive member')
            if not m.isfile() or p.name not in ('poses.npz','caption.json'):continue
            if len(p.parts)!=2 or p.parts[0] not in video_ids:continue
            used+=m.size
            if used>max_bytes:raise ValueError('Extraction budget exceeded')
            yield str(p),tar.extractfile(m).read()

def validate_caption(caption,manifest):
    if caption.get('sign_language')!='fsl' or manifest.get('sign_language')!='fsl':raise ValueError('Both record and manifest must identify fsl')
    if str(caption.get('video_id'))!=str(manifest['video_id']):raise ValueError('Video mismatch')
    return True

def select_track(frames,threshold=.3):
    """Quality/continuity heuristic, not signer identity recognition; flag ambiguity."""
    track=[];ids=[];ambiguous=[];previous=None
    for frame in frames:
        candidates=[]
        for name,person in frame.items():
            if not name.startswith('person_') or not isinstance(person,dict):continue
            points=np.zeros((128,3),np.float32)
            for part,(a,b) in PARTS.items():
                value=np.asarray(person.get(part,np.zeros((b-a,3))),dtype=np.float32)
                if value.shape!=(b-a,3):raise ValueError(f'Wrong {part} shape')
                points[a:b]=value
            valid=np.isfinite(points).all(-1)&(points[:,2]>=threshold)&(points[:,2]<=1)
            if not valid[[2,5]].all():continue
            center=points[[2,5],:2].mean(0);scale=np.linalg.norm(points[2,:2]-points[5,:2])
            if scale<=1e-6:continue
            quality=float(valid[:18].mean()+valid[18:60].mean())
            continuity=0 if previous is None else np.linalg.norm(center-previous[0])/max(scale,previous[1])
            candidates.append((quality-min(continuity,5),name,points,center,scale))
        candidates.sort(key=lambda c:c[0],reverse=True)
        if not candidates:track.append(np.zeros((128,3),np.float32));ids.append('missing');ambiguous.append(True);continue
        score,name,points,center,scale=candidates[0];ambiguous.append(len(candidates)>1 and score-candidates[1][0]<.2)
        track.append(points);ids.append(name);previous=(center,scale)
    return np.asarray(track),ids,np.asarray(ambiguous)

def preprocess(raw,frame_ids,fps=24.,threshold=.3):
    raw=np.asarray(raw,np.float32);frame_ids=np.asarray(frame_ids)
    if raw.ndim!=3 or raw.shape[1:]!=(128,3) or len(frame_ids)!=len(raw):raise ValueError('Expected Tx128x3')
    if not np.isclose(fps,24):raise ValueError('Expected documented 24fps; resampling must be explicit')
    if len(raw)<2 or np.any(np.diff(frame_ids)<=0):raise ValueError('Frame IDs must increase')
    confidence=raw[:,:,2].copy();mask=np.isfinite(raw).all(-1)&(confidence>=threshold)&(confidence<=1)
    xy=np.nan_to_num(raw[:,:,:2]);anchors=mask[:,2]&mask[:,5]
    center=(xy[:,2]+xy[:,5])/2;scales=np.linalg.norm(xy[:,2]-xy[:,5],axis=-1);good=anchors&(scales>1e-6)
    if not good.any():raise ValueError('No trustworthy shoulders')
    # A single robust recording scale retains changes in movement amplitude.
    scale=float(np.median(scales[good]));norm=(xy-center[:,None])/scale;mask&=good[:,None]
    discontinuity=np.r_[False,np.diff(frame_ids)!=1]
    speed=np.linalg.norm(np.diff(norm,axis=0),axis=-1);jump=np.zeros_like(mask);jump[1:]=speed>1.;mask&=~jump;mask[discontinuity]=False
    hand_swaps=[]
    for t in range(len(raw)):
        if mask[t,[4,7,18,39]].all():
            own=np.linalg.norm(norm[t,18]-norm[t,7])+np.linalg.norm(norm[t,39]-norm[t,4]);cross=np.linalg.norm(norm[t,18]-norm[t,4])+np.linalg.norm(norm[t,39]-norm[t,7])
            if cross+.15<own:hand_swaps.append(t);mask[t,18:60]=False
    # Do not smooth through missing frames, jumps, or suspected hand swaps.
    smooth=norm.copy()
    for t in range(1,len(raw)-1):
        eligible=mask[t-1]&mask[t]&mask[t+1]
        weights=np.clip(confidence[t-1:t+2],0,1)*mask[t-1:t+2]
        avg=(norm[t-1:t+2]*weights[:,:,None]).sum(0)/np.maximum(weights.sum(0)[:,None],1e-8)
        smooth[t,eligible]=avg[eligible]
    norm[~mask]=0;smooth[~mask]=0
    quality={'valid_fraction':float(mask.mean()),'parts':{p:float(mask[:,a:b].mean()) for p,(a,b) in PARTS.items()},'discontinuities':int(discontinuity.sum()),'suspected_hand_swap_frames':hand_swaps,'jump_landmarks':int(jump.sum()),'normalization_scale_pixels':scale,'coordinate_system':'shoulder-centered 2D XY, recording-median shoulder scale','fps':24.,'clinical_or_linguistic_validity':False}
    return {'original_xyc':raw,'normalized_xy':norm,'smoothed_xy':smooth,'confidence':confidence,'mask':mask,'frame_ids':frame_ids,'center_xy':center,'scale':np.array(scale)},quality

def load_record(npz_path,caption,manifest):
    validate_caption(caption,manifest);payload=read_npz(npz_path)
    fps=float(payload['fps']);frames=payload['frames'].tolist()
    if str(payload['video_id'].item())!=str(manifest['video_id']):raise ValueError('Pose/video mismatch')
    raw,person_ids,ambiguous=select_track(frames)
    arrays,quality=preprocess(raw,payload['frame_ids'],fps)
    arrays['mask'][ambiguous]=False;arrays['normalized_xy'][ambiguous]=0;arrays['smoothed_xy'][ambiguous]=0
    arrays['selected_person']=np.asarray(person_ids);arrays['ambiguous_person']=ambiguous
    quality['ambiguous_person_frames']=int(ambiguous.sum());quality['valid_fraction']=float(arrays['mask'].mean())
    return arrays,quality
