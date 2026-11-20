"""Five-point XY masked-window helpers. No full signing or coarticulation claim."""
import hashlib,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[4]
MAIN=ROOT.parents[1] if ROOT.parent.name=='.worktrees' else ROOT
ARTIFACTS=ROOT/'research/code/training/transitions/artifacts'
JOINTS=['head','left_wrist','right_wrist','left_middle_4','right_middle_4']
FEATURES=[j+'_'+axis for j in JOINTS for axis in ['x','y']]
T=np.arange(1,11,dtype=np.float32)/11
CLAIM='INTRA_SIGN_MASKED_WINDOW_RECONSTRUCTION_NOT_INTER_SIGN_COARTICULATION'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,obj):Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def validate_endpoints(obj):
    if not isinstance(obj,dict) or 'start' not in obj or 'end' not in obj:raise ValueError('Expected start/end arrays with10XYfeatures each')
    if 'feature_order' in obj and obj['feature_order']!=FEATURES:raise ValueError('Wrong feature order')
    rows=[]
    for key in ['start','end']:
        a=obj[key]
        if not isinstance(a,list) or len(a)!=10 or any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>1e6 for v in a):raise ValueError('Endpoints must be10finite numeric XYfeatures, magnitude<=1e6')
        rows.append(a)
    return np.asarray(rows,dtype=np.float32)
def interpolate(endpoints,method='linear'):
    x=np.asarray(endpoints,dtype=np.float32)
    if x.shape[-2:]!=(2,10) or not np.isfinite(x).all():raise ValueError('Invalid endpoint shape/values')
    t=T if method=='linear' else 6*T**5-15*T**4+10*T**3 if method=='quintic' else None
    if t is None:raise ValueError('Unknown interpolation method')
    return x[...,0,None,:]*(1-t[:,None])+x[...,1,None,:]*t[:,None]
def errors(pred,target,endpoints):
    p=np.asarray(pred,dtype=np.float64);y=np.asarray(target,dtype=np.float64);d=p-y
    if p.shape!=y.shape or p.shape[-2:]!=(10,10):raise ValueError('Expected10interiorframes x10features')
    boundary=d[..., [0,-1],:]
    return dict(rmse=float(np.sqrt(np.mean(d*d))),mae=float(np.mean(np.abs(d))),
                inner_boundary_rmse=float(np.sqrt(np.mean(boundary*boundary))),
                inner_boundary_mae=float(np.mean(np.abs(boundary))),
                supplied_anchor_rmse=0.,supplied_anchor_note='Outer anchors are copied exactly; zero is structural, not learned continuity evidence')
