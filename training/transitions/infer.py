"""Generate TEN interior bridge frames for valid five-pointXYendpoint arrays."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from common import ARTIFACTS,FEATURES,CLAIM,validate_endpoints,interpolate,sha

def bridge(obj,method='linear',checkpoint=None):
    endpoints=validate_endpoints(obj);warnings=['PENDING_LINGUISTIC_AND_ANIMATION_REVIEW','TEN_XY_FEATURES_ONLY_NOT_FULL_RIG_MOTION','NO_VALIDATED_INTER_SIGN_COARTICULATION_CLAIM']
    metadata={}
    if method in ['linear','quintic']:pred=interpolate(endpoints,method)
    elif method=='gan':
        import torch
        from models import Generator
        torch.set_num_threads(4)
        path=Path(checkpoint) if checkpoint else ARTIFACTS/'transition-intrasign-v1.pt'
        state=torch.load(path,map_location='cpu',weights_only=True)
        if state['feature_order']!=FEATURES or state['kind']!=CLAIM:raise ValueError('Checkpoint is not the expected five-point research model')
        norm=state['normalization'];mean=np.asarray(norm['mean'],dtype=np.float32);std=np.asarray(norm['std'],dtype=np.float32)
        x=(endpoints-mean)/std
        if np.max(np.abs(x))>5:warnings.append('ENDPOINTS_OUTSIDE_TYPICAL_TRAINING_RANGE')
        g=Generator();g.load_state_dict(state['generator_state']);g.eval()
        with torch.no_grad():pred=g(torch.from_numpy(x[None]))[0].numpy()*std+mean
        metadata=dict(checkpoint_sha256=sha(path),selected_epoch=state['best_epoch'],inference_noise='zeros',device='cpu')
    else:raise ValueError('Unknown bridge method')
    if pred.shape!=(10,10) or not np.isfinite(pred).all():raise ValueError('Nonfinite or invalid generated bridge')
    # Supplied anchors are preserved, not predicted or scored as learned success.
    return dict(method=method,feature_order=FEATURES,start=obj['start'],end=obj['end'],frames=pred.tolist(),bridge_with_anchors=[obj['start'],*pred.tolist(),obj['end']],output_frame_count=10,time_fractions=[(i+1)/11 for i in range(10)],timing_note='Interior fractions only; no30fps/0.33s timing guarantee; caller supplies native timing',claim=CLAIM,candidate=True,translation_validated=False,linguistic_review='pending',warnings=warnings,**metadata)

def main():
    p=argparse.ArgumentParser();p.add_argument('--endpoints',required=True,type=Path);p.add_argument('--method',choices=['linear','quintic','gan'],default='linear');p.add_argument('--checkpoint',type=Path);a=p.parse_args()
    try:result=bridge(json.loads(a.endpoints.read_text()),a.method,a.checkpoint)
    except Exception as e:
        print(json.dumps(dict(error=str(e),frames=[],candidate=False,claim=CLAIM)));return 2
    print(json.dumps(result,allow_nan=False));return 0
if __name__=='__main__':sys.exit(main())
