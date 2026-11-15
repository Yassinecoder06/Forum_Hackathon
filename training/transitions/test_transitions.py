"""Schema, boundary, source leakage and actual experiment verification."""
import json,hashlib
from collections import defaultdict
from pathlib import Path
import numpy as np
import pytest
from common import ROOT,ARTIFACTS,FEATURES,validate_endpoints,interpolate,errors,sha
from infer import bridge
RUN=ROOT/'research/runs/transition-intrasign-v1'
def load(p):return json.loads(Path(p).read_text())

def test_endpoint_validation():
    good={'start':[0.0]*10,'end':[1.0]*10,'feature_order':FEATURES}
    assert validate_endpoints(good).shape==(2,10)
    for bad in [dict(good,start=[0]*9),dict(good,end=[float('nan')]*10),dict(good,start=[True]*10),dict(good,feature_order=list(reversed(FEATURES)))]:
        with pytest.raises(ValueError):validate_endpoints(bad)

def test_deterministic_interpolators_and_exact_outer_anchors():
    obj={'start':np.arange(10,dtype=float).tolist(),'end':(np.arange(10,dtype=float)+11).tolist()}
    linear=bridge(obj,'linear');quintic=bridge(obj,'quintic')
    assert np.allclose(linear['frames'][0],np.arange(10)+1)
    assert np.allclose(linear['frames'][-1],np.arange(10)+10)
    assert not np.allclose(linear['frames'],quintic['frames'])
    for r in [linear,quintic]:
        assert r['output_frame_count']==len(r['frames'])==10
        assert r['bridge_with_anchors'][0]==obj['start'] and r['bridge_with_anchors'][-1]==obj['end']
        assert r['translation_validated'] is False

def test_split_before_windows_and_source_or_mirror_nonoverlap():
    m=load(ROOT/'research/TRANSITION_DATA_MANIFEST.json');seen=defaultdict(set)
    for s,rows in m['windows'].items():
        for r in rows:seen[r['group_id']].add(s)
        assert len(rows)==m['window_counts'][s]
    assert all(len(s)==1 for s in seen.values())
    for field in ['source_url','geometry_sha256','clip_id']:
        a=defaultdict(set)
        for r in m['clips']:a[r[field]].add(r['split'])
        assert all(len(s)==1 for s in a.values())

def test_normalizer_train_only_and_immutable_source_targets():
    m=load(ROOT/'research/TRANSITION_DATA_MANIFEST.json');train=np.load(ARTIFACTS/'train.npz')['windows'];flat=train.reshape(-1,10).astype(np.float64)
    assert np.allclose(flat.mean(0),m['normalization']['mean'])
    assert np.allclose(np.maximum(flat.std(0),1e-4),m['normalization']['std'])
    # Inspect ALL heldout windows against the actual source JSON. Never impute target points.
    cache={}
    clips={r['clip_id']:r for r in m['clips']}
    for row in m['windows']['test']:
        c=clips[row['clip_id']]
        if c['clip_id'] not in cache:
            assert sha(c['path'])==c['input_sha256'];cache[c['clip_id']]=load(c['path'])
        frames=cache[c['clip_id']]['frames'][row['first_frame']:row['first_frame']+12]
        assert len(frames)==12
        assert [f['time_ms'] for f in frames]==row['times_ms']
        for f in frames:assert all(k in f['joints'] for k in ['head','left_wrist','right_wrist','left_middle_4','right_middle_4'])

def test_measured_metrics_selection_and_one_heldout_pass():
    r=load(RUN/'metrics.json');h=load(RUN/'loss_history.json');selection=load(RUN/'selection_complete.json')
    assert r['test_evaluation_passes']==1 and selection['test_opened'] is False
    assert r['best_epoch']==min(h,key=lambda x:x['val_rmse'])['epoch']
    assert r['device']=='cpu' and r['cuda_allocated_peak_mib']==0
    if r['epochs']<50:assert r['epochs']-r['best_epoch']==8
    p=[json.loads(x) for x in (RUN/'test_predictions.jsonl').read_text().splitlines()]
    m=load(ROOT/'research/TRANSITION_DATA_MANIFEST.json');mean=np.array(m['normalization']['mean']);std=np.array(m['normalization']['std'])
    target=np.array([x['target_native_xy'] for x in p]);target=(target-mean)/std
    for method in ['linear','quintic','gan']:
        pred=(np.array([x['predictions_native_xy'][method] for x in p])-mean)/std
        assert abs(errors(pred,target,None)['rmse']-r['metrics'][method]['normalized']['rmse'])<1e-6
    if r['gan_worse_than_selected_baseline']:assert r['recommendation'] in ['linear','quintic']

def test_actual_trained_checkpoint_and_cli_bridge():
    obj=load(RUN/'example_endpoints.json');r=bridge(obj,'gan')
    assert np.asarray(r['frames']).shape==(10,10) and np.isfinite(r['frames']).all()
    assert r['bridge_with_anchors'][0]==obj['start'] and r['bridge_with_anchors'][-1]==obj['end']
    assert r['checkpoint_sha256']==load(RUN/'metrics.json')['checkpoint_sha256']
