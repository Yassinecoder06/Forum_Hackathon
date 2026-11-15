import copy
from types import SimpleNamespace as Point

import pytest

from experiments.motion_extraction.extract import normalize
from experiments.motion_extraction.convert import convert


def pose():
    p = [Point(x=.5, y=.5, z=0.) for _ in range(33)]
    p[11] = Point(x=.3, y=.4, z=0.)
    p[12] = Point(x=.7, y=.4, z=0.)
    return p


def intermediate():
    return dict(format='mediapipe-intermediate-v2', fps=10, duration_ms=200,
                frames=[dict(time_ms=t, pose=normalize(pose()), left_hand=None, right_hand=None) for t in (0, 100)])


def test_shared_transform_and_anatomical_sides():
    p = pose()
    normalized = normalize(p)
    assert normalized[11] == pytest.approx([-.2, .1, 0])
    assert normalized[12] == pytest.approx([.2, .1, 0])
    assert normalize([p[11]], reference_pose=p)[0] == normalized[11]
    assert normalized[23] == [0, 0, 0]


@pytest.mark.parametrize('kind', ['missing', 'degenerate', 'nan'])
def test_invalid_pose(kind):
    p = pose()
    if kind == 'missing': p = []
    if kind == 'degenerate': p[12] = p[11]
    if kind == 'nan': p[0].z = float('nan')
    with pytest.raises(ValueError): normalize(p)


def test_player_format_and_hands():
    d = intermediate()
    d['frames'][0]['left_hand'] = [[-.4, .2, .1]] * 21
    result = convert(d)
    assert set(result) == {'rig_version', 'fps', 'duration_ms', 'joints', 'frames'}
    assert result['rig_version'] == 'signaculture-v1'
    assert len(result['joints']) == 9
    assert result['frames'][0]['joints']['left_wrist'] == [-.4, .2, .1]
    assert result['frames'][0]['joints']['left_shoulder'][0] < 0
    assert result['frames'][0]['joints']['right_shoulder'][0] > 0


@pytest.mark.parametrize('kind', ['empty', 'duplicate', 'backwards', 'outside', 'nan', 'old'])
def test_reject_bad_intermediate(kind):
    d = copy.deepcopy(intermediate())
    if kind == 'empty': d['frames'] = []
    if kind == 'duplicate': d['frames'][1]['time_ms'] = 0
    if kind == 'backwards': d['frames'][0]['time_ms'] = 150
    if kind == 'outside': d['frames'][1]['time_ms'] = 201
    if kind == 'nan': d['frames'][0]['pose'][0][0] = float('nan')
    if kind == 'old': d['format'] = 'mediapipe-intermediate-v1'
    with pytest.raises(ValueError): convert(d)


@pytest.mark.parametrize('times', [(50, 100), (0, 200), (100,)])
def test_endpoint_holds(times):
    d = intermediate()
    d['frames'] = [dict(time_ms=t, pose=normalize(pose())) for t in times]
    result = convert(d)
    frames = result['frames']
    assert frames[0]['time_ms'] == 0
    assert frames[-1]['time_ms'] == result['duration_ms'] == 200
    assert all(a['time_ms'] < b['time_ms'] for a, b in zip(frames, frames[1:]))
    assert frames[0]['joints'] == frames[1]['joints']
    assert frames[-1]['joints'] == frames[-2]['joints']
    assert {f['time_ms'] for f in frames} == {0, 200, *times}


def test_real_backend_asset_validator(tmp_path):
    """Opt in via backend path; import actual validator without editing its worktree."""
    import importlib.util
    import os
    import sys
    import types
    import json
    from pathlib import Path
    backend_app = Path(os.environ.get('SIGNACULTURE_BACKEND_APP',
                       Path(__file__).resolve().parents[2] / 'backend/app'))
    if not (backend_app / 'main.py').is_file():
        pytest.skip('Backend app/main.py is absent; set SIGNACULTURE_BACKEND_APP for an isolated worktree')
    package = types.ModuleType('_motion_backend')
    package.__path__ = [str(Path(backend_app).resolve())]
    sys.modules[package.__name__] = package
    spec = importlib.util.spec_from_file_location('_motion_backend.main', Path(backend_app)/'main.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    d = intermediate()
    d['frames'][0]['time_ms'] = 50
    output = convert(d)
    path = tmp_path/'data/approved_motion_assets/extracted.json'
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(output))
    metadata = module.MotionAsset(motion_id='EXTRACTED', gloss_id='TEST',
        path=str(path.relative_to(tmp_path)), rig_version='signaculture-v1',
        fps=10, duration_ms=200, license='test-only', quality_status='pending',
        rights_status='pending', animation_review='pending', synthetic=False)
    assert module.asset(tmp_path, metadata) == path
