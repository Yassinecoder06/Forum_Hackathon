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
