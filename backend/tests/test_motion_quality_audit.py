"""Software fixtures exercise triage thresholds; they are not signing examples."""
import copy
from scripts.audit_library_motion import audit


def fixture():
    joints = {}
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        for n in range(1,5):
            joints[f'left_{finger}_{n}'] = [n*.015, 1., .3]
    return {'frames':[{'time_ms':t,'joints':copy.deepcopy(joints)} for t in [0,40,80]]}


def test_isolated_length_burst_flagged_without_mutating_source():
    motion=fixture()
    for key,point in motion['frames'][1]['joints'].items():
        point[0]*=3
    original=copy.deepcopy(motion)
    events=audit(motion)
    assert len(events)==1
    assert events[0]['kind']=='isolated_finger_length_burst'
    assert motion==original


def test_sustained_handshape_not_flagged():
    motion=fixture()
    for frame in motion['frames']:
        for point in frame['joints'].values():point[0]*=3
    assert audit(motion)==[]


def test_long_gap_not_mislabeled_as_isolated_spike():
    motion=fixture();motion['frames'][-1]['time_ms']=400
    for point in motion['frames'][1]['joints'].values():point[0]*=3
    assert audit(motion)==[]
