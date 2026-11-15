from backend.app.workers.hand_tracking import assign_hands
from backend.app.workers.hand_tracking import roi_box, map_roi_hand
from types import SimpleNamespace
from backend.app.workers.hand_tracking import consistent_hand, agreed_candidates


def test_two_detections_get_distinct_anatomical_sides():
    wrists = {'left': (.4, .5), 'right': (.5, .5)}
    # Both are nearer the left wrist; greedy independent matching loses the right.
    result = assign_hands([(.4, .5), (.44, .5)], wrists)
    assert result == {'left': 0, 'right': 1}
    assert len(set(result.values())) == len(result)


def test_locked_holistic_hand_is_not_reused_for_missing_side():
    wrists = {'left': (.4, .5), 'right': (.5, .5)}
    assert assign_hands([(.4, .5)], wrists, existing={'left': True}) == {}
    assert assign_hands([(.5, .5)], wrists, existing={'left': True}) == {'right': 0}


def test_ambiguous_crossing_abstains_and_distant_detection_is_rejected():
    wrists = {'left': (.4, .5), 'right': (.5, .5)}
    assert assign_hands([(.45, .5)], wrists) == {}
    assert assign_hands([(.9, .5)], wrists) == {}


def test_nearby_history_breaks_tie_without_accepting_far_detection():
    wrists = {'left': (.4, .5), 'right': (.5, .5)}
    assert assign_hands([(.445, .5)], wrists, previous={'left': (.44, .5), 'right': (.55, .5)}) == {'left': 0}


def test_roi_stays_square_at_image_edges():
    for width,height in [(320,240),(240,320)]:
        x,y,size=roi_box(width,0,500,width,height)
        assert 0 <= x <= width-size and 0 <= y <= height-size
        assert size == min(width,height)


def test_roi_coordinates_restore_original_pixels_without_mutating_detector():
    hand=SimpleNamespace(landmark=[SimpleNamespace(x=.5,y=.25,z=.1)])
    mapped=map_roi_hand(hand,40,20,100,320,240).landmark[0]
    assert mapped.x*320 == 90 and mapped.y*240 == 45
    assert mapped.z == .1*100/320
    assert hand.landmark[0].x == .5


def test_scale_agreement_rejects_different_or_unresolved_hands():
    a=SimpleNamespace(landmark=[SimpleNamespace(x=.1+i*.005,y=.3,z=0) for i in range(21)])
    close=SimpleNamespace(landmark=[SimpleNamespace(x=p.x+.001,y=p.y,z=0) for p in a.landmark])
    wrong=SimpleNamespace(landmark=[SimpleNamespace(x=p.x+.1,y=p.y,z=0) for p in a.landmark])
    assert consistent_hand(a,close,320,240)
    assert not consistent_hand(a,wrong,320,240)
    assert not consistent_hand(a,a,10,10)


def test_bidirectional_agreement_deduplicates_one_physical_hand():
    def hand(offset):return SimpleNamespace(landmark=[SimpleNamespace(x=offset+i*.005,y=.3,z=0) for i in range(21)])
    a,b,other=hand(.1),hand(.101),hand(.6)
    result=agreed_candidates([a,b,other],[b],320,240)
    assert result==[a], 'Repeated crop detections must not become two different hands'
    assert agreed_candidates([a],[other],320,240)==[]
