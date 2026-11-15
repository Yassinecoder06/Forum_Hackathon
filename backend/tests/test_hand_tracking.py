from backend.app.workers.hand_tracking import assign_hands
from backend.app.workers.hand_tracking import roi_box, map_roi_hand
from types import SimpleNamespace


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
