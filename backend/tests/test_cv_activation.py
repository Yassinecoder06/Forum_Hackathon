from scripts.activate_cv_corpus import decision


def candidate(gate):
    return {'status':'complete','comparisons':[{'capture_id':'a','gate':gate}]}


def test_regression_is_held_even_with_unbaselined_option():
    assert decision({'capture_id':'a'},candidate('review_regression_or_incomparable'),True).startswith('hold_')


def test_preview_requires_explicit_unbaselined_option():
    row={'capture_id':'a'};result=candidate('review_no_raw_baseline')
    assert decision(row,result,False).startswith('hold_')
    assert decision(row,result,True)=='activate_unbaselined_preview'


def test_capture_specific_gate_cannot_be_borrowed_from_mirror():
    assert decision({'capture_id':'b'},candidate('no_observation_loss'),True)=='missing_comparison'


def test_active_focused_fix_is_preserved():
    assert decision({'cv_generation':'focused','capture_id':'a'},candidate('no_observation_loss'),True)=='already_active'
