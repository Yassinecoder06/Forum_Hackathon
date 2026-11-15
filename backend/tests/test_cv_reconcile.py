from scripts.reconcile_cv_holds import reconcile, native_frames


def frame(t, left, right):
    return {'time_ms':t,'PLM':[{'x':t}],'PWLM':None,'FLM':None,'LLM':left,'RLM':right,'LWLM':None,'RWLM':None}


def test_whole_original_frame_preserves_coordinates_instead_of_splicing_hands():
    old=frame(0,[{'x':1}],[{'x':2}]);new=frame(0,[{'x':99}],None)
    result=reconcile([old],[new])[0]
    assert result['LLM']==old['LLM'] and result['RLM']==old['RLM']
    assert result['capture_origin']=='baseline_frame_preserved'
    assert result['hand_diagnostics']['right']['observed']


def test_union_uses_only_actual_timestamps_and_preserves_true_elapsed_time():
    result=reconcile([frame(0,[],[]),frame(80,[1],[])],[frame(0,[2],[]),frame(40,[3],[])])
    assert [f['time_ms'] for f in result]==[0,40,80]
    assert [f['dt'] for f in result]==[0,.04,.04]
    assert result[0]['capture_origin']=='corrected_detector_frame'
    assert result[-1]['LLM']==[1]


def test_native_pose_comes_from_selected_extraction_and_endpoint_is_not_observed():
    raw=reconcile([frame(0,[1],[2])],[frame(0,[3],None)])
    a={'frames':[{'time_ms':0,'joints':{'wrist':[1,2,3]}}],'duration_ms':80}
    b={'frames':[{'time_ms':0,'joints':{'wrist':[9,9,9]}}],'duration_ms':80}
    frames,duration=native_frames(raw,a,b)
    assert frames[0]['joints']==a['frames'][0]['joints']
    assert frames[-1]['playback_origin']=='endpoint_hold' and duration==80


def test_sparse_fingers_are_tracked_but_never_required_during_missing_hand():
    from scripts.reconcile_cv_holds import native_joint_metadata
    frames=[{'joints':{'wrist':[0,0,0],'index_4':[1,0,0]}},{'joints':{'wrist':[0,0,0]}}]
    required,tracked=native_joint_metadata(frames)
    assert required==['wrist'] and tracked==['index_4','wrist']
    assert 'index_4' not in frames[-1]['joints']
