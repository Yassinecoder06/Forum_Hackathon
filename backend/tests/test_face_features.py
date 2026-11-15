import math
from backend.app.workers.face_features import measure_face


def points():
    p=[{'x':.5,'y':.5} for _ in range(468)]
    for i,x,y in [(33,.3,.4),(263,.7,.4),(105,.35,.3),(159,.35,.4),(334,.65,.3),(386,.65,.4),(13,.5,.6),(14,.5,.65)]:p[i]={'x':x,'y':y}
    return p


def test_level_face_preserves_legacy_ratio_scale():
    f=measure_face(points(),640,480)
    assert abs(f['brow_left']-.25)<1e-12 and abs(f['mouth_open']-.125)<1e-12


def test_pixel_plane_rotation_does_not_change_expression_for_non_square_image():
    p=points();expected=measure_face(p,640,480)
    for angle in [-.5,.3,.8]:
        c,s=math.cos(angle),math.sin(angle)
        rotated=[]
        for q in p:
            x,y=(q['x']-.5)*640,(q['y']-.5)*480
            rotated.append({'x':.5+(c*x-s*y)/640,'y':.5+(s*x+c*y)/480})
        actual=measure_face(rotated,640,480)
        assert all(abs(expected[k]-actual[k])<1e-12 for k in expected)


def test_degenerate_eye_line_is_missing_not_confident_neutral():
    p=points();p[263]=p[33].copy()
    assert measure_face(p,640,480) is None


def test_derived_face_repair_preserves_body_fingers_and_marks_endpoint_hold():
    from copy import deepcopy
    from scripts.apply_face_roll import derive
    hand={'left':{'observed':True},'right':{'observed':False}}
    source={'frames':[{'time_ms':0,'joints':{'left_index_4':[1,2,3]},'hand_tracking':hand,'face':{'mouth_open':0,'brow_left':0,'brow_right':0}}, {'time_ms':100,'joints':{'left_index_4':[1,2,3]},'hand_tracking':hand,'face':None}]}
    before=deepcopy(source)
    raw=[{'time_ms':0,'FLM':points(),'image_width':640,'image_height':480}]
    updated,count,_=derive(source,raw)
    assert source==before and count==1
    assert updated['frames'][0]['joints']==before['frames'][0]['joints']
    assert updated['frames'][0]['hand_tracking']==hand
    assert updated['frames'][-1]['playback_origin']=='endpoint_hold'
    assert updated['frames'][-1]['face'] is None
