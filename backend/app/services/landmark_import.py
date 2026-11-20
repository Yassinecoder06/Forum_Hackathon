"""Import position clips or Animics Keyframe raw PLM/LLM/RLM landmark arrays.

The upstream field names and anatomical indices follow UPF-GTI Animics,
https://github.com/upf-gti/animics (src/skeleton.js, src/Mediapipe.js).
This adapter is independently implemented; it does not import Script Mode BML.
"""
from copy import deepcopy
import math
from statistics import median
from ..workers.face_features import measure_face

BODY={'head':0,'left_shoulder':11,'right_shoulder':12,'left_elbow':13,'right_elbow':14,'left_wrist':15,'right_wrist':16}
FINGERS={'thumb':[1,2,3,4],'index':[5,6,7,8],'middle':[9,10,11,12],'ring':[13,14,15,16],'pinky':[17,18,19,20]}

def image_dimensions(frame):
    width, height = frame.get('image_width'), frame.get('image_height')
    if all(type(v) is int and 0 < v <= 16384 for v in (width, height)):
        return width, height
    return None

def imported_face(frame):
    points = frame.get('FLM')
    dimensions = image_dimensions(frame)
    if not isinstance(points, list) or len(points) < 387:
        return None
    if dimensions is None:
        return None
    try:
        return measure_face(points, *dimensions)
    except (AttributeError, KeyError, TypeError, ValueError, OverflowError):
        return None

def validate_motion(data):
    if not isinstance(data,dict) or data.get('rig_version')!='signaculture-v1': raise ValueError('Format de mouvement inconnu.')
    fps=data.get('fps');duration=data.get('duration_ms');frames=data.get('frames')
    if not isinstance(fps,(int,float)) or not math.isfinite(fps) or not 0<fps<=240: raise ValueError('FPS invalide.')
    if not isinstance(duration,int) or not 0<duration<=60000: raise ValueError('Durée limitée à 60 secondes.')
    if not isinstance(frames,list) or not 2<=len(frames)<=15000: raise ValueError('Séquence vide ou trop longue.')
    previous=-1
    for frame in frames:
        t=frame.get('time_ms')
        if not isinstance(t,int) or isinstance(t,bool) or not previous<t<=duration: raise ValueError('Horodatage invalide.')
        previous=t
        joints=frame.get('joints')
        if not isinstance(joints,dict) or not set(BODY).union({'hips','neck'}).issubset(joints): raise ValueError('Articulations corporelles manquantes.')
        if any(not isinstance(xyz,list) or len(xyz)!=3 or any(not isinstance(n,(int,float)) or isinstance(n,bool) or not math.isfinite(n) for n in xyz) for xyz in joints.values()): raise ValueError('Coordonnées invalides.')
    if frames[0]['time_ms']!=0 or frames[-1]['time_ms']!=duration: raise ValueError('Les images doivent couvrir toute la durée du clip.')
    return data

def convert_animics(data, fps=30):
    if isinstance(data,dict): return validate_motion(data)
    if not isinstance(data,list) or not 2<=len(data)<=1800: raise ValueError('Export Animics vide ou trop long.')
    frames=[];left=right=faces=0;time=0
    if any(not isinstance(frame, dict) for frame in data):
        raise ValueError('Image Animics invalide.')
    deltas = [f['dt'] for f in data if isinstance(f.get('dt'), (int,float)) and 0 < f['dt'] < math.inf]
    timing = bool(deltas)
    if timing:
        fps = min(240, 1 / median(deltas))
    first_stamp = None
    last_stamp = -1
    sampled = 0
    for index,frame in enumerate(data):
        if timing:
            delta=frame.get('dt',0)
            if not isinstance(delta,(int,float)) or not math.isfinite(delta) or delta<0: raise ValueError('Intervalle Animics invalide.')
            time+=delta
            stamp=round(time*1000)
        else: stamp=round(index*1000/fps)
        if first_stamp is None:
            first_stamp = stamp
        stamp -= first_stamp
        if stamp <= last_stamp:
            continue
        last_stamp = stamp
        sampled += 1
        pose=frame.get('PLM')
        if 'PLM' in frame and pose is None:
            continue
        if not isinstance(pose,list) or len(pose)<33: raise ValueError('Export Keyframe attendu : PLM doit contenir 33 points.')
        for point in pose:
            if not isinstance(point,dict) or any(not isinstance(point.get(a),(int,float)) or not math.isfinite(point[a]) for a in ('x','y','z')): raise ValueError('Point Animics invalide.')
        dimensions = image_dimensions(frame)
        width, height = dimensions or (1, 1)
        shoulder_span = math.hypot((pose[11]['x']-pose[12]['x'])*width,
                                   (pose[11]['y']-pose[12]['y'])*height)
        if shoulder_span < (3 if dimensions else 1e-5):
            continue
        scale = .4 / shoulder_span
        world = frame.get('PWLM')
        if dimensions and isinstance(world, list) and len(world) == 33 and all(
            isinstance(p, dict) and all(
                isinstance(p.get(axis), (int, float)) and math.isfinite(p[axis])
                for axis in ('x', 'y', 'z')
            ) for p in world
        ):
            # The Animics world array already stores Y up and Z forward.
            origin = {axis: (world[23][axis] + world[24][axis]) / 2 for axis in ('x', 'y', 'z')}
            def point(i):
                return [world[i]['x'] - origin['x'], 1 + (world[i]['y'] - origin['y']),
                        world[i]['z'] - origin['z']]
        else:
            origin = {axis: (pose[23][axis] + pose[24][axis]) / 2 for axis in ('x', 'y', 'z')}
            def point(i):
                return [(pose[i]['x'] - origin['x']) * width * scale,
                        1 - (pose[i]['y'] - origin['y']) * height * scale,
                        -(pose[i]['z'] - origin['z']) * width * scale]
        joints={name:point(i) for name,i in BODY.items()}
        joints['hips']=[0.,1.,0.];joints['neck']=[(point(11)[a]+point(12)[a])/2 for a in range(3)]
        tracking = {}
        for side,key in [('left','LLM'),('right','RLM')]:
            hand=frame.get(key)
            observed = isinstance(hand,list) and len(hand)==21
            tracking[side] = {'observed': observed, 'source': 'animics' if observed else 'missing'}
            if not observed: continue
            anchor=joints[f'{side}_wrist'];wrist=hand[0]
            for finger,indices in FINGERS.items():
                for number,i in enumerate(indices,1):
                    p=hand[i]
                    joints[f'{side}_{finger}_{number}']=[
                        anchor[0] + (p['x'] - wrist['x']) * width * scale,
                        anchor[1] - (p['y'] - wrist['y']) * height * scale,
                        anchor[2] - (p['z'] - wrist['z']) * width * scale,
                    ]
        face = imported_face(frame)
        left += int(tracking['left']['observed'])
        right += int(tracking['right']['observed'])
        faces += int(face is not None)
        frames.append({'time_ms':stamp,'joints':joints,'face':face,'hand_tracking':tracking})
    if not frames: raise ValueError('Aucune image utilisable.')
    retained = len(frames)
    duration = last_stamp + round(1000 / fps)
    if frames[0]['time_ms'] > 0:
        first = deepcopy(frames[0])
        first.update(time_ms=0, playback_origin='endpoint_hold')
        frames.insert(0, first)
    last = deepcopy(frames[-1])
    last.update(time_ms=duration, playback_origin='endpoint_hold')
    frames.append(last)
    return validate_motion({'rig_version':'signaculture-v1','fps':fps,'duration_ms':duration,'joints':list(BODY)+['hips','neck'],
        'frames':frames,'synthetic':False,'capture_method':'animics-keyframe-import',
        'quality':{'sampled_frames':sampled,'pose_frames':retained,'missing_frames':sampled-retained,'pose_coverage':round(retained/sampled,3),'left_hand_coverage':round(left/sampled,3),'right_hand_coverage':round(right/sampled,3),'face_coverage':round(faces/sampled,3)},
        'warnings':['Import de positions Animics ; calibration de profondeur approximative.']+([] if timing else ['Horodatage absent : cadence supposée de 30 images par seconde.'])})
