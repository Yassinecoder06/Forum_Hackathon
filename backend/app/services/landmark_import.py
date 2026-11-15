"""Import position clips or Animics Keyframe raw PLM/LLM/RLM landmark arrays.

The upstream field names and anatomical indices follow UPF-GTI Animics,
https://github.com/upf-gti/animics (src/skeleton.js, src/Mediapipe.js).
This adapter is independently implemented; it does not import Script Mode BML.
"""
from copy import deepcopy
import math

BODY={'head':0,'left_shoulder':11,'right_shoulder':12,'left_elbow':13,'right_elbow':14,'left_wrist':15,'right_wrist':16}
FINGERS={'thumb':[1,2,3,4],'index':[5,6,7,8],'middle':[9,10,11,12],'ring':[13,14,15,16],'pinky':[17,18,19,20]}

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
    frames=[];left=right=0;time=0
    timing=any(isinstance(f.get('dt'),(int,float)) and f['dt']>0 for f in data)
    for index,frame in enumerate(data):
        pose=frame.get('PLM')
        if not isinstance(pose,list) or len(pose)<33: raise ValueError('Export Keyframe attendu : PLM doit contenir 33 points.')
        for point in pose:
            if not isinstance(point,dict) or any(not isinstance(point.get(a),(int,float)) or not math.isfinite(point[a]) for a in ('x','y','z')): raise ValueError('Point Animics invalide.')
        width=math.hypot(pose[11]['x']-pose[12]['x'],pose[11]['y']-pose[12]['y'])
        if width<1e-5: raise ValueError('Échelle des épaules indéterminée.')
        scale=.4/width;origin={a:(pose[23][a]+pose[24][a])/2 for a in ('x','y','z')}
        def point(i): return [(pose[i]['x']-origin['x'])*scale,1-(pose[i]['y']-origin['y'])*scale,-(pose[i]['z']-origin['z'])*scale]
        joints={name:point(i) for name,i in BODY.items()}
        joints['hips']=[0.,1.,0.];joints['neck']=[(point(11)[a]+point(12)[a])/2 for a in range(3)]
        for side,key in [('left','LLM'),('right','RLM')]:
            hand=frame.get(key)
            if not isinstance(hand,list) or len(hand)!=21: continue
            anchor=joints[f'{side}_wrist'];wrist=hand[0]
            for finger,indices in FINGERS.items():
                for number,i in enumerate(indices,1):
                    p=hand[i]
                    joints[f'{side}_{finger}_{number}']=[anchor[0]+(p['x']-wrist['x'])*scale,anchor[1]-(p['y']-wrist['y'])*scale,anchor[2]-(p['z']-wrist['z'])*scale]
            if side=='left':left+=1
            else:right+=1
        if timing:
            delta=frame.get('dt',0)
            if not isinstance(delta,(int,float)) or not math.isfinite(delta) or delta<0: raise ValueError('Intervalle Animics invalide.')
            time+=delta
            stamp=round(time*1000)
        else: stamp=round(index*1000/fps)
        # Animics dt is a frame delta in seconds. Duplicate initial timestamps
        # occur in some exports and are omitted instead of creating zero spans.
        if frames and stamp<=frames[-1]['time_ms']: continue
        frames.append({'time_ms':stamp,'joints':joints})
    if not frames: raise ValueError('Aucune image utilisable.')
    first_time=frames[0]['time_ms']
    for frame in frames: frame['time_ms']-=first_time
    duration=frames[-1]['time_ms']+round(1000/fps)
    last=deepcopy(frames[-1]);last['time_ms']=duration;frames.append(last)
    return validate_motion({'rig_version':'signaculture-v1','fps':fps,'duration_ms':duration,'joints':list(BODY)+['hips','neck'],
        'frames':frames,'synthetic':False,'capture_method':'animics-keyframe-import',
        'quality':{'sampled_frames':len(data),'pose_coverage':1.,'left_hand_coverage':round(left/len(data),3),'right_hand_coverage':round(right/len(data),3),'face_coverage':0},
        'warnings':['Import de positions Animics ; calibration de profondeur approximative.']+([] if timing else ['Horodatage absent : cadence supposée de 30 images par seconde.'])})
