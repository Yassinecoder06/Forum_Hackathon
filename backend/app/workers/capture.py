"""CPU video capture worker. Runs in .venv-motion, independently of the API runtime."""
import argparse
import gzip
from copy import deepcopy
import json
import math
from pathlib import Path
from types import SimpleNamespace
import cv2
import mediapipe as mp
import numpy as np
if __package__:
    from .hand_tracking import assign_hands
else:
    from hand_tracking import assign_hands

BODY = {'head': 0, 'left_shoulder': 11, 'right_shoulder': 12,
        'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16}
FINGERS = {'thumb': [1, 2, 3, 4], 'index': [5, 6, 7, 8],
           'middle': [9, 10, 11, 12], 'ring': [13, 14, 15, 16], 'pinky': [17, 18, 19, 20]}

def pose_frame(result, width, height):
    image = result.pose_landmarks.landmark
    world = result.pose_world_landmarks.landmark if result.pose_world_landmarks else None
    shoulder_px = math.hypot((image[11].x-image[12].x)*width, (image[11].y-image[12].y)*height)
    if shoulder_px < 3:
        raise ValueError('Épaules non détectées')
    pixel_scale = .4 / shoulder_px
    if world:
        origin = np.mean([[world[i].x,world[i].y,world[i].z] for i in (23,24)], axis=0)
        def body(i):
            point = world[i]
            value = np.array([point.x,point.y,point.z])-origin
            return [float(value[0]), float(1-value[1]), float(-value[2])]
    else:
        origin = np.mean([[image[i].x,image[i].y,image[i].z] for i in (23,24)], axis=0)
        def body(i):
            point = image[i]
            return [(point.x-origin[0])*width*pixel_scale, 1-(point.y-origin[1])*height*pixel_scale, -(point.z-origin[2])*width*pixel_scale]
    joints = {name: body(index) for name,index in BODY.items()}
    joints['hips'] = [0.,1.,0.]
    joints['neck'] = np.mean([body(11),body(12)],axis=0).tolist()
    # Anchor each hand's relative depth to its anatomical pose wrist. MediaPipe
    # hand z is wrist-relative; subtracting the body hip z here would be wrong.
    hands = {}
    for side,wrist in [('left',15),('right',16)]:
        hand = getattr(result,f'{side}_hand_landmarks')
        hands[side] = hand is not None
        if not hand:
            continue
        anchor = joints[f'{side}_wrist']
        pts = hand.landmark
        for finger,indices in FINGERS.items():
            for number,i in enumerate(indices,1):
                joints[f'{side}_{finger}_{number}'] = [anchor[0]+(pts[i].x-pts[0].x)*width*pixel_scale,
                    anchor[1]-(pts[i].y-pts[0].y)*height*pixel_scale,
                    anchor[2]-(pts[i].z-pts[0].z)*width*pixel_scale]
    face = result.face_landmarks
    facial = None
    if face:
        pts = face.landmark
        # Dimensionless measurements, not grammatical interpretation.
        eye_width = max(abs(pts[33].x-pts[263].x),.001)
        facial = {'mouth_open': abs(pts[13].y-pts[14].y)/eye_width,
                  'brow_left': abs(pts[105].y-pts[159].y)/eye_width,
                  'brow_right': abs(pts[334].y-pts[386].y)/eye_width}
    if not all(math.isfinite(n) for xyz in joints.values() for n in xyz):
        raise ValueError('Coordonnées invalides')
    return joints, hands, facial


def extract(source, target, progress, raw_target=None):
    cap = cv2.VideoCapture(str(source))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if not cap.isOpened() or not math.isfinite(fps) or not 1<=fps<=240 or count<=0:
        raise ValueError('Cette vidéo ne peut pas être décodée.')
    if count/fps > 60:
        raise ValueError('Le clip doit durer au maximum 60 secondes.')
    stride = max(1,math.ceil(fps/20))
    frames, missing, left, right, faces, processed = [],0,0,0,0,0
    raw_frames=[]
    previous_hands={}
    def raw_points(landmarks,world=False):
        if landmarks is None:return None
        values=[]
        for point in landmarks.landmark:
            item={'x':float(point.x),'y':float(-point.y if world else point.y),'z':float(-point.z if world else point.z)}
            if point.HasField('visibility'):item['visibility']=float(point.visibility)
            values.append(item)
        return values
    def report(value):
        temp = progress.with_suffix('.tmp')
        temp.write_text(json.dumps({'progress':value}))
        temp.replace(progress)
    try:
        with mp.solutions.holistic.Holistic(model_complexity=1, min_detection_confidence=.5, min_tracking_confidence=.5) as model, mp.solutions.hands.Hands(max_num_hands=2, min_detection_confidence=.35, min_tracking_confidence=.4) as hand_model:
            index = 0
            while True:
                ok, image = cap.read()
                if not ok: break
                if index % stride:
                    index+=1
                    continue
                if max(image.shape[:2])>960:
                    image=cv2.resize(image,None,fx=960/max(image.shape[:2]),fy=960/max(image.shape[:2]))
                rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                result=model.process(rgb)
                hand_sources={side:'holistic' if getattr(result,f'{side}_hand_landmarks') else 'missing' for side in ('left','right')}
                if result.pose_landmarks and (not result.left_hand_landmarks or not result.right_hand_landmarks):
                    detected=hand_model.process(rgb)
                    sides={'left':result.left_hand_landmarks,'right':result.right_hand_landmarks}
                    if detected.multi_hand_landmarks:
                        wrists={side:(result.pose_landmarks.landmark[i].x,result.pose_landmarks.landmark[i].y) for side,i in [('left',15),('right',16)]}
                        points=[(hand.landmark[0].x,hand.landmark[0].y) for hand in detected.multi_hand_landmarks]
                        assignments=assign_hands(points,wrists,{s:h is not None for s,h in sides.items()},previous_hands)
                        for side,chosen in assignments.items():
                            sides[side]=detected.multi_hand_landmarks[chosen]
                            hand_sources[side]='hands_fallback'
                    result=SimpleNamespace(pose_landmarks=result.pose_landmarks,pose_world_landmarks=result.pose_world_landmarks,
                        face_landmarks=result.face_landmarks,left_hand_landmarks=sides['left'],right_hand_landmarks=sides['right'])
                previous_hands={side:(hand.landmark[0].x,hand.landmark[0].y) for side in ('left','right') if (hand:=getattr(result,f'{side}_hand_landmarks')) is not None}
                if raw_target is not None:
                    # Animics Keyframe recorder names/axis convention, pinned reference
                    # upf-gti/animics daf09928 src/Mediapipe.js processLandmarks/recording.
                    # Holistic does not expose independent world-hand points: keep them null.
                    raw_frames.append({'dt':0. if not raw_frames else stride/fps,'time_ms':round(index*1000/fps),
                        'image_width':image.shape[1],'image_height':image.shape[0],
                        'PLM':raw_points(result.pose_landmarks),'PWLM':raw_points(result.pose_world_landmarks,True),
                        'LLM':raw_points(result.left_hand_landmarks),'RLM':raw_points(result.right_hand_landmarks),
                        'LWLM':None,'RWLM':None,'FLM':raw_points(result.face_landmarks)})
                processed+=1
                if result.pose_landmarks:
                    try:
                        joints,hands,face=pose_frame(result,image.shape[1],image.shape[0])
                        frames.append({'time_ms':round(index*1000/fps),'joints':joints,'face':face,
                            'hand_tracking':{s:{'observed':hands[s],'source':hand_sources[s]} for s in ('left','right')}})
                        left+=int(hands['left']);right+=int(hands['right']);faces+=int(face is not None)
                    except ValueError: missing+=1
                else: missing+=1
                index+=1
                if processed%5==0: report(min(95,round(index/count*95)))
    finally: cap.release()
    if not frames:
        raise ValueError('Aucune personne détectée. Utilisez un plan fixe, de face, avec le buste et les mains visibles.')
    duration=round(index*1000/fps)
    if frames[0]['time_ms']>0:
        first=deepcopy(frames[0]);first['time_ms']=0;frames.insert(0,first)
    if frames[-1]['time_ms']<duration:
        last=deepcopy(frames[-1]);last['time_ms']=duration;frames.append(last)
    joint_names=sorted(set().union(*(f['joints'] for f in frames)))
    # Do not invent fingers during occlusion. Sparse finger keys are intentional;
    # the player hides a hand when its landmarks are missing.
    quality={'sampled_frames':processed,'pose_frames':processed-missing,
        'pose_coverage':round((processed-missing)/processed,3),
        'left_hand_coverage':round(left/processed,3),'right_hand_coverage':round(right/processed,3),
        'face_coverage':round(faces/processed,3),'missing_frames':missing}
    motion={'rig_version':'signaculture-v1','fps':fps/stride,'duration_ms':duration,
        'joints':list(BODY)+['hips','neck'],'tracked_joints':joint_names,'frames':frames,
        'quality':quality,'synthetic':False,'capture_method':'mediapipe-holistic-hands-fallback-0.10.21-assignment-v2',
        'warnings':['Profondeur monoculaire approximative. Les occultations nécessitent une revue.']}
    target.write_text(json.dumps(motion,allow_nan=False))
    if raw_target is not None:
        with gzip.open(raw_target,'wt',encoding='utf-8') as handle:json.dump(raw_frames,handle,allow_nan=False,separators=(',',':'))
    report(100)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);parser.add_argument('target',type=Path);parser.add_argument('progress',type=Path);parser.add_argument('--raw-animics',type=Path)
    args=parser.parse_args()
    try: extract(args.source,args.target,args.progress,args.raw_animics)
    except Exception as exc:
        args.target.with_suffix('.error').write_text(str(exc));raise
