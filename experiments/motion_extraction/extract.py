"""Extract genuine pose landmarks; requires mediapipe legacy solutions and OpenCV."""
import argparse
import json
from pathlib import Path


def normalize(points, shoulder_width_m=0.4):
    """Image coordinates to hip-centered, Y-up approximate meters; not metric capture."""
    left, right = points[11], points[12]
    scale = ((left.x-right.x)**2 + (left.y-right.y)**2)**0.5
    if scale < 1e-6:
        raise ValueError('Shoulder scale unavailable')
    origin = [(getattr(points[23], a)+getattr(points[24], a))/2 for a in ('x','y','z')]
    return [[(p.x-origin[0])*shoulder_width_m/scale,
             -(p.y-origin[1])*shoulder_width_m/scale,
             -(p.z-origin[2])*shoulder_width_m/scale] for p in points]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    import cv2
    import mediapipe as mp
    cap = cv2.VideoCapture(str(args.video))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps <= 0:
        raise ValueError('Cannot open video or determine FPS')
    frames = []
    missing = []
    index = 0
    try:
        with mp.solutions.holistic.Holistic(static_image_mode=False) as model:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                result = model.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                time_ms = round(index*1000/fps)
                if result.pose_landmarks:
                    raw = result.pose_landmarks.landmark
                    frames.append({'time_ms':time_ms,'pose':normalize(raw),
                                   'visibility':[p.visibility for p in raw],
                                   'left_hand': [[p.x,p.y,p.z] for p in result.left_hand_landmarks.landmark] if result.left_hand_landmarks else None,
                                   'right_hand': [[p.x,p.y,p.z] for p in result.right_hand_landmarks.landmark] if result.right_hand_landmarks else None})
                else:
                    missing.append(time_ms)
                index += 1
    finally:
        cap.release()
    if not index:
        raise ValueError('No decoded frames')
    args.output.write_text(json.dumps({'format':'mediapipe-intermediate-v1','fps':fps,'duration_ms':round(index*1000/fps),'frames':frames,'missing_pose_ms':missing,'synthetic':False,'review_status':'pending','normalization':'hip origin; shoulder width assumed 0.4m; hands remain raw image coordinates'},allow_nan=False))


if __name__ == '__main__':
    main()
