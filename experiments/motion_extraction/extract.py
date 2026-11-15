"""Extract genuine pose landmarks; requires mediapipe legacy solutions and OpenCV."""
import argparse
import json
from pathlib import Path


def normalize(points, shoulder_width_m=0.4, reference_pose=None):
    """Apply one hip/shoulder transform to pose or image-space hand points."""
    import math
    reference = points if reference_pose is None else reference_pose
    if len(reference) != 33:
        raise ValueError('Expected 33 pose landmarks')
    if not math.isfinite(shoulder_width_m) or shoulder_width_m <= 0:
        raise ValueError('Invalid shoulder width')
    if any(not math.isfinite(getattr(p, axis)) for p in list(reference)+list(points) for axis in ('x','y','z')):
        raise ValueError('Nonfinite landmark coordinates')
    left, right = reference[11], reference[12]
    scale = math.hypot(left.x-right.x, left.y-right.y)
    if scale < 1e-6:
        raise ValueError('Shoulder scale unavailable')
    origin = [(getattr(reference[23], a)+getattr(reference[24], a))/2 for a in ('x','y','z')]
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
    if not hasattr(mp, 'solutions'):
        parser.error('MediaPipe legacy solutions unavailable; install optional pinned runtime')
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
                    try:
                        frame = {'time_ms': time_ms, 'pose': normalize(raw),
                                 'visibility': [p.visibility for p in raw]}
                        for side in ('left', 'right'):
                            hand = getattr(result, f'{side}_hand_landmarks')
                            frame[f'{side}_hand'] = normalize(hand.landmark, reference_pose=raw) if hand else None
                        frames.append(frame)
                    except ValueError:
                        missing.append(time_ms)
                else:
                    missing.append(time_ms)
                index += 1
    finally:
        cap.release()
    if not frames:
        parser.error('No usable pose frames (missing, degenerate or nonfinite landmarks)')
    args.output.write_text(json.dumps({'format':'mediapipe-intermediate-v2','fps':fps,'duration_ms':round(index*1000/fps),'frames':frames,'missing_pose_ms':missing,'synthetic':False,'review_status':'pending','normalization':'hip origin; shoulder width assumed 0.4m; pose and hands share transform; depth approximate'},allow_nan=False))


if __name__ == '__main__':
    main()
