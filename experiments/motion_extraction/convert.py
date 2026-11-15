"""Convert normalized MediaPipe intermediate v2 into player position frames."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path

JOINTS = {'head': 0, 'left_shoulder': 11, 'right_shoulder': 12,
          'left_elbow': 13, 'right_elbow': 14, 'left_wrist': 15, 'right_wrist': 16}


def points3(points, count):
    if not isinstance(points, list) or len(points) != count:
        raise ValueError(f'Expected {count} landmarks')
    if any(not isinstance(p, list) or len(p) != 3 or
           any(not isinstance(n, (int, float)) or isinstance(n, bool) or not math.isfinite(n) for n in p) for p in points):
        raise ValueError('Landmarks must contain finite coordinate triples')
    return points


def convert(data):
    if data.get('format') != 'mediapipe-intermediate-v2':
        raise ValueError('Expected intermediate v2; re-extract old v1 videos to normalize hands consistently')
    fps, duration = data['fps'], data['duration_ms']
    if not math.isfinite(fps) or fps <= 0 or not isinstance(duration, int) or duration <= 0:
        raise ValueError('Invalid FPS or duration')
    frames, previous = [], -1
    for frame in data['frames']:
        time = frame['time_ms']
        if not isinstance(time, int) or isinstance(time, bool) or time <= previous or time < 0 or time > duration:
            raise ValueError('Timestamps must be strictly increasing milliseconds within duration')
        previous = time
        pose = points3(frame['pose'], 33)
        joints = {name: pose[index] for name, index in JOINTS.items()}
        for name, indices in [('hips', (23, 24)), ('neck', (11, 12))]:
            joints[name] = [(pose[indices[0]][axis]+pose[indices[1]][axis])/2 for axis in range(3)]
        # Anatomical labels are retained: never flip or swap left/right.
        for side in ('left', 'right'):
            hand = frame.get(f'{side}_hand')
            if hand is not None:
                points3(hand, 21)
                # The player contract requires wrists; fingers are omitted because
                # no shared finger naming/retargeting contract has been agreed.
                joints[f'{side}_wrist'] = hand[0]
        frames.append({'time_ms': time, 'joints': joints})
    if not frames:
        raise ValueError('No usable pose frames; cannot create a player asset')
    # Synthesized boundary holds preserve observed timestamps and total duration.
    if frames[0]['time_ms'] > 0:
        frames.insert(0, {'time_ms': 0, 'joints': deepcopy(frames[0]['joints'])})
    if frames[-1]['time_ms'] < duration:
        frames.append({'time_ms': duration, 'joints': deepcopy(frames[-1]['joints'])})
    return dict(rig_version='signaculture-v1', fps=fps, duration_ms=duration,
                joints=list(frames[0]['joints']), frames=frames)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    try:
        motion = convert(json.loads(args.input.read_text()))
    except (ValueError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    args.output.write_text(json.dumps(motion, allow_nan=False, indent=2)+'\n')


if __name__ == '__main__':
    main()
