"""Find the active part ("core") of a captured lexical sign clip.

Dictionary recordings start and end with the hands at rest. Joining whole clips makes the
avatar drop its hands between every sign, so the bridge between signs only ever connects two
rest poses. The core is where either wrist has left that clip's own opening (or closing) rest
position. Captured frames are never modified: plans only record which time window to play.
"""
from functools import lru_cache
import json
import math
from pathlib import Path

MOVE_M = 0.04      # below this peak travel the clip has no detectable sign movement
CORE_SHARE = 0.4   # the core starts/ends once a wrist covers 40 % of the sign's own peak travel
                   # from rest; signers' idle drift is 3-8 cm, sign sizes range from ~10 to ~50 cm
PAD_FRAMES = 0     # the approach from / release to rest is replaced by the bridge
MIN_CORE_MS = 300  # very short cores are widened around their peak, never back to the whole clip
LEAD_IN_MS = 400   # sentence start: keep part of the first sign's approach from rest
LEAD_OUT_MS = 600  # sentence end: keep part of the last sign's return to rest
# Every window boundary is snapped to a captured frame time, so playback never resamples there.


def _wrists(frame):
    joints = frame.get('joints', {})
    return {side: joints[f'{side}_wrist'] for side in ('left', 'right') if f'{side}_wrist' in joints}


def _travel(frame, reference):
    current = _wrists(frame)
    return max((math.dist(current[s], reference[s]) for s in current if s in reference), default=0.0)


def core_indices(frames):
    """(first, last) frame index of the signing core; whole clip when undetectable."""
    last = len(frames) - 1
    if len(frames) < 3:
        return 0, max(0, last)
    opening, closing = _wrists(frames[0]), _wrists(frames[-1])
    from_open = [_travel(f, opening) for f in frames]
    to_close = [_travel(f, closing) for f in frames]
    if max(from_open) < MOVE_M or max(to_close) < MOVE_M:
        return 0, last
    start_at, end_at = max(MOVE_M, CORE_SHARE * max(from_open)), max(MOVE_M, CORE_SHARE * max(to_close))
    i = next(k for k, d in enumerate(from_open) if d >= start_at)
    j = next(k for k in range(last, -1, -1) if to_close[k] >= end_at)
    if j < i:  # opening and closing rests disagree: fall back to the most active stretch
        i = j = max(range(len(frames)), key=lambda k: min(from_open[k], to_close[k]))
    i, j = max(0, i - PAD_FRAMES), min(last, j + PAD_FRAMES)
    while frames[j]['time_ms'] - frames[i]['time_ms'] < MIN_CORE_MS and (i > 0 or j < last):
        i, j = max(0, i - 1), min(last, j + 1)  # widen evenly around the peak
    return i, j


def core_window(motion):
    """(start_ms, end_ms) of the signing core of one motion file, on captured frame times."""
    frames = motion.get('frames') or []
    if not frames:
        return 0, int(motion.get('duration_ms') or 0)
    i, j = core_indices(frames)
    return int(frames[i]['time_ms']), int(frames[j]['time_ms'])


@lru_cache(maxsize=4096)
def _cached(path, mtime):
    frames = json.loads(Path(path).read_text(encoding='utf-8')).get('frames') or []
    times = tuple(int(f['time_ms']) for f in frames)
    return times, core_indices(frames)


def clip_windows(paths):
    """Windows for a sequence of motion files: inner boundaries trimmed to the cores, while the
    sentence keeps a short approach at the start and a short release at the end."""
    found = [_cached(str(Path(p)), Path(p).stat().st_mtime) for p in paths]
    windows = [[times[i], times[j]] for times, (i, j) in found]
    if windows:
        times, (i, _) = found[0]
        windows[0][0] = next(t for t in times if t >= times[i] - LEAD_IN_MS)
        times, (_, j) = found[-1]
        windows[-1][1] = next(t for t in reversed(times) if t <= times[j] + LEAD_OUT_MS)
    return [tuple(w) for w in windows]
