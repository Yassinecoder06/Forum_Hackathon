from backend.app.services.sign_core import core_window, clip_windows
import json


def clip(heights, step=40):
    """Rest -> sign -> rest, as right-wrist heights (metres) sampled every `step` ms."""
    frames = [{'time_ms': i * step, 'joints': {'left_wrist': [0.15, 1.0, 0.1], 'right_wrist': [-0.15, h, 0.2]}}
              for i, h in enumerate(heights)]
    return {'duration_ms': frames[-1]['time_ms'], 'frames': frames}


def test_core_skips_opening_and_closing_rest_on_captured_frames():
    motion = clip([1.0] * 10 + [1.2, 1.4] + [1.5] * 8 + [1.4, 1.2] + [1.0] * 10)  # 480 ms of signing
    start, end = core_window(motion)
    times = {f['time_ms'] for f in motion['frames']}
    assert start in times and end in times
    # Peak travel 0.5 m: the core runs from the rising frames to the falling frames, rests excluded.
    assert 10 * 40 <= start <= 11 * 40 and 20 * 40 <= end <= 21 * 40


def test_absent_motion_keeps_the_whole_clip():
    assert core_window(clip([1.0] * 20)) == (0, 19 * 40)


def test_short_motion_is_widened_around_its_peak_not_to_the_rests():
    start, end = core_window(clip([1.0] * 10 + [1.1, 1.1] + [1.0] * 10))
    assert end - start >= 300 and 0 < start < 10 * 40 and 11 * 40 < end < 21 * 40


def test_idle_drift_before_the_sign_is_not_part_of_the_core():
    # 5 cm of standing drift, then a 40 cm sign: the core starts with the sign, not the drift.
    start, _ = core_window(clip([1.0] * 5 + [1.05] * 10 + [1.2, 1.4] + [1.4] * 10 + [1.0] * 8))
    assert start >= 15 * 40


def test_sentence_keeps_approach_and_release(tmp_path):
    paths = []
    for name in 'ab':
        path = tmp_path / f'{name}.json'
        path.write_text(json.dumps(clip([1.0] * 20 + [1.3] * 12 + [1.0] * 20)))
        paths.append(path)
    (a_start, a_end), (b_start, b_end) = clip_windows(paths)
    core_start, core_end = core_window(json.loads(paths[0].read_text()))
    assert a_start < core_start and a_end == core_end
    assert b_start == core_start and b_end > core_end
