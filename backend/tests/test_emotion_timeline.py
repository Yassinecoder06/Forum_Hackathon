"""Windowing and contract checks for the time-varying emotion layer (no Torch needed)."""
import pytest
from pydantic import ValidationError
from backend.app.workers.emotion_timeline import build_timeline, voiced_span, window_bounds
from backend.app.schemas.models import AudioFeatures

SCORES = {'angry': 0, 'disgusted': 0, 'fearful': 0, 'happy': .7, 'neutral': .3,
          'other': 0, 'sad': 0, 'surprised': 0, 'unknown': 0}


def test_windows_overlap_and_reach_the_end():
    bounds = window_bounds(16000 * 5 + 300, 16000)
    assert bounds[0] == (0, 32000)
    assert all(b[0] - a[0] == 16000 for a, b in zip(bounds, bounds[1:-1]))
    assert bounds[-1][1] == 16000 * 5 + 300
    assert all(end - start == 32000 for start, end in bounds)


def test_voiced_span_trims_padded_silence():
    # 50 ms frames at 16 kHz = 800 samples; speech in frames 28..63 of 85 (CaFE-like padding).
    rms = [0.0] * 28 + [0.05] * 36 + [0.0] * 21
    start, end = voiced_span(rms, 16000, 85 * 800)
    assert start == 28 * 800 - 1600 and end == 64 * 800 + 1600
    assert voiced_span([0.0, 0.005], 16000, 1600) is None
    assert voiced_span([0.05], 16000, 500) == (0, 500)


def test_short_audio_is_a_single_window_and_empty_has_none():
    assert window_bounds(8000, 16000) == [(0, 8000)]
    assert window_bounds(0, 16000) == []


def test_silent_windows_are_skipped_and_times_are_ms():
    timeline = build_timeline([(0, 32000), (16000, 48000)], 16000, [None, SCORES])
    assert timeline == [{'start_ms': 1000, 'end_ms': 3000, 'scores': SCORES}]
    with pytest.raises(ValueError):
        build_timeline([(0, 32000)], 16000, [])


def features(timeline):
    return dict(duration_ms=3000, frames=[], emotion_hypothesis=SCORES,
                emotion_timeline=timeline, model_version='test', warnings=[])


def test_contract_accepts_timeline_and_rejects_invalid_windows():
    ok = AudioFeatures(**features([{'start_ms': 0, 'end_ms': 2000, 'scores': SCORES},
                                   {'start_ms': 1000, 'end_ms': 3000, 'scores': SCORES}]))
    assert len(ok.emotion_timeline) == 2
    assert AudioFeatures(**{k: v for k, v in features(None).items() if k != 'emotion_timeline'}).emotion_timeline is None
    for bad in ([{'start_ms': 0, 'end_ms': 4000, 'scores': SCORES}],
                [{'start_ms': 1000, 'end_ms': 3000, 'scores': SCORES}, {'start_ms': 0, 'end_ms': 2000, 'scores': SCORES}],
                [{'start_ms': 0, 'end_ms': 2000, 'scores': {**SCORES, 'happy': 1.5}}]):
        with pytest.raises(ValidationError):
            AudioFeatures(**features(bad))
