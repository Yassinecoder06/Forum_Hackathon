"""External audio payloads must fail validation before JSON serialization."""
import json

import pytest
from pydantic import ValidationError

from .models import AudioFeatures, AudioFrame, MotionAsset, AnimationPlan


def frame(**changes):
    return dict(start_ms=0, end_ms=10, pitch_hz=None, energy=0, is_pause=True, **changes)


def audio(frames=None):
    return dict(duration_ms=20, frames=[] if frames is None else frames,
                emotion_hypothesis=None, model_version='signal-only', warnings=[])


@pytest.mark.parametrize('field,value', [
    ('pitch_hz', 0), ('pitch_hz', -1), ('pitch_hz', float('inf')),
    ('pitch_hz', float('-inf')), ('pitch_hz', float('nan')),
    ('energy', -1), ('energy', float('inf')), ('energy', float('nan')),
    ('start_ms', 10), ('start_ms', 11), ('start_ms', -1),
])
def test_invalid_frame(field, value):
    row = frame()
    row[field] = value
    with pytest.raises(ValidationError):
        AudioFrame.model_validate(row)


@pytest.mark.parametrize('intervals', [
    [(0, 21)], [(10, 20), (0, 10)], [(0, 11), (10, 20)], [(0, 10), (0, 10)],
])
def test_invalid_timeline(intervals):
    frames = [dict(frame(), start_ms=start, end_ms=end) for start, end in intervals]
    with pytest.raises(ValidationError):
        AudioFeatures.model_validate(audio(frames))


@pytest.mark.parametrize('frames', [[], [frame()],
    [frame(), dict(frame(), start_ms=10, end_ms=20, pitch_hz=120, energy=.5, is_pause=False)],
    [dict(frame(), start_ms=5, end_ms=10)]])
def test_valid_empty_silent_voiced_and_gapped_audio(frames):
    model = AudioFeatures.model_validate(audio(frames))
    assert json.loads(model.model_dump_json())['frames'] == frames


@pytest.mark.parametrize('fps', [float('inf'), float('-inf'), float('nan'), 0])
def test_motion_fps_must_be_finite_positive(fps):
    with pytest.raises(ValidationError):
        MotionAsset.model_validate(dict(motion_id='m', gloss_id='g', path='x',
            rig_version='signaculture-v1', fps=fps, duration_ms=10, license='test',
            quality_status='pending', rights_status='pending', animation_review='pending', synthetic=True))


def test_optional_compose_audio_and_nested_rejection():
    plan = dict(plan_id='p', source_text='', target_language='LSF', segments=[],
        unknowns=[], duration_ms=0, linguistic_review='pending', animation_review='pending',
        rights_status='pending', render_allowed_for_publication=False)
    assert AnimationPlan.model_validate(plan).audio_features is None
    assert AnimationPlan.model_validate(dict(plan, audio_features=None)).audio_features is None
    invalid = frame()
    invalid['energy'] = float('inf')
    with pytest.raises(ValidationError):
        AnimationPlan.model_validate(dict(plan, audio_features=audio([invalid])))
