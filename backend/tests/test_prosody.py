"""Word-level prominence, pace and word-aligned sign timing (no ASR model needed)."""
from types import SimpleNamespace
from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services.animation_timing import EMPHASIS_HOLD_MS, RATE_MAX, RATE_MIN, word_timing
from backend.app.services.prosody import align_tokens, analyse_prosody, segment_prosody, syllables


def frames_for(words, loud=None, high=None, base_energy=0.05, base_pitch=200.0):
    """40 ms frames covering each word; the 'loud'/'high' word gets +8 dB / +4 semitones."""
    frames = []
    for i, (_, start, end) in enumerate(words):
        energy = base_energy * (2.5 if i == loud else 1.0)
        pitch = base_pitch * (2 ** (4 / 12) if i == high else 1.0)
        for t in range(start, end, 40):
            frames.append(dict(start_ms=t, end_ms=min(end, t + 40), pitch_hz=pitch, energy=energy, is_pause=False))
    return frames


def words_of(spec):
    return [dict(text=w, start_ms=s, end_ms=e) for w, s, e in spec]


SENTENCE = [('Bonjour', 0, 450), ('je', 500, 620), ('suis', 650, 820), ('content', 850, 1250), ('de', 1280, 1380),
            ('te', 1400, 1500), ('voir', 1520, 1850)]


def test_french_syllable_estimate():
    assert [syllables(w) for w in ['bonjour', 'bienvenue', 'culture', 'musée', 'merci', 'art', 'travaille']] == [2, 3, 2, 2, 2, 1, 2]


def test_louder_higher_word_is_the_only_emphasis():
    result = analyse_prosody(frames_for(SENTENCE, loud=3, high=3), words_of(SENTENCE))
    assert [w['text'] for w in result['words'] if w['emphasized']] == ['content']
    assert result['pace_syl_per_s'] > 0


def test_flat_speech_has_no_emphasis_despite_normalisation():
    result = analyse_prosody(frames_for(SENTENCE), words_of(SENTENCE))
    assert not any(w['emphasized'] for w in result['words'])


def test_contrastive_stress_moves_with_the_voice():
    first = analyse_prosody(frames_for(SENTENCE, loud=0, high=0), words_of(SENTENCE))
    later = analyse_prosody(frames_for(SENTENCE, loud=6, high=6), words_of(SENTENCE))
    assert [w['text'] for w in first['words'] if w['emphasized']] == ['Bonjour']
    assert [w['text'] for w in later['words'] if w['emphasized']] == ['voir']


def test_alignment_survives_an_edited_transcript():
    words = words_of([('bonjour', 0, 400), ('euh', 420, 500), ('merci', 520, 900)])
    assert align_tokens(['Bonjour', ',', 'merci', '!'], words) == [0, None, 2, None]
    prosody = dict(pace_syl_per_s=4, words=[dict(w, prominence=p, emphasized=e, syl_per_s=4, liveliness=.5)
                                             for w, p, e in zip(words, [1.6, 0, -1], [True, False, False])])
    per_sign = segment_prosody([(0, 1), (2, 3)], ['Bonjour', ',', 'merci', '!'], prosody)
    assert per_sign[0]['emphasis'] == 0.8 and per_sign[0]['word_start_ms'] == 0
    assert per_sign[1]['emphasis'] == 0 and per_sign[1]['word_start_ms'] == 520


def test_word_timing_follows_words_clamps_rate_and_holds_emphasis():
    motions = [SimpleNamespace(duration_ms=800), SimpleNamespace(duration_ms=800)]
    audio = SimpleNamespace(duration_ms=6000)
    per_sign = [dict(word_start_ms=100, word_end_ms=400, emphasis=1.0, liveliness=.5),
                dict(word_start_ms=5000, word_end_ms=5400, emphasis=0.0, liveliness=.5)]
    (first, second), total = word_timing(motions, per_sign, audio)
    assert first[0] == 100 and second[0] == 5000
    assert first[2] == EMPHASIS_HOLD_MS
    # Long gap: slowed only down to RATE_MIN, plus emphasis slowdown and hold.
    assert first[1] == round(800 / RATE_MIN * 1.12) + EMPHASIS_HOLD_MS
    fast = [dict(word_start_ms=0, word_end_ms=100, emphasis=0, liveliness=0),
            dict(word_start_ms=300, word_end_ms=400, emphasis=0, liveliness=0)]
    (a, b), _ = word_timing(motions, fast, audio)
    assert a[1] == round(800 / RATE_MAX) and b[0] >= a[0] + a[1] + 200
    assert word_timing(motions, [None, None], audio) is None


def test_compose_uses_word_timing_and_emphasis(monkeypatch, tmp_path):
    monkeypatch.setenv('SIGNACULTURE_DB_PATH', str(tmp_path / 'state.sqlite'))
    words = [dict(text='Bonjour', start_ms=300, end_ms=900, prominence=1.8, emphasized=True, syl_per_s=3.3, liveliness=.6),
             dict(text='ça', start_ms=950, end_ms=1100, prominence=-0.5, emphasized=False, syl_per_s=6.6, liveliness=.6),
             dict(text='va', start_ms=1120, end_ms=1400, prominence=-0.7, emphasized=False, syl_per_s=3.5, liveliness=.6)]
    audio = dict(duration_ms=3000, frames=[], emotion_hypothesis=None, model_version='test', warnings=[],
                 prosody=dict(pace_syl_per_s=4.1, words=words))
    plan = TestClient(app).post('/compose-animation', json={'text': 'Bonjour ça va', 'audio_features': audio}).json()
    assert plan['timing_mode'] == 'audio_words'
    segment = plan['segments'][0]
    assert segment['start_ms'] == 300 and segment['emphasis'] == 0.9 and segment['hold_ms'] == round(EMPHASIS_HOLD_MS * 0.9)
    assert segment['liveliness'] == 0.6
