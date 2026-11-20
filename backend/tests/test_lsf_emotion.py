"""Voice emotion -> speed and size of the LSF engines' manual signing (never the face); model folder settings.
Authored fixtures, no interpreter data and no model inference."""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.services import capture_library
from backend.app.services.lsf_emotion import ARM_HAND, apply_emotion, emotion_params
from backend.tests.test_lsf_pose_convert import fixture

client = TestClient(main.app)


def test_params_follow_dominant_label_and_confidence():
    angry = emotion_params({"angry": 0.9, "neutral": 0.1})
    assert angry["label"] == "angry" and angry["speed"] == pytest.approx(1.135) and angry["amplitude"] == pytest.approx(1.18)
    assert emotion_params({"sad": 1.0})["speed"] < 1 and emotion_params({"sad": 1.0})["amplitude"] < 1
    for neutralish in ({"neutral": 0.7, "happy": 0.3}, {"unknown": 1.0}, {"other": 0.9}):
        p = emotion_params(neutralish)
        assert p["label"] == "neutral" and p["speed"] == 1 and p["amplitude"] == 1
    assert emotion_params(None) is None and emotion_params({"foo": 1.0}) is None
    assert emotion_params({"angry": 1.0}, strength=0.0)["speed"] == 1


def test_apply_changes_hands_and_timing_not_face():
    x = np.zeros((100, 157, 2), np.float32)
    x[:, 16, 0] = np.sin(np.linspace(0, 6, 100))
    x[:, 65:, 1] = 0.5
    y = apply_emotion(x, emotion_params({"angry": 1.0}))
    assert len(y) == round(100 / 1.15)
    assert np.ptp(y[:, 16, 0]) == pytest.approx(1.2 * np.ptp(x[:, 16, 0]), rel=0.05)
    assert np.allclose(y[:, 65:, 1], 0.5) and np.allclose(y[:, 65:, 0], 0)          # face untouched
    assert set(range(65, 157)).isdisjoint(ARM_HAND)


def test_generate_applies_emotion_and_moves_markers(monkeypatch):
    def fake_generate(text, engine, seed=0, groups=None):
        return fixture(T=100), {"engine": engine, "lemmas": ["x"], "unknown": [],
                                "sign_spans": [{"start_frame": 10, "end_frame": 90, "lemma": "x", "clause": 0}]}

    monkeypatch.setattr(main.lsf_engines, "generate", fake_generate)
    calm = client.post("/lsf/generate", json={"text": "Tu ne viens pas.", "engine": "b2", "order": "lsf"}).json()
    angry = client.post("/lsf/generate", json={"text": "Tu ne viens pas.", "engine": "b2", "order": "lsf", "emotion": {"angry": 1.0}}).json()
    assert "emotion" not in calm["generation"]
    e = angry["generation"]["emotion"]
    assert e["label"] == "angry" and e["validated"] is False
    assert angry["duration_ms"] < calm["duration_ms"]                                   # faster signing
    assert angry["nonmanual"][0]["end_ms"] < calm["nonmanual"][0]["end_ms"]             # markers follow the retiming


def test_model_folders_from_env(monkeypatch, tmp_path):
    asr, emo = tmp_path / "turbo", tmp_path / "emo-large"
    asr.mkdir(); emo.mkdir()
    for name in ("config.json", "model.bin", "tokenizer.json", "vocabulary.json"):
        (asr / name).write_text("{}")
    (emo / "model.pt").write_bytes(b"x")
    monkeypatch.setenv("SIGNACULTURE_ASR_MODEL_DIR", str(asr))
    monkeypatch.setenv("SIGNACULTURE_EMOTION_MODEL_DIR", str(emo))
    monkeypatch.setenv("SIGNACULTURE_ASR_MODEL_ID", "mobiuslabsgmbh/faster-whisper-large-v3-turbo")
    assert capture_library.asr_model_ready() and capture_library.emotion_model_ready()      # vocabulary.json accepted
    assert capture_library.asr_model_id() == "mobiuslabsgmbh/faster-whisper-large-v3-turbo"
    assert capture_library.emotion_model_id() == "emo-large"
    monkeypatch.delenv("SIGNACULTURE_ASR_MODEL_DIR")
    monkeypatch.delenv("SIGNACULTURE_ASR_MODEL_ID")
    assert capture_library.asr_model_id() == "Systran/faster-whisper-small"


def test_emotion_segments_needs_model(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_EMOTION_MODEL_DIR", str(tmp_path / "none"))
    r = client.post("/analyze-audio/emotion-segments", files={"file": ("f.mp4", b"x")}, data={"cues": '[{"start_ms":0,"end_ms":1000}]'})
    assert r.status_code == 503
    assert client.post("/analyze-audio/emotion-segments", files={"file": ("f.mp4", b"x")}, data={"cues": "nope"}).status_code == 422
