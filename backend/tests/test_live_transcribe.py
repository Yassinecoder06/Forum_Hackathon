"""Live mode endpoints: one phrase in, text out, through the persistent worker (faked here: no model needed)."""
import io
import wave

import numpy as np
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.services import live_asr as live

client = TestClient(main.app)


def wav_bytes(seconds=0.5, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(rate)
        w.writeframes((np.sin(np.arange(int(seconds * rate)) / 8) * 8000).astype("<i2").tobytes())
    return buf.getvalue()


def test_phrase_is_passed_to_the_worker_as_16k_wav(monkeypatch):
    seen = {}

    def fake(wav, timeout=60):
        with wave.open(str(wav)) as w:
            seen.update(rate=w.getframerate(), channels=w.getnchannels())
        return {"text": "Bonjour à tous.", "segments": [], "ms": 12}

    monkeypatch.setattr(live.live_asr, "transcribe", fake)
    r = client.post("/live/transcribe", files={"file": ("phrase.wav", wav_bytes(), "audio/wav")})
    assert r.status_code == 200 and r.json()["text"] == "Bonjour à tous."
    assert seen == {"rate": 16000, "channels": 1}


def test_missing_model_gives_clear_503(monkeypatch):
    def unavailable(*a, **k):
        raise live.LiveUnavailable("Transcription locale indisponible")

    monkeypatch.setattr(live.live_asr, "warm", unavailable)
    monkeypatch.setattr(live.live_asr, "transcribe", unavailable)
    assert client.post("/live/warmup").status_code == 503
    assert client.post("/live/transcribe", files={"file": ("phrase.wav", wav_bytes(), "audio/wav")}).status_code == 503


def test_empty_upload_rejected():
    assert client.post("/live/transcribe", files={"file": ("phrase.wav", b"", "audio/wav")}).status_code == 422
