"""Local French speech recognition with a local faster-whisper model (default: the pinned small model;
SIGNACULTURE_ASR_MODEL_DIR / _ID / _REVISION describe another local copy, e.g. large-v3-turbo)."""
import argparse
import json
import os
from pathlib import Path

_CUSTOM = bool(os.getenv("SIGNACULTURE_ASR_MODEL_DIR"))
MODEL_REVISION = os.getenv("SIGNACULTURE_ASR_MODEL_REVISION") or ("local" if _CUSTOM else "536b0662742c02347bc0e980a01041f333bce120")
MODEL_ID = os.getenv("SIGNACULTURE_ASR_MODEL_ID") or (Path(os.environ["SIGNACULTURE_ASR_MODEL_DIR"]).name if _CUSTOM else "Systran/faster-whisper-small")


def _samples(audio: Path):
    """The API always passes 16 kHz mono 16-bit PCM WAV (ffmpeg): read it directly as float32 samples, so
    transcription does not depend on PyAV's decoder (whose API changes between versions). Other files fall
    back to faster-whisper's own decoding."""
    import wave

    import numpy as np

    try:
        with wave.open(str(audio)) as stream:
            if stream.getframerate() == 16000 and stream.getnchannels() == 1 and stream.getsampwidth() == 2:
                return np.frombuffer(stream.readframes(stream.getnframes()), dtype="<i2").astype("float32") / 32768.0
    except (wave.Error, EOFError):
        pass
    return str(audio.resolve())


def load_model(model_path: Path):
    """-> (WhisperModel, device, compute_type): CUDA when CTranslate2 sees a usable GPU, else CPU int8."""
    import ctranslate2
    from faster_whisper import WhisperModel

    model_path = model_path.resolve()
    required = ("config.json", "model.bin", "tokenizer.json")
    missing = [name for name in required if not (model_path / name).is_file()]
    if not any((model_path / name).is_file() for name in ("vocabulary.txt", "vocabulary.json")):
        missing.append("vocabulary.txt or vocabulary.json")
    if missing:
        raise FileNotFoundError("ASR model files are missing: " + ", ".join(missing))

    device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        return WhisperModel(str(model_path), device=device, compute_type=compute_type, cpu_threads=4), device, compute_type
    except (RuntimeError, ValueError):
        if device != "cuda":
            raise
        # Some installations expose CUDA devices without compatible CTranslate2
        # kernels. Fall back to CPU rather than failing the audio analysis.
        return WhisperModel(str(model_path), device="cpu", compute_type="int8", cpu_threads=4), "cpu", "int8"


def run(audio: Path, model_path: Path, output: Path) -> dict:
    from faster_whisper import WhisperModel

    model, device, compute_type = load_model(model_path)
    model_path = model_path.resolve()

    try:
        segments = _decode(model, audio)
    except RuntimeError:
        if device != "cuda":
            raise
        # CUDA can be visible while its runtime libraries (e.g. cuBLAS) are missing; that only
        # fails at inference time. Retry the whole decode on CPU.
        device, compute_type = "cpu", "int8"
        model = WhisperModel(
            str(model_path), device=device, compute_type=compute_type, cpu_threads=4
        )
        segments = _decode(model, audio)
    result = {
        "text": " ".join(segment["text"] for segment in segments).strip(),
        "language": "fr",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "device": device,
        "compute_type": compute_type,
        "segments": segments,
    }
    output.write_text(json.dumps(result, ensure_ascii=False, allow_nan=False))
    return result


def _decode(model, audio: Path) -> list:
    generated, _ = model.transcribe(
        _samples(audio),  # decoded here: some faster-whisper/PyAV pairs fail to open files themselves
        language="fr",
        task="transcribe",
        beam_size=3,
        vad_filter=True,
        condition_on_previous_text=False,
        word_timestamps=True,
    )
    segments = []
    for segment in generated:
        if not segment.text.strip():
            continue
        words = []
        for word in segment.words or []:
            text = word.word.strip()
            if not text:
                continue
            start = max(0, round(word.start * 1000))
            words.append({
                "start_ms": start,
                "end_ms": max(start + 1, round(word.end * 1000)),
                "text": text,
                "probability": round(min(1.0, max(0.0, float(word.probability))), 4),
            })
        segments.append({
            "start_ms": max(0, round(segment.start * 1000)),
            "end_ms": max(1, round(segment.end * 1000)),
            "text": segment.text.strip(),
            "words": words,
        })
    return segments


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("model", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.audio, args.model, args.output)
