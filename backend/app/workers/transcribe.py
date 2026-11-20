"""Local French speech recognition using the pinned faster-whisper model."""
import argparse
import json
from pathlib import Path

MODEL_REVISION = "536b0662742c02347bc0e980a01041f333bce120"
MODEL_ID = "Systran/faster-whisper-small"


def run(audio: Path, model_path: Path, output: Path) -> dict:
    import ctranslate2
    from faster_whisper import WhisperModel

    model_path = model_path.resolve()
    required = ("config.json", "model.bin", "tokenizer.json", "vocabulary.txt")
    missing = [name for name in required if not (model_path / name).is_file()]
    if missing:
        raise FileNotFoundError("ASR model files are missing: " + ", ".join(missing))

    device = "cuda" if ctranslate2.get_cuda_device_count() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        model = WhisperModel(
            str(model_path), device=device, compute_type=compute_type, cpu_threads=4
        )
    except (RuntimeError, ValueError):
        if device != "cuda":
            raise
        # Some installations expose CUDA devices without compatible CTranslate2
        # kernels. Fall back to CPU rather than failing the audio analysis.
        device, compute_type = "cpu", "int8"
        model = WhisperModel(
            str(model_path), device=device, compute_type=compute_type, cpu_threads=4
        )

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
        str(audio.resolve()),
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
