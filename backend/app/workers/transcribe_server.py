"""Persistent French speech recognition for the live mode: the Whisper model is loaded ONCE, then each request is
one JSON line on stdin ({"wav": path}) answered by one JSON line on stdout ({"text", "segments", "ms"} or {"error"}).
Run by the API (services/live_asr.py) with the audio runtime; nothing leaves the machine.

    python transcribe_server.py MODEL_DIR"""
import json
import sys
import time
from pathlib import Path

try:
    from .transcribe import MODEL_ID, _decode, load_model
except ImportError:  # launched as a standalone script by the API
    from transcribe import MODEL_ID, _decode, load_model


def reply(obj):
    # ASCII-escaped JSON: the pipe's encoding differs between platforms (cp1252 on Windows), escapes never do.
    sys.stdout.write(json.dumps(obj, allow_nan=False) + "\n")
    sys.stdout.flush()


def main(model_dir: str):
    model, device, compute_type = load_model(Path(model_dir))
    reply({"ready": True, "model_id": MODEL_ID, "device": device, "compute_type": compute_type})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        start = time.perf_counter()
        try:
            request = json.loads(line)
            try:
                segments = _decode(model, Path(request["wav"]))
            except RuntimeError:
                if device != "cuda":
                    raise
                # CUDA visible but its runtime missing: switch to CPU for good.
                from faster_whisper import WhisperModel
                model, device, compute_type = WhisperModel(model_dir, device="cpu", compute_type="int8", cpu_threads=4), "cpu", "int8"
                segments = _decode(model, Path(request["wav"]))
            reply({"text": " ".join(s["text"] for s in segments).strip(), "segments": segments, "device": device,
                   "ms": round((time.perf_counter() - start) * 1000)})
        except Exception as exc:  # one bad request must not kill the worker
            reply({"error": f"{type(exc).__name__}: {exc}"})


if __name__ == "__main__":
    main(sys.argv[1])
