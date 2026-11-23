"""Live speech recognition: one persistent Whisper worker (workers/transcribe_server.py) in the audio runtime, so a
spoken phrase is transcribed in about a second instead of reloading the model for every chunk. Local only.

The worker runs in its own process (as the batch workers do): the audio stack never shares the API process with
the LSF engines' PyTorch, and a crash there cannot take the API down; it is restarted on the next request."""
from __future__ import annotations

import atexit
import json
import os
import queue
import subprocess
import threading
from pathlib import Path

from .capture_library import ROOT, asr_model_dir, asr_model_ready, runtime_python, storage


class LiveUnavailable(RuntimeError):
    pass


class LiveASR:
    def __init__(self):
        self.proc: subprocess.Popen | None = None
        self.lines: queue.Queue = queue.Queue()
        self.info: dict | None = None
        self.lock = threading.Lock()

    def _start(self):
        python, model = runtime_python("audio"), asr_model_dir()
        if not python.is_file() or not asr_model_ready(model):
            raise LiveUnavailable("Transcription locale indisponible : modèle Whisper ou environnement audio absent.")
        log = open(storage() / "live_asr.log", "a", encoding="utf-8")
        self.proc = subprocess.Popen([str(python), str(ROOT / "backend/app/workers/transcribe_server.py"), str(model)],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True, encoding="utf-8",
                                     cwd=ROOT, env=os.environ.copy())
        self.lines = queue.Queue()
        threading.Thread(target=self._read, args=(self.proc, self.lines), daemon=True).start()
        self.info = self._next(timeout=240)            # model load (first time: a few seconds)
        if not self.info.get("ready"):
            self.stop()
            raise LiveUnavailable(self.info.get("error") or "Le service de transcription n'a pas démarré.")

    @staticmethod
    def _read(proc, lines):
        for line in proc.stdout:
            lines.put(line)
        lines.put(None)

    def _next(self, timeout):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.stop()
            raise LiveUnavailable("La transcription n'a pas répondu à temps.") from None
        if line is None:
            self.stop()
            raise LiveUnavailable("Le service de transcription s'est arrêté.")
        return json.loads(line)

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def warm(self) -> dict:
        with self.lock:
            if not self._alive():
                self._start()
            return dict(self.info or {})

    def transcribe(self, wav: Path, timeout: float = 60) -> dict:
        with self.lock:
            if not self._alive():
                self._start()
            self.proc.stdin.write(json.dumps({"wav": str(wav)}) + "\n")
            self.proc.stdin.flush()
            result = self._next(timeout)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
        self.proc = None


live_asr = LiveASR()
atexit.register(live_asr.stop)
