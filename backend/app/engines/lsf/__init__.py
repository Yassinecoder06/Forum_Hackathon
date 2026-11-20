"""LSF motion engines (inference only): C0 text-to-pose generator and B2 sign stitching.

Models are trained on private interpreter footage and are NOT in this repository. They are
loaded lazily from local files (see config.py) the first time an engine is used and kept in
memory. Output is an automatic proxy, never a validated LSF translation.
"""
from __future__ import annotations

import threading
import time

from . import config
from .config import EngineUnavailable

ENGINES = ("c0", "b2")
_cache: dict[str, object] = {}
_lock = threading.Lock()
_run = threading.Lock()          # one generation at a time (seeded sampling, shared model state)

# Neutral brow ratios (face_features.measure_face scale) of the B2 lexicon rest pose; used by
# C0, which has no rest frame of its own. Engineering reference, not a linguistic neutral.
DEFAULT_FACE_NEUTRAL = {"brow_left": 0.428, "brow_right": 0.424}


def _engine(name: str):
    with _lock:
        if name not in _cache:
            try:
                import torch  # noqa: F401
            except ImportError as exc:
                raise EngineUnavailable("Moteurs LSF : PyTorch n'est pas installé "
                                        "(pip install -r backend/requirements-lsf-engines.txt).") from exc
            config.require(name)
            if name == "c0":
                from .c0 import C0Engine
                _cache[name] = C0Engine(config.device())
            else:
                from .b2 import B2Engine
                _cache[name] = B2Engine()
        return _cache[name]


def generate(text: str, engine: str, seed: int = 0):
    """-> (poses (T,157,2|3) at 25 fps in normalised engine coordinates, metadata dict)."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}")
    start = time.perf_counter()
    model = _engine(engine)
    with _run:
        return _generate(model, text, engine, seed, start)


def _generate(model, text, engine, seed, start):
    if engine == "c0":
        poses, meta = model.generate(text, seed=seed)
        meta["face_neutral"] = DEFAULT_FACE_NEUTRAL
    else:
        poses, meta = model.generate(text)
        from ...services.lsf_pose_convert import face_ratios
        rest = face_ratios(model.lex.rest[:, :2])
        meta["face_neutral"] = {k: rest[k] for k in ("brow_left", "brow_right")} if rest else DEFAULT_FACE_NEUTRAL
    meta["engine"] = engine
    meta["seconds"] = round(time.perf_counter() - start, 3)
    return poses, meta


def status():
    return config.status()
