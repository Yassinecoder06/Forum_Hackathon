"""LSF motion engines (inference only): C0 text-to-pose generator and B2 sign stitching.

Models are trained on private interpreter footage and are NOT in this repository. They are
loaded lazily from local files (see config.py) the first time an engine is used and kept in
memory. Output is an automatic proxy, never a validated LSF translation.
"""
from __future__ import annotations

import json
import re
import threading
import time
import unicodedata

from . import config
from .config import EngineUnavailable

ENGINES = ("c0", "b2")
_cache: dict[str, object] = {}
_lock = threading.Lock()
_run = threading.Lock()          # one generation at a time (seeded sampling, shared model state)

# Neutral face channels of the B2 lexicon rest pose; used by C0, which has no rest frame of its own.
# Engineering reference, not a linguistic neutral.
from ...services.lsf_pose_convert import FACE_NEUTRAL_DEFAULT as DEFAULT_FACE_NEUTRAL  # noqa: E402


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


def text_key(text: str) -> str:
    """Sentence identity for the curated C0 takes: case, spacing, apostrophes and final punctuation ignored."""
    t = unicodedata.normalize("NFC", text).replace("’", "'").lower()
    return re.sub(r"\s+", " ", t).strip(" .!?")


_curated: dict[str, object] | None = None


def curated_c0() -> dict:
    """Curated C0 takes (<model dir>/c0/curated.npz): for the demo sentences, the seed-0 sample that was checked
    against the real interpreter. Sampling depends on the CPU and PyTorch build, so another machine would otherwise
    draw a different take of the same sentence."""
    global _curated
    if _curated is None:
        found: dict[str, object] = {}
        p = config.model_dir() / "c0" / "curated.npz"
        if p.is_file():
            import numpy as np
            z = np.load(p, allow_pickle=False)
            for i, text in enumerate(json.loads(str(z["texts"]))):
                found[text_key(text)] = z[f"poses_{i}"]
        _curated = found
    return _curated


def generate(text: str, engine: str, seed: int = 0, groups=None):
    """-> (poses (T,157,2|3) at 25 fps in normalised engine coordinates, metadata dict).
    groups: B2 only, gloss lists per clause in LSF order (lsf_order); ignored by C0, which reads French text."""
    if engine not in ENGINES:
        raise ValueError(f"unknown engine {engine!r}")
    start = time.perf_counter()
    model = _engine(engine)
    with _run:
        return _generate(model, text, engine, seed, start, groups)


def _generate(model, text, engine, seed, start, groups=None):
    take = curated_c0().get(text_key(text)) if engine == "c0" and seed == 0 else None
    if take is not None:
        poses, meta = take, {"curated": True, "sampling": {"seed": 0}}
        meta["face_neutral"] = DEFAULT_FACE_NEUTRAL
    elif engine == "c0":
        poses, meta = model.generate(text, seed=seed)
        meta["face_neutral"] = DEFAULT_FACE_NEUTRAL
    else:
        poses, meta = model.generate(text, groups)
        from ...services.lsf_pose_convert import face_ratios
        rest = face_ratios(model.lex.rest[:, :2])
        meta["face_neutral"] = {k: rest.get(k, v) for k, v in DEFAULT_FACE_NEUTRAL.items()} if rest else DEFAULT_FACE_NEUTRAL
    meta["engine"] = engine
    meta["seconds"] = round(time.perf_counter() - start, 3)
    return poses, meta


def status():
    return config.status()
