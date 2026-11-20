"""Model file locations for the LSF engines. Nothing is ever downloaded at request time.

All files live outside Git (they are trained on private interpreter footage). Point the
API at them either with one folder:

    SIGNACULTURE_LSF_MODEL_DIR=/path/to/lsf-models     (default: <storage>/lsf-models, i.e. .runtime/lsf-models)
        c0/tok_d2_l2.pt  c0/gen_d2_l2.pt
        b2/transition_lsf.pt  b2/prosody.json  b2/lexicon_union_corpus_fast.npz

or file by file (each overrides the folder layout):

    SIGNACULTURE_LSF_C0_TOKENIZER, SIGNACULTURE_LSF_C0_GENERATOR,
    SIGNACULTURE_LSF_B2_TRANSITIONS, SIGNACULTURE_LSF_B2_PROSODY, SIGNACULTURE_LSF_B2_LEXICON

The C0 text encoder is the public `intfloat/multilingual-e5-base`; it is loaded from the local
Hugging Face cache only (`local_files_only`), or from SIGNACULTURE_LSF_TEXT_ENCODER (a local
folder). SIGNACULTURE_LSF_DEVICE = auto | cpu | cuda (default auto).
"""
from __future__ import annotations

import os
from pathlib import Path

E5_ID = "intfloat/multilingual-e5-base"

FILES = {
    "c0_tokenizer": ("SIGNACULTURE_LSF_C0_TOKENIZER", "c0/tok_d2_l2.pt"),
    "c0_generator": ("SIGNACULTURE_LSF_C0_GENERATOR", "c0/gen_d2_l2.pt"),
    "b2_transitions": ("SIGNACULTURE_LSF_B2_TRANSITIONS", "b2/transition_lsf.pt"),
    "b2_prosody": ("SIGNACULTURE_LSF_B2_PROSODY", "b2/prosody.json"),
    "b2_lexicon": ("SIGNACULTURE_LSF_B2_LEXICON", "b2/lexicon_union_corpus_fast.npz"),
}
ENGINE_FILES = {"c0": ("c0_tokenizer", "c0_generator"), "b2": ("b2_transitions", "b2_prosody", "b2_lexicon")}


class EngineUnavailable(RuntimeError):
    """A model file or dependency is missing: the API answers 503 with this message."""


def model_dir() -> Path:
    if os.getenv("SIGNACULTURE_LSF_MODEL_DIR"):
        return Path(os.environ["SIGNACULTURE_LSF_MODEL_DIR"])
    root = Path(__file__).resolve().parents[4]
    return Path(os.getenv("SIGNACULTURE_STORAGE", str(root / ".runtime"))) / "lsf-models"


def path(key: str) -> Path:
    env, relative = FILES[key]
    return Path(os.environ[env]) if os.getenv(env) else model_dir() / relative


def missing(engine: str) -> list[str]:
    return [f"{FILES[k][0]} ({path(k)})" for k in ENGINE_FILES[engine] if not path(k).is_file()]


def require(engine: str) -> None:
    gone = missing(engine)
    if gone:
        raise EngineUnavailable(
            f"Moteur LSF {engine.upper()} indisponible : fichier(s) de modèle absent(s) : {', '.join(gone)}. "
            "Les poids sont privés et ne sont pas dans le dépôt ; voir docs/pipeline/LSF_ENGINES.md.")


def device() -> str:
    choice = os.getenv("SIGNACULTURE_LSF_DEVICE", "auto").lower()
    if choice in ("cpu", "cuda"):
        return choice
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def status() -> dict:
    out = {}
    for engine in ENGINE_FILES:
        gone = missing(engine)
        out[engine] = {"available": not gone, "missing": gone}
    return out
