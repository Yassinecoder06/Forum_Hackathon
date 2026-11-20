"""B2 sign stitching, inference only.

Copied from sign-model/signlib.py (Lexicon, text_to_lemmas, TransitionModel, blend, retime)
and sign-model/pathB2.py (compose_b2, lowpass). Per-word clips from a lexicon are chained
with learned LSF transitions (an MLP correction on top of a smoothstep), real prosody
(clip speed kept, visible gap = 75th percentile of real gaps) and a 6 Hz zero-phase
low-pass. Words outside the lexicon are skipped and reported as unknown.
"""
from __future__ import annotations

import json
import re
import unicodedata

import numpy as np

from . import config

TOTAL_POINTS = 157
FPS = 25.0
CTX, K = 6, 12
# Function words that slipped into the lexicon are not signed (sign-model render_b1_demo STOPN).
_STOP = set("le la les un une des de du et en au aux que qui ce cette ces son sa ses leur leurs pour par sur dans avec plus pas ne se "
            "sont est être a ont avoir fait faire il elle ils elles nous vous je tu on y ou mais donc car comme aussi très tout tous "
            "toute toutes cela ça ceci celui celle ceux elles même autre autres sans sous entre vers chez depuis alors ainsi encore "
            "déjà bien".split())


def _norm(w):
    w = unicodedata.normalize("NFKD", w.lower())
    return "".join(c for c in w if not unicodedata.combining(c))


STOPN = {_norm(w) for w in _STOP} | {"ce", "des", "cet"}


def key_of(s):
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


class Lexicon:
    def __init__(self, path):
        z = np.load(path, allow_pickle=False)
        self.names = [str(n) for n in z["names"]]
        self.frames = z["frames"]
        self.offsets = z["offsets"]
        self.rest = z["rest"].astype("float32")
        self.index = {n: i for i, n in enumerate(self.names)}
        self.key_index = {}
        for n in self.names:
            self.key_index.setdefault(key_of(n), n)

    def __len__(self):
        return len(self.names)

    def clip(self, lemma):
        i = self.index[lemma]
        return self.frames[self.offsets[i]:self.offsets[i + 1]].astype("float32")


def _fr_lemmatize(tok):
    try:
        import simplemma
        return simplemma.lemmatize(tok, lang="fr")
    except Exception:
        for suf in ("aux", "es", "s", "e"):
            if tok.endswith(suf) and len(tok) > len(suf) + 2:
                return tok[: -len(suf)] + ("al" if suf == "aux" else "")
        return tok


def text_to_lemmas(text, lex):
    """Words found in the lexicon (exact or lemmatised), in text order; the rest is unknown."""
    lemmas, unknown = [], []
    for tok in re.findall(r"[\w'-]+", text.lower()):
        tok = tok.split("'")[-1] if "'" in tok else tok
        cands = [key_of(tok), key_of(_fr_lemmatize(tok))]
        hit = next((lex.key_index[c] for c in cands if c in lex.key_index), None)
        if hit and hit not in STOPN:
            lemmas.append(hit)
        elif not hit and _norm(tok) not in STOPN:
            unknown.append(tok)
    return lemmas, unknown


def _smoothstep(n):
    t = np.linspace(0, 1, n + 2)[1:-1]
    return (t * t * (3 - 2 * t))[:, None, None]


def blend(a_last, b_first, n):
    w = _smoothstep(n)
    return a_last[None] * (1 - w) + b_first[None] * w


def _resample_fixed(clip, n=16):
    L = len(clip)
    idx = np.linspace(0, L - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, L - 1)
    w = (idx - lo)[:, None, None]
    return clip[lo] * (1 - w) + clip[hi] * w


def retime(x, speed):
    if abs(speed - 1.0) < 1e-3:
        return x
    T = len(x)
    n = max(2, int(round(T / speed)))
    idx = np.linspace(0, T - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (idx - lo)[:, None, None]
    return x[lo] * (1 - w) + x[hi] * w


class TransitionModel:
    def __init__(self, path, ctx=CTX, K=K):
        import torch
        import torch.nn as nn
        ck = torch.load(path, map_location="cpu", weights_only=True)
        cfg = ck["cfg"]
        self.torch, self.ctx, self.K = torch, ctx, K
        d, mods = 2 * ctx * TOTAL_POINTS * 3 + 1, []
        for _ in range(cfg["layers"]):
            mods += [nn.Linear(d, cfg["hidden"]), nn.GELU(), nn.Dropout(cfg.get("dropout", 0.0))]
            d = cfg["hidden"]
        mods += [nn.Linear(d, K * TOTAL_POINTS * 3)]
        self.net = nn.Sequential(*mods)
        self.net.load_state_dict(ck["state"])
        self.net.eval()

    def transition(self, a_clip, b_clip, n):
        ctx, K = self.ctx, self.K
        a, b = a_clip[-ctx:], b_clip[:ctx]
        if len(a) < ctx:
            a = np.concatenate([np.repeat(a[:1], ctx - len(a), axis=0), a])
        if len(b) < ctx:
            b = np.concatenate([b, np.repeat(b[-1:], ctx - len(b), axis=0)])
        X = np.concatenate([a, b]).astype("float32")[None]
        w = _smoothstep(K)[None]
        base = X[:, ctx - 1][:, None] * (1 - w) + X[:, ctx][:, None] * w
        feats = np.concatenate([X.reshape(1, -1), [[float(np.clip(n, 2, 20)) / 20.0]]], axis=1).astype("float32")
        with self.torch.no_grad():
            res = self.net(self.torch.from_numpy(feats)).numpy().reshape(1, K, TOTAL_POINTS, 3)
        pred = (base + res)[0]
        return pred if n == K else _resample_fixed(pred, n)


def lowpass(x, cutoff=6.0, fps=FPS):
    from scipy.signal import butter, filtfilt
    if len(x) < 16:
        return x
    b, a = butter(3, cutoff / (fps / 2))
    return filtfilt(b, a, x, axis=0).astype("float32")


def compose_b2(lex, lemmas, tm, prosody):
    rest = lex.rest
    gap = int(np.clip(round(prosody["gap_s_p75"] * FPS), 3, 15))
    min_len = int(max(4, round(prosody["sign_s_p10"] * FPS)))
    hold = 6
    parts = [np.repeat(rest[None], hold, 0)]
    prev, prev_clip = rest, None
    for lem in lemmas:
        c = lex.clip(lem)
        if len(c) < min_len:
            c = retime(c, len(c) / min_len)
        parts.append(tm.transition(prev_clip, c, gap) if prev_clip is not None else blend(prev, c[0], 8))
        parts.append(c)
        prev, prev_clip = c[-1], c
    parts.append(blend(prev, rest, 8))
    parts.append(np.repeat(rest[None], hold, 0))
    return lowpass(np.concatenate(parts).astype("float32"))


class B2Engine:
    def __init__(self):
        config.require("b2")
        try:
            import scipy  # noqa: F401
        except ImportError as exc:
            raise config.EngineUnavailable("Moteur B2 : le paquet 'scipy' n'est pas installé "
                                           "(pip install -r backend/requirements-lsf-engines.txt).") from exc
        self.lex = Lexicon(config.path("b2_lexicon"))
        self.tm = TransitionModel(config.path("b2_transitions"))
        self.prosody = json.loads(config.path("b2_prosody").read_text(encoding="utf-8"))

    def generate(self, text):
        lemmas, unknown = text_to_lemmas(text, self.lex)
        if lemmas:
            x = compose_b2(self.lex, lemmas, self.tm, self.prosody)
        else:
            x = np.repeat(self.lex.rest[None], int(FPS), 0)
        return x, {"lemmas": lemmas, "unknown": unknown, "lexicon_size": len(self.lex)}
