"""Sign -> text, vocabulary-limited: what the person signed in front of the webcam, matched against the sign lexicon.

The browser runs MediaPipe (pose + hands + face) on the webcam and sends the landmarks in the corpus layout of
157 points (sign-model/signlib.py PARTS: upper_pose 23, left_hand 21, right_hand 21, lips 40, left/right eyebrow 10,
left/right eye 16), image coordinates (x, y in 0..1, y down), NaN where a part was not seen. Coordinates must have the
true aspect ratio: x / width and y / WIDTH (so one unit is the same length on both axes).

Steps (numpy only):
  1. preparation as for the corpus (signlib.prepare_video_poses): short gaps filled, origin = shoulder midpoint,
     unit = shoulder width, y up, light smoothing, 25 fps; a hand never seen is put at its rest position;
  2. segmentation into signs by wrist motion and stillness: a sign is a stretch where a hand is raised or moving,
     split at pauses (hands still for 0.2 s) and at the hands' lowering;
  3. each segment is reduced to its core (signlib/lex_isolated core: hands above 60 % of their peak height), resampled
     to 20 frames and compared with every lexicon template by DTW (Sakoe-Chiba band) on
       - both wrist trajectories relative to the shoulders (the movement and the place of the sign),
       - both hand shapes (21 points relative to the wrist, scaled by the palm length; weighted down when the hand rests);
     the query is also compared mirrored (left-handed signers);
  4. scores = softmax of the negative distances (temperature from the offline evaluation); confidence = best score.

Templates: the B2 lexicon (SIGNACULTURE_LSF_B2_LEXICON, lexicon_union_corpus_fast.npz, 942 words), or a recognition bank
(SIGNACULTURE_SIGN_TEMPLATES: same npz format, several clips per word, optional `display` names). Both are private
files, never in Git. Templates from the isolated-sign dictionaries (4:3 videos, signer != "corpus") have their y
compressed by 3/4 so all sources share the true aspect ratio.
"""
from __future__ import annotations

import os
import re
import threading
import time
import unicodedata
from dataclasses import dataclass

import numpy as np

TOTAL_POINTS = 157
FPS = 25.0
L_SHOULDER, R_SHOULDER, L_WRIST, R_WRIST = 11, 12, 15, 16
LH, RH = 23, 44                      # first point of left_hand / right_hand
N_RESAMPLE = 20
BAND = 4
ISO_Y = 0.75                         # 4:3 dictionary videos: y in units of height -> units of width
REST_Y = -1.06                       # wrist height at rest in the lexicon (shoulder widths below the shoulders)
RAISED = 0.35                        # a hand is "signing" when its wrist is this much above rest
W_TRAJ, W_SHAPE = 1.0, 0.35
TEMPERATURE = 0.06
MAX_SEGMENTS = 8


class RecognitionUnavailable(RuntimeError):
    """No sign lexicon on this server: the API returns 503."""


# --------------------------------------------------------------------------- preparation (as signlib.prepare_video_poses)
_SG7 = np.array([-2, 3, 6, 7, 6, 3, -2], dtype="float32") / 21.0     # Savitzky-Golay window 7, order 2


def smooth(x: np.ndarray) -> np.ndarray:
    if len(x) < 9:
        return x
    pad = np.concatenate([np.repeat(x[:1], 3, 0), x, np.repeat(x[-1:], 3, 0)])
    out = np.zeros_like(x)
    for k, w in enumerate(_SG7):
        out += w * pad[k:k + len(x)]
    return out


def _neighbours(bad: np.ndarray):
    """Per (frame, column): index of the previous and next valid frame (-1 / T when there is none)."""
    T = bad.shape[0]
    idx = np.arange(T)[:, None]
    prev = np.where(~bad, idx, -1)
    np.maximum.accumulate(prev, axis=0, out=prev)
    nxt = np.where(~bad, idx, T)[::-1]
    nxt = np.minimum.accumulate(nxt, axis=0)[::-1]
    return prev, nxt


def fill_gaps(x: np.ndarray, max_gap: int = 8) -> np.ndarray:
    """Linear interpolation of NaN runs of at most max_gap frames, per coordinate (signlib.fill_gaps, vectorised:
    runs at the start or end of the take are filled with the nearest value, as np.interp does there)."""
    T = x.shape[0]
    flat = x.reshape(T, -1).copy()
    bad = np.isnan(flat)
    if not bad.any():
        return flat.reshape(x.shape)
    prev, nxt = _neighbours(bad)
    has_p, has_n = prev >= 0, nxt < T
    run = np.where(has_p, prev, -1)
    run = np.where(has_n, nxt, T) - run - 1                       # length of the NaN run each frame belongs to
    fill = bad & (has_p | has_n) & (run <= max_gap)
    cols = np.arange(flat.shape[1])[None, :].repeat(T, 0)
    vp = flat[np.clip(prev, 0, T - 1), cols]
    vn = flat[np.clip(nxt, 0, T - 1), cols]
    t = np.arange(T)[:, None]
    w = np.where(has_p & has_n, (t - prev) / np.maximum(nxt - prev, 1), np.where(has_p, 0.0, 1.0))
    val = vp * (1 - w) + vn * w
    val = np.where(has_p, val, vn)
    val = np.where(has_n, val, vp)
    flat[fill] = val[fill]
    return flat.reshape(x.shape)


def normalize(x: np.ndarray) -> np.ndarray:
    """Origin at the median shoulder midpoint, unit = median shoulder width, y up (signlib.normalize)."""
    sh = 0.5 * (x[:, L_SHOULDER] + x[:, R_SHOULDER])
    ok = ~np.isnan(sh).any(1)
    if not ok.any():
        raise ValueError("Épaules non détectées : placez-vous face à la caméra, buste visible.")
    c = np.nanmedian(sh[ok], axis=0)
    s = float(np.nanmedian(np.linalg.norm(x[ok, L_SHOULDER, :2] - x[ok, R_SHOULDER, :2], axis=1)))
    x = (x - c) / max(s, 1e-3)
    x[..., 1] *= -1.0
    return x


def resample_time(x: np.ndarray, t_ms: np.ndarray, fps: float = FPS) -> np.ndarray:
    """Irregular webcam frames -> a regular grid at fps (linear, NaN kept where a neighbour is NaN)."""
    t = (t_ms - t_ms[0]) / 1000.0
    n = max(1, int(np.floor(t[-1] * fps)) + 1)
    grid = np.arange(n) / fps
    hi = np.clip(np.searchsorted(t, grid, side="left"), 1, len(t) - 1) if len(t) > 1 else np.zeros(n, int)
    if len(t) == 1:
        return x[:1].copy()
    lo = hi - 1
    w = ((grid - t[lo]) / np.maximum(t[hi] - t[lo], 1e-6)).clip(0, 1)[:, None, None]
    return (x[lo] * (1 - w) + x[hi] * w).astype("float32")


def prepare(raw: np.ndarray, rest: np.ndarray, t_ms: np.ndarray | None = None, fps: float = FPS) -> tuple[np.ndarray, np.ndarray]:
    """(T,157,3) raw image coordinates (true aspect ratio, NaN = missing) -> normalized 25 fps clip + hand presence (T,2)."""
    x = np.asarray(raw, dtype="float32")
    if x.ndim != 3 or x.shape[1:] != (TOTAL_POINTS, 3):
        raise ValueError(f"Points attendus : (images, {TOTAL_POINTS}, 3).")
    if t_ms is not None and len(t_ms) == len(x) and len(x) > 1:
        x = resample_time(x, np.asarray(t_ms, dtype="float64"))
    elif abs(fps - FPS) > 0.5 and len(x) > 1:
        x = resample_time(x, np.arange(len(x)) * 1000.0 / fps)
    x = fill_gaps(x)
    pres = np.stack([~np.isnan(x[:, LH:LH + 21, 0]).any(1), ~np.isnan(x[:, RH:RH + 21, 0]).any(1)], 1)
    x = normalize(x)
    T = len(x)
    flat = x.reshape(T, -1)
    bad = np.isnan(flat)
    if bad.any():                                                        # previous valid value, else next, else rest
        prev, nxt = _neighbours(bad)
        cols = np.arange(flat.shape[1])[None, :].repeat(T, 0)
        src = np.where(prev >= 0, prev, np.where(nxt < T, nxt, 0))
        val = flat[src, cols]
        never = bad.all(0)
        val[:, never] = rest.reshape(-1)[never]
        flat = np.where(bad, val, flat)
    x = flat.reshape(x.shape)
    for k, o in enumerate((LH, RH)):                                     # a hand almost never seen rests
        if pres[:, k].mean() < 0.15:
            x[:, o:o + 21] = rest[o:o + 21]
    return smooth(x), pres


# --------------------------------------------------------------------------- segmentation
@dataclass
class Segment:
    start: int
    end: int          # exclusive, frames at 25 fps


def activity(x: np.ndarray, rest_y: float = REST_Y) -> tuple[np.ndarray, np.ndarray]:
    wy = x[:, [L_WRIST, R_WRIST], 1]
    raised = (wy > rest_y + RAISED).any(1)
    sp = np.r_[0.0, np.linalg.norm(np.diff(x[:, [L_WRIST, R_WRIST], :2], axis=0), axis=2).max(1)]
    sp = np.convolve(sp, np.ones(3) / 3, mode="same")
    return raised, sp


def segment(x: np.ndarray, rest_y: float = REST_Y, min_len: int = 6, pause: int = 8, still: float = 0.012,
            max_sign: int = 150, bridge: int = 8) -> list[Segment]:
    """Signs = runs where a hand is raised (drops of up to `bridge` frames bridged). Only a very long run (more than
    max_sign / 2 frames, i.e. signing without lowering the hands) is split at pauses where both wrists stay still for
    `pause` frames: holds inside a sign must not cut it. Segments shorter than min_len frames are dropped."""
    raised, sp = activity(x, rest_y)
    T = len(x)
    act = raised.copy()
    # bridge short drops
    edges = np.diff(np.r_[0, act.astype(int), 0])
    starts, ends = np.where(edges == 1)[0], np.where(edges == -1)[0]
    for a, b in zip(ends[:-1], starts[1:]):
        if b - a <= bridge:
            act[a:b] = True
    edges = np.diff(np.r_[0, act.astype(int), 0])
    runs = list(zip(np.where(edges == 1)[0], np.where(edges == -1)[0]))
    out: list[Segment] = []
    for a, b in runs:
        if b - a < min_len:
            continue
        cuts = [a]
        if b - a > max_sign // 2:
            calm = sp[a:b] < still
            k = a + 3
            while k < b - 3:
                if calm[k - a] :
                    j = k
                    while j < b and calm[j - a]:
                        j += 1
                    if j - k >= pause and k - cuts[-1] >= min_len and b - j >= min_len:
                        cuts.append((k + j) // 2)
                    k = j
                else:
                    k += 1
        cuts.append(b)
        for s, e in zip(cuts[:-1], cuts[1:]):
            if e - s >= min_len:
                out.append(Segment(int(s), int(e)))
    if not out and T >= min_len and (sp > still * 2).sum() >= min_len:     # signing low (no raised hand): whole take
        out = [Segment(0, T)]
    return out[:MAX_SEGMENTS]


def core(x: np.ndarray, rest_y: float = REST_Y, frac: float = 0.6, pad: int = 1) -> np.ndarray:
    """Sign core (lex_isolated.core): frames where the higher wrist is above frac x its peak height over rest."""
    h = (x[:, [L_WRIST, R_WRIST], 1] - rest_y).max(1)
    if h.max() <= 0:
        return x
    on = h >= frac * h.max()
    i0, i1 = int(np.argmax(on)), len(on) - int(np.argmax(on[::-1]))
    c = x[max(0, i0 - pad):min(len(x), i1 + pad)]
    return c if len(c) >= 3 else x


# --------------------------------------------------------------------------- features + DTW
def resample(x: np.ndarray, n: int = N_RESAMPLE) -> np.ndarray:
    L = len(x)
    idx = np.linspace(0, L - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, L - 1)
    w = (idx - lo)[:, None, None]
    return x[lo] * (1 - w) + x[hi] * w


def mirror(x: np.ndarray) -> np.ndarray:
    """Left/right swap (a left-handed signer): x -> -x and the two arms, wrists and hands exchanged."""
    m = x.copy()
    m[..., 0] *= -1
    for a, b in ((11, 12), (13, 14), (15, 16), (17, 18), (19, 20), (21, 22)):
        m[:, [a, b]] = m[:, [b, a]]
    m[:, LH:LH + 21], m[:, RH:RH + 21] = m[:, RH:RH + 21].copy(), m[:, LH:LH + 21].copy()
    return m


def features(x: np.ndarray, rest_y: float = REST_Y) -> np.ndarray:
    """(L,157,3) normalized clip -> (N_RESAMPLE, D) weighted features: wrists (4) + 2 hand shapes (2 x 40)."""
    c = resample(core(x, rest_y))
    wr = c[:, [L_WRIST, R_WRIST], :2].reshape(len(c), -1) * W_TRAJ
    parts = [wr]
    for o, wi in ((LH, L_WRIST), (RH, R_WRIST)):
        h = c[:, o:o + 21, :2]
        rel = h[:, 1:] - h[:, :1]
        scale = np.linalg.norm(h[:, 9] - h[:, 0], axis=1).clip(0.05)[:, None, None]
        shape = (rel / scale).reshape(len(c), -1)
        up = np.clip((c[:, wi, 1] - rest_y) / RAISED, 0.25, 1.0)[:, None]      # a resting hand counts less
        parts.append(shape * up * (W_SHAPE / np.sqrt(20)))
    return np.concatenate(parts, 1).astype("float32")


def dtw_batch(q: np.ndarray, T: np.ndarray, band: int = BAND) -> np.ndarray:
    """DTW distance between one query (N,D) and M templates (M,N,D), Sakoe-Chiba band, mean cost along the path."""
    M, N, Dm = T.shape
    # squared distances by one matrix product: |q|^2 + |t|^2 - 2 q.t  -> (M, N, N)
    qq = (q * q).sum(1)
    tt = (T * T).sum(2)
    cross = (T.reshape(M * N, Dm) @ q.T).reshape(M, N, N).transpose(0, 2, 1)
    C = np.sqrt(np.maximum(qq[None, :, None] + tt[:, None, :] - 2 * cross, 0)).astype("float32")
    inf = np.float32(np.inf)
    D = np.full((M, N + 1, N + 1), inf, dtype="float32")
    D[:, 0, 0] = 0
    for i in range(1, N + 1):
        for j in range(max(1, i - band), min(N, i + band) + 1):
            D[:, i, j] = C[:, i - 1, j - 1] + np.minimum(np.minimum(D[:, i - 1, j], D[:, i, j - 1]), D[:, i - 1, j - 1])
    return D[:, N, N] / (2 * N)


# --------------------------------------------------------------------------- templates
def key_of(s: str) -> str:
    s = "".join(c for c in unicodedata.normalize("NFD", s.lower()) if unicodedata.category(c) != "Mn")
    return " ".join(re.sub(r"[^a-z0-9]+", " ", s).split())


class Templates:
    """A sign bank: names (one per clip, several clips may share a word), display labels, features (M,N,D)."""

    def __init__(self, names, clips, sources, rest, display=None):
        self.rest = rest.astype("float32")
        self.rest_y = float(np.mean(self.rest[[L_WRIST, R_WRIST], 1]))
        self.names = list(names)
        self.sources = list(sources)
        self.display = dict(display or {})
        feats = []
        for c, s in zip(clips, sources):
            c = c.astype("float32").copy()
            if s != "corpus":
                c[..., 1] *= ISO_Y
            feats.append(features(c, self.rest_y))
        self.F = np.stack(feats) if feats else np.zeros((0, N_RESAMPLE, 84), "float32")
        self.words = sorted(set(self.names))
        self.word_index = {w: i for i, w in enumerate(self.words)}
        self.clip_word = np.array([self.word_index[n] for n in self.names])

    @classmethod
    def from_npz(cls, path):
        z = np.load(path, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        off = z["offsets"]
        frames = z["frames"]                                  # read once (each z[...] decompresses the array)
        clips = [frames[off[i]:off[i + 1]] for i in range(len(names))]
        sources = [str(s) for s in z["signers"]] if "signers" in z.files else ["corpus"] * len(names)
        display = {}
        if "display" in z.files:
            display = {n: str(d) for n, d in zip(names, z["display"])}
        return cls(names, clips, sources, z["rest"], display)

    def __len__(self):
        return len(self.words)

    def label(self, word: str) -> str:
        return self.display.get(word, word)

    def add(self, names, clips, sources, display=None):
        """More clips (the person's own signs): same features, words added to the vocabulary if new."""
        if not names:
            return self
        F = [features(c.astype("float32"), self.rest_y) for c in clips]
        self.F = np.concatenate([self.F, np.stack(F)])
        self.names += list(names)
        self.sources += list(sources)
        self.display.update(display or {})
        self.words = sorted(set(self.names))
        self.word_index = {w: i for i, w in enumerate(self.words)}
        self.clip_word = np.array([self.word_index[n] for n in self.names])
        return self


# --------------------------------------------------------------------------- recognizer
def _softmax_scores(d: np.ndarray, temperature: float) -> np.ndarray:
    z = -(d - d.min()) / temperature
    e = np.exp(z)
    return e / e.sum()


class Recognizer:
    def __init__(self, templates: Templates | None = None, temperature: float = TEMPERATURE):
        self._templates = templates
        self.temperature = temperature
        self.lock = threading.Lock()

    @property
    def templates(self) -> Templates:
        with self.lock:
            if self._templates is None:
                self._templates = load_default_templates()
            return self._templates

    def word_distances(self, clip: np.ndarray, exclude: np.ndarray | None = None, mirrored: bool = True) -> np.ndarray:
        """Normalized clip -> best DTW distance per WORD (min over its templates, and over the mirrored query)."""
        tp = self.templates
        F = tp.F if exclude is None else tp.F[~exclude]
        cw = tp.clip_word if exclude is None else tp.clip_word[~exclude]
        d = dtw_batch(features(clip, tp.rest_y), F)
        if mirrored:
            d = np.minimum(d, dtw_batch(features(mirror(clip), tp.rest_y), F))
        best = np.full(len(tp.words), np.inf, dtype="float32")
        np.minimum.at(best, cw, d)
        return best

    def rank(self, clip: np.ndarray, k: int = 5, exclude: np.ndarray | None = None, vocabulary: list[str] | None = None) -> list[dict]:
        tp = self.templates
        best = self.word_distances(clip, exclude)
        if vocabulary:
            allowed = np.zeros(len(best), bool)
            allowed[[tp.word_index[w] for w in vocabulary if w in tp.word_index]] = True
            best = np.where(allowed, best, np.inf)
        ok = np.isfinite(best)
        order = np.argsort(best)[: max(k, 1)]
        scores = np.zeros_like(best)
        scores[ok] = _softmax_scores(best[ok], self.temperature)
        return [{"gloss": tp.words[i], "label": tp.label(tp.words[i]), "score": round(float(scores[i]), 4),
                 "distance": round(float(best[i]), 4)} for i in order if ok[i]]

    def recognize(self, raw: np.ndarray, t_ms: np.ndarray | None = None, fps: float = FPS, k: int = 5,
                  vocabulary: list[str] | None = None) -> dict:
        t0 = time.perf_counter()
        tp = self.templates
        x, pres = prepare(raw, tp.rest, t_ms, fps)
        segs = segment(x, tp.rest_y)
        if vocabulary:                      # the person's own signs always count
            vocabulary = sorted({w for w in vocabulary if w in tp.word_index} | {n for n, src in zip(tp.names, tp.sources) if src == "user"})
        out = []
        for s in segs:
            cands = self.rank(x[s.start:s.end], k, vocabulary=vocabulary)
            out.append({"start_ms": round(s.start * 1000 / FPS), "end_ms": round(s.end * 1000 / FPS), "candidates": cands})
        glosses = [{"gloss": o["candidates"][0]["gloss"], "label": o["candidates"][0]["label"], "score": o["candidates"][0]["score"]}
                   for o in out if o["candidates"]]
        confidence = float(np.mean([g["score"] for g in glosses])) if glosses else 0.0
        text = " ".join(g["label"] for g in glosses)
        return {
            "glosses": glosses,
            "segments": out,
            "text_guess": (text[:1].upper() + text[1:]) if text else "",
            "confidence": round(confidence, 4),
            "frames": int(len(x)),
            "hands_seen": round(float(pres.any(1).mean()), 3) if len(pres) else 0.0,
            "vocabulary": len(vocabulary) if vocabulary else len(tp),
            "ms": round((time.perf_counter() - t0) * 1000),
        }

    def main_clip(self, raw: np.ndarray, t_ms: np.ndarray | None = None, fps: float = FPS) -> np.ndarray:
        """A take of ONE sign -> its normalized clip (the longest segment), to store as the person's own template."""
        tp = self.templates
        x, _ = prepare(raw, tp.rest, t_ms, fps)
        segs = segment(x, tp.rest_y)
        if not segs:
            raise ValueError("Aucun signe dans la prise : levez les mains devant la poitrine.")
        s = max(segs, key=lambda s: s.end - s.start)
        return x[s.start:s.end]

    def learn(self, gloss: str, raw: np.ndarray, t_ms: np.ndarray | None = None, fps: float = FPS) -> dict:
        """Store the person's own version of a sign (private, on this server) and use it right away."""
        word = key_of(gloss)[:40]
        if not word:
            raise ValueError("Indiquez le mot signé.")
        clip = self.main_clip(raw, t_ms, fps)
        tp = self.templates
        label = tp.label(word) if word in tp.word_index else gloss.strip()[:40]
        user_store().append(word, clip, label)
        with self.lock:
            tp.add([word], [clip], ["user"], {word: label})
        count = sum(1 for n, src in zip(tp.names, tp.sources) if src == "user" and n == word)
        return {"gloss": word, "label": label, "count": count}

    def forget(self, gloss: str) -> int:
        removed = user_store().remove(key_of(gloss))
        with self.lock:
            self._templates = None                        # reloaded with the remaining personal signs
        return removed


class UserStore:
    """The person's own sign takes (normalized clips), a private npz under the server storage (never in Git)."""

    def __init__(self, path):
        self.path = path

    def load(self):
        if not os.path.isfile(self.path):
            return [], [], {}
        z = np.load(self.path, allow_pickle=False)
        names = [str(n) for n in z["names"]]
        off = z["offsets"]
        frames = z["frames"]
        clips = [frames[off[i]:off[i + 1]].astype("float32") for i in range(len(names))]
        return names, clips, {n: str(d) for n, d in zip(names, z["display"])}

    def _save(self, names, clips, display):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        if not names:
            if os.path.isfile(self.path):
                os.remove(self.path)
            return
        offs = np.cumsum([0] + [len(c) for c in clips])
        tmp = self.path + ".tmp.npz"
        np.savez_compressed(tmp, names=np.array(names), frames=np.concatenate(clips).astype("float16"), offsets=offs,
                            display=np.array([display.get(n, n) for n in names]))
        os.replace(tmp, self.path)

    def append(self, gloss, clip, label):
        names, clips, display = self.load()
        self._save(names + [gloss], clips + [clip], {**display, gloss: label})

    def remove(self, gloss):
        names, clips, display = self.load()
        keep = [i for i, n in enumerate(names) if n != gloss]
        self._save([names[i] for i in keep], [clips[i] for i in keep], display)
        return len(names) - len(keep)

    def summary(self):
        names, _, display = self.load()
        out = {}
        for n in names:
            out.setdefault(n, {"gloss": n, "label": display.get(n, n), "count": 0})["count"] += 1
        return sorted(out.values(), key=lambda r: r["gloss"])


def user_store() -> UserStore:
    path = os.getenv("SIGNACULTURE_SIGN_USER_TEMPLATES")
    if not path:
        from .capture_library import storage
        path = str(storage() / "sign_user_templates.npz")
    return UserStore(path)


# Everyday words of the lexicon for a conversation (the UI's "conversation" vocabulary): a small vocabulary makes the
# recognition far more reliable than the full lexicon (scripts/sign_recognition_eval.py measures both).
CONVERSATION = ("bonjour merci oui comment pourquoi quand quoi toi aller manger boire travail travailler "
                "ecole famille pouvoir savoir comprendre sourd entendant langue signe demain hier matin soir semaine annee "
                "beau soleil froid chaud content triste fatigue malade attendre encore finir commencer apprendre regarder voir "
                "ecrire telephone voiture train bus ville paris france film cinema theatre culture livre sport jouer enfant "
                "frere soeur payer acheter cafe eau pain grand nouveau bon mauvais super pardon").split()


def vocabulary_info() -> dict:
    tp = recognizer.templates
    conv = [w for w in CONVERSATION if w in tp.word_index]
    return {"size": len(tp), "conversation": [{"gloss": w, "label": tp.label(w)} for w in conv],
            "user": user_store().summary()}


def load_default_templates() -> Templates:
    path = os.getenv("SIGNACULTURE_SIGN_TEMPLATES")
    if not path:
        from ..engines.lsf import config
        path = str(config.path("b2_lexicon"))
    if not os.path.isfile(path):
        raise RecognitionUnavailable("Lexique de signes absent sur ce serveur : la reconnaissance des signes est indisponible.")
    tp = Templates.from_npz(path)
    names, clips, display = user_store().load()
    return tp.add(names, clips, ["user"] * len(names), display)


def available() -> bool:
    path = os.getenv("SIGNACULTURE_SIGN_TEMPLATES")
    if not path:
        from ..engines.lsf import config
        path = str(config.path("b2_lexicon"))
    return os.path.isfile(path)


recognizer = Recognizer()
