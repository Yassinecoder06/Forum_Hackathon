"""Offline accuracy and latency of the webcam sign recognition (backend/app/services/sign_recognition.py).

    python scripts/sign_recognition_eval.py [--lexicon L.npz] [--poses DIR] [--out DIR]

Private data (never in Git):
  --lexicon  the B2 lexicon as served (lexicon_union_corpus_fast.npz: 222 words from the interpreter corpus + 720 words
             from isolated-sign dictionaries, ONE clip per word)
  --poses    MediaPipe Holistic poses of the isolated-sign videos (LSF_data <source>/<word>.npz: pts, t_ms; sources
             education-nationale, elix, jauvert, 4:3 videos). Some words are filmed by 2-3 sources = different signers.

Every query takes the SAME path as a webcam take: raw image coordinates (true aspect ratio) -> prepare -> segment ->
core -> DTW against the lexicon. Leave-one-out: a query video is never its own template.
  cross-signer   query = a dictionary video of a word whose lexicon clip comes from ANOTHER source
                 (iso: another dictionary; corpus: the interpreter corpus). Full lexicon (942), the ~100 evaluated
                 words, and the everyday conversation words.
  re-performance the lexicon clip's own video, re-performed synthetically (tempo 0.75-1.35 with a non-linear warp,
                 position +-0.1 and size +-8 % of the shoulder width, landmark noise, hands lost in 10 % of the frames):
                 a stand-in for a person matched against their OWN recorded sign (personal templates).
  phrase         2-3 re-performed signs in a row with the hands lowered in between: segmentation + per-sign accuracy.
Top-1 / top-5 / median rank, and the time per take (prepare + segment + DTW) for a 4 s take.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from backend.app.services import sign_recognition as R  # noqa: E402

LEX = r"D:\matignon\pathB\b2\lexicon_union_corpus_fast.npz"
POSES = r"D:\matignon\external\hf_LSF_data\poses"
OUT = r"D:\matignon\signrec"
ASPECT = {"education-nationale": 288 / 384, "elix": 480 / 640, "jauvert": 480 / 640}


def load_iso(poses_dir):
    """Every isolated video -> source, word key, raw pts in true aspect ratio (y in units of the width), t_ms."""
    out = []
    for f in sorted(glob.glob(os.path.join(poses_dir, "*", "*.npz"))):
        src = os.path.basename(os.path.dirname(f))
        key = R.key_of(os.path.splitext(os.path.basename(f))[0])
        if not key or " " in key:
            continue
        z = np.load(f)
        pts = z["pts"].astype("float32").copy()
        if len(pts) < 5 or np.isfinite(pts[:, :23, 0]).all(1).mean() < 0.8:
            continue
        pts[..., 1] *= ASPECT.get(src, 0.75)
        out.append(dict(source=src, key=key, pts=pts, t_ms=z["t_ms"].astype("float64")))
    return out


def reperform(raw, t_ms, rng):
    """The same signer signing the word again (synthetic): tempo, position, size, noise, lost hands."""
    T = len(raw)
    speed = rng.uniform(0.75, 1.35)
    n = max(6, int(T / speed))
    u = np.linspace(0, 1, n)
    warp = u + rng.uniform(-0.08, 0.08) * np.sin(np.pi * u)               # non-linear time warp, ends fixed
    src = np.clip(warp, 0, 1) * (T - 1)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    w = (src - lo)[:, None, None]
    x = raw[lo] * (1 - w) + raw[hi] * w
    sh = np.nanmedian(np.linalg.norm(raw[:, 11, :2] - raw[:, 12, :2], axis=1))
    centre = np.nanmedian((raw[:, 11, :2] + raw[:, 12, :2]) / 2, axis=0)
    k = rng.uniform(0.92, 1.08)
    x[..., :2] = (x[..., :2] - centre) * k + centre + rng.uniform(-0.1, 0.1, 2) * sh
    x[..., :2] += rng.normal(0, 0.01 * sh, x[..., :2].shape)
    lost = rng.random(n) < 0.1
    x[lost, R.LH:R.RH + 21] = np.nan
    t = (t_ms[lo] * (1 - w[:, 0, 0]) + t_ms[hi] * w[:, 0, 0]) / speed
    return x.astype("float32"), t - t[0]


def ranks_for(rec, queries, vocab=None):
    tp = rec.templates
    allowed = None
    if vocab is not None:
        allowed = np.zeros(len(tp.words), bool)
        allowed[[tp.word_index[w] for w in vocab if w in tp.word_index]] = True
    ranks, ms = [], []
    for clip, word in queries:
        t0 = time.perf_counter()
        d = rec.word_distances(clip)
        ms.append((time.perf_counter() - t0) * 1000)
        if allowed is not None:
            d = np.where(allowed, d, np.inf)
        order = np.argsort(d)
        ranks.append(int(np.where(order == tp.word_index[word])[0][0]))
    return np.array(ranks), ms


def summary(name, ranks, ms, vocab_size, extra=None):
    r = np.asarray(ranks)
    res = {"set": name, "queries": int(len(r)), "vocabulary": int(vocab_size),
           "top1": round(float((r == 0).mean()), 3), "top5": round(float((r < 5).mean()), 3),
           "median_rank": int(np.median(r)) + 1, "chance_top1": round(1 / vocab_size, 4),
           "dtw_ms": round(float(np.mean(ms)), 1) if ms else None, **(extra or {})}
    print(json.dumps(res, ensure_ascii=False), flush=True)
    return res


def main_segment(rec, raw, t_ms):
    tp = rec.templates
    x, _ = R.prepare(raw, tp.rest, t_ms)
    segs = R.segment(x, tp.rest_y)
    if not segs:
        return x, 0
    s = max(segs, key=lambda s: s.end - s.start)
    return x[s.start:s.end], len(segs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lexicon", default=LEX)
    ap.add_argument("--poses", default=POSES)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()
    iso = load_iso(a.poses)
    rec = R.Recognizer(R.Templates.from_npz(a.lexicon))
    tp = rec.templates
    print(f"{len(iso)} isolated videos, lexicon {len(tp)} words ({time.time() - t0:.0f} s)", flush=True)
    src_of = dict(zip(tp.names, tp.sources))
    conv = [w for w in R.CONVERSATION if w in tp.word_index]
    results = []

    # ---- cross-signer
    cross = {"iso": [], "corpus": []}
    nseg = []
    for it in iso:
        w = it["key"]
        if w not in src_of or src_of[w] == it["source"]:
            continue
        clip, k = main_segment(rec, it["pts"], it["t_ms"])
        nseg.append(k)
        cross["corpus" if src_of[w] == "corpus" else "iso"].append((clip, w))
    both = cross["iso"] + cross["corpus"]
    evaluated = sorted({w for _, w in both})
    seg_info = {"one_segment": round(float(np.mean(np.array(nseg) == 1)), 3), "no_segment": round(float(np.mean(np.array(nseg) == 0)), 3)}
    for name, qs in (("cross-signer, lexicon clip from another dictionary", cross["iso"]),
                     ("cross-signer, lexicon clip from the interpreter corpus", cross["corpus"])):
        r, ms = ranks_for(rec, qs)
        results.append(summary(name + " | full lexicon", r, ms, len(tp)))
    r, ms = ranks_for(rec, both, evaluated)
    results.append(summary("cross-signer, all | vocabulary = the evaluated words", r, ms, len(evaluated), seg_info))
    qc = [q for q in both if q[1] in conv]
    if qc:
        r, ms = ranks_for(rec, qc, conv)
        results.append(summary("cross-signer, all | everyday conversation vocabulary", r, ms, len(conv)))

    # ---- re-performance (the lexicon clip's own video, signed again)
    own = [it for it in iso if src_of.get(it["key"]) == it["source"]]
    qs = []
    for it in own:
        raw, t = reperform(it["pts"], it["t_ms"], rng)
        clip, _ = main_segment(rec, raw, t)
        qs.append((clip, it["key"]))
    r, ms = ranks_for(rec, qs)
    results.append(summary("re-performance (own template, synthetic) | full lexicon", r, ms, len(tp)))
    qc = [q for q in qs if q[1] in conv]
    r, ms = ranks_for(rec, qc, conv)
    results.append(summary("re-performance (own template, synthetic) | everyday conversation vocabulary", r, ms, len(conv)))

    # ---- phrases: 2-3 re-performed signs with the hands lowered in between, full webcam path (recognize)
    by_key = {it["key"]: it for it in own}
    pool = [w for w in conv if w in by_key] or list(by_key)
    seg_ok, sign_ok, n_signs, lat = 0, 0, 0, []
    for k in range(60):
        words = list(rng.choice(pool, size=int(rng.integers(2, 4)), replace=False))
        parts, ts = [], []
        for w in words:
            raw, t = reperform(by_key[w]["pts"], by_key[w]["t_ms"], rng)
            raw = R.resample_time(raw, t)                                     # 25 fps, as the browser's take
            parts.append(raw)
            rest = np.repeat(raw[-1:], 12, 0).copy()
            rest[:, 15, 1] = rest[:, 16, 1] = np.nanmax(raw[:, 11, 1]) + 3 * np.nanmedian(np.abs(raw[:, 11, 0] - raw[:, 12, 0]))
            rest[:, R.LH:R.RH + 21] = np.nan                                  # hands down, out of view
            parts.append(rest)
        take = np.concatenate(parts)
        t1 = time.perf_counter()
        out = rec.recognize(take, None, 25.0, 5, conv)
        lat.append((time.perf_counter() - t1) * 1000)
        got = [g["gloss"] for g in out["glosses"]]
        seg_ok += len(got) == len(words)
        n_signs += len(words)
        sign_ok += sum(1 for a_, b_ in zip(got, words) if a_ == b_) if len(got) == len(words) else 0
    results.append({"set": "phrases of 2-3 re-performed everyday signs (webcam path, conversation vocabulary)", "phrases": 60,
                    "segment_count_correct": round(seg_ok / 60, 3), "signs_correct": round(sign_ok / max(1, n_signs), 3),
                    "recognize_ms_mean": round(float(np.mean(lat)), 1)})
    print(json.dumps(results[-1], ensure_ascii=False), flush=True)

    # ---- latency of one 4 s take (prepare + segment + DTW on the full lexicon)
    it = own[0]
    raw = np.concatenate([it["pts"]] * max(1, int(np.ceil(100 / len(it["pts"])))))[:100]
    times = []
    for _ in range(5):
        t1 = time.perf_counter()
        rec.recognize(raw, None, 25.0)
        times.append((time.perf_counter() - t1) * 1000)
    results.append({"set": "latency, 4 s take, full lexicon", "recognize_ms_median": round(float(np.median(times)), 1)})
    print(json.dumps(results[-1]), flush=True)
    json.dump({"results": results, "lexicon": a.lexicon, "iso_videos": len(iso),
               "params": {"N": R.N_RESAMPLE, "band": R.BAND, "w_traj": R.W_TRAJ, "w_shape": R.W_SHAPE, "temperature": R.TEMPERATURE}},
              open(os.path.join(a.out, "eval.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
