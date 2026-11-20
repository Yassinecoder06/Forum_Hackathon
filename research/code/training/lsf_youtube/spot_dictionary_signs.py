"""Caption-guided dictionary sign spotting in interpreted LSF broadcasts.

For every caption word that names a concept in the local France-LSF dictionary,
search the interpreter's MediaPipe motion shortly after the word is spoken for
the stretch that best matches the dictionary recording (subsequence DTW over
wrist-to-shoulder positions and wrist-relative 2D handshapes).

Two controls keep the output honest:
* null scores: each dictionary sign is also matched at random times of the same
  video, giving a per-sign percentile for every observed score;
* decoy signs: for every word occurrence, a random *different* dictionary sign is
  searched in the same window. If true words do not clear the threshold clearly
  more often than decoys, there is no usable signal.

Outputs are unreviewed candidates. A low score is evidence of similar motion,
not proof that the interpreter produced that sign.

    PYTHONPATH=research/code .venv/bin/python -m training.lsf_youtube.spot_dictionary_signs \
        POSES.npz CAPTIONS.json3 .runtime/library.json OUT_DIR
"""
import argparse
import gzip
import json
from pathlib import Path
import random
import re

import numpy as np
import simplemma

FPS = 12.5
LAG = (-1000, 7000)      # search window around the spoken word, ms
NULL_SAMPLES = 40
REST_MARGIN = .35        # shoulder widths a wrist must rise above rest to count as signing
STOP = set("le la les un une des de du d l à au aux et ou en y ce ça se sa son ses je tu il elle on nous vous ils elles "
           "me te ne pas que qui quoi dont où est être avoir a ont sont été fait faire mais donc car par pour sur dans "
           "avec sans plus moins très tout tous toute toutes".split())


def normalize_concept(text):
    return " ".join(re.sub(r"·\w+", "", text.casefold()).split())


# ---------- features: identical definition for broadcast and dictionary ----------
def features(shoulder_l, shoulder_r, wrists, hands):
    """Per-frame vector: 2 sides x (wrist xy rel. shoulders, 20 handshape xy). NaN when missing.

    shoulder_*: (T,2) pixels; wrists: dict side->(T,2); hands: dict side->(T,21,2) or NaN."""
    centre = (shoulder_l + shoulder_r) / 2
    width = np.linalg.norm(shoulder_l - shoulder_r, axis=1, keepdims=True)
    width = np.where(width > 1e-6, width, np.nan)
    parts = []
    for side in ("left", "right"):
        hand = hands[side]
        wrist = np.where(np.isnan(hand[:, 0]), wrists[side], hand[:, 0])
        parts.append((wrist - centre) / width)
        rel = hand[:, 1:] - hand[:, :1]
        scale = np.linalg.norm(hand[:, 9] - hand[:, 0], axis=1)[:, None, None]
        scale = np.where(scale > 1e-6, scale, np.nan)
        parts.append((rel / scale).reshape(len(hand), -1))
    return np.concatenate(parts, axis=1).astype(np.float32)   # (T, 2+40+2+40)


def broadcast_features(npz, roi_px):
    pts = npz["pts"].astype(np.float32)
    w, h = roi_px[2] - roi_px[0], roi_px[3] - roi_px[1]
    xy = pts[..., :2] * np.array([w, h], np.float32)
    return features(xy[:, 11], xy[:, 12], {"left": xy[:, 15], "right": xy[:, 16]},
                    {"left": xy[:, 23:44], "right": xy[:, 44:65]}), npz["t_ms"].astype(np.int64)


def dictionary_features(path):
    frames = json.load(gzip.open(path, "rt"))
    if not frames:
        return None
    w, h = frames[0]["image_width"], frames[0]["image_height"]
    t, sl, sr, wl, wr, hl, hr = [], [], [], [], [], [], []
    nan2, nan21 = np.full(2, np.nan), np.full((21, 2), np.nan)
    for f in frames:
        pose = f.get("PLM")
        if not pose:
            continue
        p = np.array([[q["x"] * w, q["y"] * h] for q in pose], np.float32)
        t.append(f["time_ms"]); sl.append(p[11]); sr.append(p[12]); wl.append(p[15]); wr.append(p[16])
        for out, key in ((hl, "LLM"), (hr, "RLM")):
            out.append(np.array([[q["x"] * w, q["y"] * h] for q in f[key]], np.float32) if f.get(key) else nan21)
    if len(t) < 4:
        return None
    feat = features(np.array(sl), np.array(sr), {"left": np.array(wl), "right": np.array(wr)},
                    {"left": np.array(hl), "right": np.array(hr)})
    # Resample to the broadcast rate and keep the active part (a hand above the waist).
    t = np.array(t, float)
    grid = np.arange(t[0], t[-1] + 1, 1000 / FPS)
    idx = np.clip(np.searchsorted(t, grid), 0, len(t) - 1)
    feat = feat[idx]
    # Keep only the signing core: frames where a wrist is clearly raised above
    # its own rest height (median of the first and last frames, hands down).
    raised = np.zeros(len(feat), bool)
    for col in (1, 43):                                  # wrist y relative to shoulders (smaller = higher)
        y = feat[:, col]
        rest = np.nanmedian(np.concatenate([y[:3], y[-3:]]))
        raised |= np.nan_to_num(rest - y, nan=0) > REST_MARGIN
    active = np.where(raised)[0]
    if len(active) < 4:
        return None
    return feat[active[0]:active[-1] + 1]


# ---------- matching ----------
WEIGHTS = np.concatenate([np.full(2, 3.0), np.full(40, .15), np.full(2, 3.0), np.full(40, .15)]).astype(np.float32)


def frame_cost(q, s):
    """Pairwise cost (len q, len s); NaN components are ignored, fully-missing pairs get a penalty."""
    diff = np.abs(q[:, None, :] - s[None, :, :]) * WEIGHTS
    valid = ~np.isnan(diff)
    total = np.where(valid, diff, 0).sum(-1)
    weight = (valid * WEIGHTS).sum(-1)
    return np.where(weight > 0, total / np.maximum(weight, 1e-6), 2.0)


def subsequence_dtw(q, s):
    """Best mean per-frame cost of all of q against any contiguous part of s; returns (score, start, end).

    Slope-limited steps (1,1), (1,2), (2,1): the matched stretch is between half
    and twice the dictionary duration, and every query frame is counted once, so
    the score is a mean over the query and cannot collapse onto one frame."""
    c = frame_cost(q, s)
    n, m = c.shape
    inf = np.inf
    acc = np.full((n, m), inf); start = np.zeros((n, m), int)
    acc[0] = c[0]; start[0] = np.arange(m)
    for i in range(1, n):
        cands = np.full((3, m), inf); origin = np.zeros((3, m), int)
        cands[0, 1:] = acc[i - 1, :-1]; origin[0, 1:] = start[i - 1, :-1]            # (1,1)
        cands[1, 2:] = acc[i - 1, :-2]; origin[1, 2:] = start[i - 1, :-2]            # (1,2): stream faster
        if i >= 2:
            cands[2, 1:] = acc[i - 2, :-1] + c[i - 1, 1:]; origin[2, 1:] = start[i - 2, :-1]  # (2,1): stream slower
        k = np.argmin(cands, axis=0)
        acc[i] = cands[k, np.arange(m)] + c[i]
        start[i] = origin[k, np.arange(m)]
    j = int(np.argmin(acc[-1]))
    if not np.isfinite(acc[-1, j]):
        return inf, 0, 0
    return float(acc[-1, j] / n), int(start[-1, j]), j


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("poses", type=Path)
    p.add_argument("captions", type=Path)
    p.add_argument("library", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--max-occurrences", type=int, default=6, help="per concept, to bound runtime")
    a = p.parse_args()
    rng = random.Random(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)

    summary = json.loads(a.poses.with_name(a.poses.name.replace(".npz", ".summary.json")).read_text())
    stream, t_ms = broadcast_features(np.load(a.poses), summary["crop_xyxy_pixels"])

    # dictionary: single-word concepts -> list of (gloss_id, feature sequence)
    lexicon = {}
    for r in json.loads(a.library.read_text()):
        if r.get("sign_language") not in (None, "fsl"):
            continue
        for name in [r["gloss"]["concept_fr"], *r["gloss"].get("aliases", [])]:
            key = normalize_concept(name)
            if " " not in key and len(key) > 2 and key not in STOP:
                lexicon.setdefault(key, []).append(r)
    cache = {}

    def variants(key):
        if key not in cache:
            seqs = []
            for r in lexicon[key]:
                f = dictionary_features(r["raw_animics_path"]) if r.get("raw_animics_path") else None
                if f is not None and 4 <= len(f) <= 60:
                    seqs.append((r["gloss"]["gloss_id"], r["gloss"]["concept_fr"], r.get("capture_id"), f))
            cache[key] = seqs
        return cache[key]

    def window(center):
        lo, hi = np.searchsorted(t_ms, center + LAG[0]), np.searchsorted(t_ms, center + LAG[1])
        return lo, hi

    # Interpreter rest level: the lowest typical wrist height over the whole video.
    rest = {col: np.nanpercentile(stream[:, col], 90) for col in (1, 43)}

    def at_rest(lo, hi):
        part = stream[lo:hi + 1]
        raised = np.zeros(len(part), bool)
        for col in (1, 43):
            raised |= np.nan_to_num(rest[col] - part[:, col], nan=0) > REST_MARGIN
        return raised.mean() < .5

    def best(key, lo, hi):
        result = None
        for gid, concept, capture, q in variants(key):
            if hi - lo < len(q):
                continue
            score, s, e = subsequence_dtw(q, stream[lo:hi])
            if at_rest(lo + s, lo + e):
                continue
            if result is None or score < result[0]:
                result = (score, gid, concept, capture, lo + s, lo + e)
        return result

    # caption words with their own timestamps
    events = json.loads(a.captions.read_text())["events"]
    words = []
    for e in events:
        for s in e.get("segs") or []:
            text = s.get("utf8", "").strip().casefold()
            for token in re.findall(r"[^\W\d_]+", text):
                lemma = simplemma.lemmatize(token, lang="fr")
                key = token if token in lexicon else lemma if lemma in lexicon else None
                if key:
                    words.append((e["tStartMs"] + s.get("tOffsetMs", 0), token, key))
    span = (t_ms[0] + 10_000, t_ms[-1] - 10_000)
    words = [w for w in words if span[0] < w[0] < span[1]]
    per_key = {}
    for w in words:
        per_key.setdefault(w[2], []).append(w)
    selected = [w for ws in per_key.values() for w in (rng.sample(ws, a.max_occurrences) if len(ws) > a.max_occurrences else ws)]
    keys = sorted(per_key)
    print(f"{len(words)} caption words match {len(keys)} dictionary concepts; testing {len(selected)} occurrences", flush=True)

    null = {}

    def null_scores(key):
        if key not in null:
            mentions = [w[0] for w in per_key.get(key, [])]
            scores = []
            while len(scores) < NULL_SAMPLES:
                c = rng.uniform(*span)
                if any(abs(c - m) < 20_000 for m in mentions):
                    continue
                r = best(key, *window(c))
                if r:
                    scores.append(r[0])
            null[key] = np.array(scores)
        return null[key]

    rows = []
    for k, (time, token, key) in enumerate(sorted(selected)):
        lo, hi = window(time)
        hit = best(key, lo, hi)
        decoy_key = rng.choice([x for x in keys if x != key])
        decoy = best(decoy_key, lo, hi)
        if not hit or not decoy:
            continue
        pct = float((null_scores(key) <= hit[0]).mean())
        dpct = float((null_scores(decoy_key) <= decoy[0]).mean())
        rows.append({
            "spoken_ms": int(time), "word": token, "concept": hit[2], "gloss_id": hit[1],
            "dictionary_capture": hit[3], "match_ms": [int(t_ms[hit[4]]), int(t_ms[hit[5]])],
            "lag_ms": int(t_ms[hit[4]] - time), "score": round(hit[0], 4), "null_percentile": round(pct, 3),
            "decoy_concept": decoy[2], "decoy_null_percentile": round(dpct, 3),
            "review": "pending",
        })
        if k % 25 == 0:
            print(f"{k}/{len(selected)} {token}: score {hit[0]:.3f}, null pct {pct:.2f} (decoy {dpct:.2f})", flush=True)

    with open(a.out / "candidates.jsonl", "w") as f:
        for r in sorted(rows, key=lambda r: (r["null_percentile"], r["score"])):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    report = {"video_id": summary["video_id"], "caption_words_matching_dictionary": len(words),
              "concepts": len(keys), "tested_occurrences": len(rows), "lag_window_ms": LAG, "null_samples": NULL_SAMPLES}
    for cut in (.01, .05, .10):
        real = np.mean([r["null_percentile"] <= cut for r in rows]) if rows else 0
        decoy = np.mean([r["decoy_null_percentile"] <= cut for r in rows]) if rows else 0
        report[f"pass_rate_at_{int(cut*100)}pct"] = {"true_word": round(float(real), 3), "decoy_sign": round(float(decoy), 3)}
    if rows:
        lags = [r["lag_ms"] for r in rows if r["null_percentile"] <= .05]
        report["median_lag_ms_of_passing"] = int(np.median(lags)) if lags else None
    (a.out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
