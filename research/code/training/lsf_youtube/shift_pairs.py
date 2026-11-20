"""Shift weak subtitle/pose pairs by the interpreter's measured lag.

Subtitle windows follow the *speaker's* clock; the interpreter signs later.
The lag is estimated per block of the recording (interpreters take turns) by
cross-correlating the caption word rate with the interpreter's hand speed:
when more words are spoken, the hands get busier a moment later. Each window
is then moved by its block's lag and its pose frames, coverage and candidate
flag are recomputed exactly as in ``align.py``. The original pairs file is
not modified.

Optional validation: given dictionary-spotting candidates, report how often a
strongly matched sign falls inside the window of the sentence containing its
word, before and after the shift.

    PYTHONPATH=research/code .venv/bin/python -m training.lsf_youtube.shift_pairs \
        POSES.npz CAPTIONS.json3 PAIRS.jsonl [--spotting candidates.jsonl]
"""
import argparse
import json
from pathlib import Path

import numpy as np

STEP_MS = 80                 # 12.5 Hz
MAX_LAG_MS = 6000
BLOCK_MS = 10 * 60 * 1000
MIN_PEAK = 0.05              # correlation below this falls back to the video-wide lag


def word_times(captions):
    out = []
    for event in json.loads(captions.read_text())["events"]:
        for seg in event.get("segs") or []:
            if seg.get("utf8", "").strip():
                out.append(event["tStartMs"] + seg.get("tOffsetMs", 0))
    return np.array(sorted(out), dtype=np.int64)


def hand_speed(pts, t_ms):
    """Mean wrist + fingertip speed per frame (ROI-normalized units per frame), NaN-safe."""
    idx = [15, 16, 23, 31, 35, 44, 52, 56]           # pose wrists, hand wrists, index/middle tips
    xy = pts[:, idx, :2].astype(np.float32)
    v = np.linalg.norm(np.diff(xy, axis=0), axis=2)
    finite = np.isfinite(v)
    v = np.where(finite.any(axis=1), np.where(finite, v, 0).sum(axis=1) / np.maximum(finite.sum(axis=1), 1), 0.0)
    v = np.concatenate([[0], np.nan_to_num(v, nan=0.0)])
    gaps = np.diff(t_ms, prepend=t_ms[0]) > 3 * STEP_MS   # do not count jumps across sampling gaps
    v[gaps] = 0
    return np.clip(v, 0, np.percentile(v, 99))


def smooth(x, frames=12):
    kernel = np.ones(frames) / frames
    return np.convolve(x, kernel, mode="same")


def best_lag(words_rate, speed):
    """Lag (ms, signing after speech) maximizing the normalized cross-correlation."""
    a = (words_rate - words_rate.mean()) / (words_rate.std() + 1e-9)
    b = (speed - speed.mean()) / (speed.std() + 1e-9)
    lags = np.arange(-1000 // STEP_MS, MAX_LAG_MS // STEP_MS + 1)
    corr = [float(np.mean(a[: len(a) - k] * b[k:])) if k >= 0 else float(np.mean(a[-k:] * b[: len(b) + k])) for k in lags]
    k = int(np.argmax(corr))
    return int(lags[k] * STEP_MS), corr[k]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("poses", type=Path)
    p.add_argument("captions", type=Path)
    p.add_argument("pairs", type=Path)
    p.add_argument("--spotting", type=Path)
    a = p.parse_args()

    z = np.load(a.poses)
    t_ms, has = z["t_ms"].astype(np.int64), z["has"].astype(bool)
    speed = smooth(hand_speed(z["pts"], t_ms))
    words = word_times(a.captions)
    rate = smooth(np.bincount(np.clip(np.searchsorted(t_ms, words), 0, len(t_ms) - 1), minlength=len(t_ms)).astype(float), 25)

    overall, overall_peak = best_lag(rate, speed)
    blocks = []
    for start in range(int(t_ms[0]), int(t_ms[-1]), BLOCK_MS):
        lo, hi = np.searchsorted(t_ms, [start, start + BLOCK_MS])
        if hi - lo < 2000 or rate[lo:hi].sum() < 50:
            blocks.append({"start_ms": start, "lag_ms": overall, "peak": None, "source": "video-wide (too little speech)"})
            continue
        lag, peak = best_lag(rate[lo:hi], speed[lo:hi])
        ok = peak >= MIN_PEAK and 0 <= lag <= MAX_LAG_MS
        blocks.append({"start_ms": start, "lag_ms": lag if ok else overall, "peak": round(peak, 3),
                       "source": "block" if ok else "video-wide (weak block correlation)"})

    def lag_at(ms):
        return blocks[min(len(blocks) - 1, max(0, (ms - int(t_ms[0])) // BLOCK_MS))]["lag_ms"]

    rows = [json.loads(line) for line in a.pairs.open()]
    shifted = []
    for r in rows:
        lag = lag_at(r["start_ms"])
        s, e = r["start_ms"] + lag, r["end_ms"] + lag
        start, end = (int(i) for i in np.searchsorted(t_ms, [s, e]))
        mask = has[start:end]
        if len(mask) < 5:
            continue
        complete = t_ms[start] - s <= 160 and e - t_ms[end - 1] <= 160
        pose, any_hand = float(mask[:, 0].mean()), float((mask[:, 1] | mask[:, 2]).mean())
        shifted.append({**r, "start_ms": int(s), "end_ms": int(e), "speech_start_ms": r["start_ms"], "speech_end_ms": r["end_ms"],
                        "lag_ms": lag, "frame_start": start, "frame_stop": end, "pose_coverage": pose,
                        "any_hand_coverage": any_hand, "both_hand_coverage": float((mask[:, 1] & mask[:, 2]).mean()),
                        "alignment": "subtitle-time shifted by estimated interpreter lag (weak)",
                        "candidate": bool(complete and pose >= .85 and any_hand >= .5 and len(r["french_text"].split()) >= 3)})
    out = a.pairs.with_name(a.pairs.name.replace(".pairs.jsonl", ".pairs.lagshifted.jsonl"))
    out.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in shifted))

    report = {"video_id": rows[0]["video_id"] if rows else None, "video_lag_ms": overall, "video_peak": round(overall_peak, 3),
              "block_minutes": BLOCK_MS // 60000, "blocks": blocks, "pairs_in": len(rows), "pairs_out": len(shifted),
              "candidates_before": sum(r["candidate"] for r in rows), "candidates_after": sum(r["candidate"] for r in shifted)}

    if a.spotting and a.spotting.is_file():
        strong = [json.loads(line) for line in a.spotting.open()]
        strong = [s for s in strong if s["null_percentile"] <= .01]

        def inside(windows, key_s, key_e, s):
            for w in windows:
                if w[key_s] <= s["spoken_ms"] < w[key_e]:   # the sentence that contains the spoken word
                    mid = (s["match_ms"][0] + s["match_ms"][1]) / 2
                    return w["start_ms"] <= mid < w["end_ms"]
            return None
        before = [inside(rows, "start_ms", "end_ms", s) for s in strong]
        after = [inside(shifted, "speech_start_ms", "speech_end_ms", s) for s in strong]
        before = [x for x in before if x is not None]
        after = [x for x in after if x is not None]
        report["validation"] = {
            "strong_word_matches": len(strong),
            "sign_inside_own_sentence_before": round(float(np.mean(before)), 3) if before else None,
            "sign_inside_own_sentence_after": round(float(np.mean(after)), 3) if after else None,
            "note": "uses unreviewed spotting matches (about half may be wrong); indicative only",
        }
    a.pairs.with_name(a.pairs.name.replace(".pairs.jsonl", ".lag.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "blocks"}, indent=2, ensure_ascii=False))
    print("block lags (s):", [round(b["lag_ms"] / 1000, 1) for b in blocks])


if __name__ == "__main__":
    main()
