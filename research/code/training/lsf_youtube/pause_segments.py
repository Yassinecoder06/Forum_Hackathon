"""Snap lag-shifted subtitle windows to the interpreter's own pauses.

1. Hand speed (wrists + fingertips) is smoothed; pauses are local minima that
   fall below the 25th percentile of activity in their 10-minute block.
2. The recording is cut at those pauses into signing chunks of 0.8-12 s
   (over-long chunks are split at their deepest internal minimum).
3. Each chunk is assigned to the lag-shifted sentence window it overlaps most;
   the sentence then spans exactly its chunks. A chunk belongs to one sentence
   only, so neighbouring pairs no longer share signing. Sentences that receive
   no chunk keep their lag-shifted window.

Input is ``<id>.pairs.lagshifted.jsonl`` from ``shift_pairs.py``; output is
``<id>.pairs.paused.jsonl`` plus ``<id>.pauses.json``. Pairs remain weak,
unreviewed alignments.

    PYTHONPATH=research/code .venv/bin/python -m training.lsf_youtube.pause_segments \
        POSES.npz SHIFTED_PAIRS.jsonl [--spotting candidates.jsonl]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from training.lsf_youtube.shift_pairs import BLOCK_MS, hand_speed, smooth

MIN_CHUNK = 10       # frames (0.8 s at 12.5 Hz)
MAX_CHUNK = 150      # frames (12 s)
PAUSE_PCT = 25


def pauses(speed, t_ms):
    """Indices of pause frames: local minima below the block's activity percentile."""
    minima = np.where((speed[1:-1] <= speed[:-2]) & (speed[1:-1] <= speed[2:]))[0] + 1
    keep = []
    for start in range(int(t_ms[0]), int(t_ms[-1]) + 1, BLOCK_MS):
        lo, hi = np.searchsorted(t_ms, [start, start + BLOCK_MS])
        if hi - lo < 50:
            continue
        cut = np.percentile(speed[lo:hi], PAUSE_PCT)
        keep += [i for i in minima[(minima >= lo) & (minima < hi)] if speed[i] <= cut]
    gaps = np.where(np.diff(t_ms) > 240)[0]          # sampling gaps are always boundaries
    return np.unique(np.concatenate([keep, gaps, gaps + 1, [0, len(t_ms) - 1]]).astype(int))


def chunks(boundaries, speed):
    """Turn pause frames into [start, stop) chunks within length limits."""
    out = []
    for a, b in zip(boundaries[:-1], boundaries[1:]):
        stack = [(int(a), int(b))]
        while stack:
            s, e = stack.pop()
            if e - s > MAX_CHUNK:                    # split at the deepest internal minimum
                m = s + MIN_CHUNK + int(np.argmin(speed[s + MIN_CHUNK:e - MIN_CHUNK]))
                stack += [(m, e), (s, m)]
            elif e - s >= MIN_CHUNK:
                out.append((s, e))
            elif out and s == out[-1][1]:            # too short: merge into the previous chunk
                out[-1] = (out[-1][0], e)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("poses", type=Path)
    p.add_argument("pairs", type=Path, help="<id>.pairs.lagshifted.jsonl")
    p.add_argument("--spotting", type=Path)
    a = p.parse_args()

    z = np.load(a.poses)
    t_ms, has = z["t_ms"].astype(np.int64), z["has"].astype(bool)
    speed = smooth(hand_speed(z["pts"], t_ms), 6)
    cuts = chunks(pauses(speed, t_ms), speed)
    rows = [json.loads(line) for line in a.pairs.open()]

    # assign each chunk to the sentence window it overlaps most
    starts = np.array([r["frame_start"] for r in rows]); stops = np.array([r["frame_stop"] for r in rows])
    owned = {i: [] for i in range(len(rows))}
    last = 0
    for s, e in cuts:                                # chunks in time order
        overlap = np.minimum(stops, e) - np.maximum(starts, s)
        i = int(np.argmax(overlap))
        if overlap[i] <= 0:
            continue
        i = max(i, last)                             # monotonic: never give a later chunk to an earlier sentence
        owned[i].append((s, e)); last = i

    spans = [(min(c[0] for c in owned[i]), max(c[1] for c in owned[i])) if owned[i] else None for i in range(len(rows))]
    out, snapped = [], 0
    for i, r in enumerate(rows):
        if spans[i]:
            s, e = spans[i]
            snapped += 1
            how = "lag-shifted, snapped to interpreter pauses (weak)"
        else:
            # no chunk of its own: keep the shifted window, clipped to the gap between snapped neighbours
            prev_end = max([sp[1] for sp in spans[:i] if sp] or [0])
            next_start = min([sp[0] for sp in spans[i + 1:] if sp] or [len(t_ms)])
            s, e = max(r["frame_start"], prev_end), min(r["frame_stop"], next_start)
            how = r["alignment"] + "; no pause-delimited chunk"
        mask = has[s:e]
        if len(mask) < 5:
            continue
        pose, any_hand = float(mask[:, 0].mean()), float((mask[:, 1] | mask[:, 2]).mean())
        out.append({**r, "start_ms": int(t_ms[s]), "end_ms": int(t_ms[e - 1]) + 80, "frame_start": int(s), "frame_stop": int(e),
                    "chunks": len(owned[i]), "pose_coverage": pose, "any_hand_coverage": any_hand,
                    "both_hand_coverage": float((mask[:, 1] & mask[:, 2]).mean()), "alignment": how,
                    "candidate": bool(pose >= .85 and any_hand >= .5 and len(r["french_text"].split()) >= 3)})
    target = a.pairs.with_name(a.pairs.name.replace(".pairs.lagshifted.jsonl", ".pairs.paused.jsonl"))
    target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out))

    lengths = lambda rs: float(np.median([(r["frame_stop"] - r["frame_start"]) * .08 for r in rs]))
    shared = lambda rs: float(np.mean([max(0, x["frame_stop"] - y["frame_start"]) > 0 for x, y in zip(rs, rs[1:])]))
    report = {"video_id": rows[0]["video_id"] if rows else None, "chunks": len(cuts),
              "median_chunk_s": round(float(np.median([(e - s) * .08 for s, e in cuts])), 2),
              "pairs_in": len(rows), "pairs_out": len(out), "snapped_to_pauses": snapped,
              "median_window_s": {"lagshifted": round(lengths(rows), 2), "paused": round(lengths(out), 2)},
              "neighbours_sharing_frames": {"lagshifted": round(shared(rows), 3), "paused": round(shared(out), 3)},
              "candidates": {"lagshifted": sum(r["candidate"] for r in rows), "paused": sum(r["candidate"] for r in out)}}

    if a.spotting and a.spotting.is_file():
        strong = [json.loads(line) for line in a.spotting.open()]
        strong = [s for s in strong if s["null_percentile"] <= .01]

        def rate(rs):
            hits = []
            for s in strong:
                w = next((r for r in rs if r["speech_start_ms"] <= s["spoken_ms"] < r["speech_end_ms"]), None)
                if w:
                    mid = (s["match_ms"][0] + s["match_ms"][1]) / 2
                    hits.append(w["start_ms"] <= mid < w["end_ms"])
            return round(float(np.mean(hits)), 3) if hits else None
        report["validation"] = {"strong_word_matches": len(strong),
                                "sign_inside_own_sentence": {"lagshifted": rate(rows), "paused": rate(out)},
                                "note": "unreviewed spotting matches (about half may be wrong); indicative only"}
    a.pairs.with_name(a.pairs.name.replace(".pairs.lagshifted.jsonl", ".pauses.json")).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
