"""Re-slice reviewed signing intervals against the original normalized pose cache."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def key(row, split):
    return (split, row["source_video_id"], int(row["start_ms"]), int(row["end_ms"]))


def prepare(candidates, cohorts, root):
    index = {}
    for split, rows in cohorts.items():
        for row in rows:
            identity = key(row, split)
            if identity in index:
                raise ValueError("Ambiguous original interval")
            index[identity] = row
    cache = {}
    output = []
    counts = Counter()
    seen = set()
    for review in candidates:
        identity = key(review, review["split"])
        if identity in seen or identity not in index:
            raise ValueError("Duplicate or unknown reviewed cue")
        seen.add(identity)
        row = index[identity]
        caption = row["french_caption"]
        if caption.lstrip().startswith(("=", "+", "-", "@")):
            caption = "'" + caption
        if (review["caption"] != caption or review.get("review_evidence") != "vidéo source"
                or review.get("meaning_alignment") != "complète"):
            raise ValueError("Expected matching caption with full, source-backed review")
        start, end = review["reviewed_start_ms"], review["reviewed_end_ms"]
        if type(start) is not int or type(end) is not int or not 0 <= start < end:
            raise ValueError("Invalid corrected timestamps")
        if row["source"] != "SignerX/SignVerse-2M" or row["sign_language"] != "fsl" or row["subtitle_language"] != "fr":
            raise ValueError("Unexpected recording provenance")
        path = root / row["pose_path"]
        digest = row["pose_sha256"]
        if (path, digest) not in cache:
            if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError("Pose checksum mismatch")
            with np.load(path, allow_pickle=False) as data:
                ids, masks, xy = data["frame_ids"], data["mask"], data["normalized_xy"]
                if (ids.ndim != 1 or np.any(np.diff(ids) <= 0)
                        or masks.shape != (len(ids), 128) or xy.shape != (len(ids), 128, 2)):
                    raise ValueError("Unexpected normalized pose layout")
                cache[path, digest] = ids, masks.astype(bool) & np.isfinite(xy).all(-1)
        ids, masks = cache[path, digest]
        fps = row["fps"]
        if fps != 24:
            raise ValueError("Expected the audited 24 FPS source")
        first, stop = ((ms * fps + 999) // 1000 for ms in (start, end))
        low, high = np.searchsorted(ids, (first, stop))
        if not 500 <= end-start <= 15000 or high-low != stop-first:
            counts["outside_duration_or_frame_coverage"] += 1
            continue
        body = float(masks[low:high, :18].mean())
        hands = float(masks[low:high, 18:60].mean())
        if min(body, hands) < .5:
            counts["insufficient_visibility"] += 1
            continue
        output.append({**row, "start_ms": start, "end_ms": end,
                       "original_start_ms": row["start_ms"], "original_end_ms": row["end_ms"],
                       "pose_row_slice": [int(low), int(high)],
                       "observed_body_fraction": body, "observed_hand_fraction": hands,
                       "evaluation_cohort": review["split"], "review_item": review["item"],
                       "alignment_status": "source-video reviewed meaning and timing"})
        counts["accepted"] += 1
        counts[f"accepted_{review['split']}"] += 1
    return output, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("intervals", type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--additional", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cohorts = {split: [] for split in ("train", "val", "test", "new_test")}
    for row in records(args.original):
        cohorts[row["split"]].append(row)
    for row in records(args.additional):
        if row["split"] != "test":
            raise ValueError("Additional cohort must contain test records only")
        cohorts["new_test"].append(row)
    output, counts = prepare(records(args.intervals), cohorts, args.root)
    with args.output.open("x", encoding="utf-8") as file:
        for row in output:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"accepted": len(output), "counts": counts}, indent=2))


if __name__ == "__main__":
    main()
