"""Index weak French subtitle/pose pairs from local SignVerse samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import numpy as np

FPS = 24
STAMP = r"(?:\d{2,}:)?[0-5]\d:[0-5]\d\.\d{3}"
CUE = re.compile(rf"^({STAMP})\s+-->\s+({STAMP})(?:\s+.*)?$")


def milliseconds(value):
    seconds, minutes, *hours = reversed(value.split(":"))
    whole, fraction = seconds.split(".")
    return (int(hours[0]) * 3600000 if hours else 0) + int(minutes) * 60000 + int(whole) * 1000 + int(fraction)


def cues(path):
    content = path.read_text(encoding="utf-8-sig").strip()
    if not content.startswith("WEBVTT"):
        raise ValueError(f"Invalid WebVTT header: {path.name}")
    for block in re.split(r"\n\s*\n", content)[1:]:
        lines = block.splitlines()
        if not lines or lines[0].startswith(("NOTE", "STYLE", "REGION")):
            continue
        matches = [(i, CUE.fullmatch(line)) for i, line in enumerate(lines[:2])]
        match = next(((i, found) for i, found in matches if found), None)
        if match is None:
            raise ValueError(f"Invalid cue in {path.name}")
        index, match = match
        yield milliseconds(match[1]), milliseconds(match[2]), " ".join(lines[index+1:])


def prepare(raw, poses):
    counts = Counter()
    splits = Counter()
    manifest = json.loads((poses / "manifest.json").read_text())
    if manifest["sign_language"] != "fsl" or manifest["fps"] != FPS:
        raise ValueError("Expected local France-LSF pose manifest at 24 fps")
    rows = []
    for split, recordings in sorted(manifest["groups"].items()):
        for video in recordings:
            folder = raw / video
            source = json.loads((folder / "manifest.json").read_text())
            if (source["sign_language"], source["split"], source["video_id"]) != ("fsl", split, video):
                raise ValueError("Pose and source provenance differ")
            subtitle = folder / f"{video}.fr.vtt"
            path = poses / f"{video}.npz"
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            with np.load(path, allow_pickle=False) as data:
                ids, mask = data["frame_ids"], data["mask"]
            if (np.diff(ids) <= 0).any() or len(mask) != len(ids):
                raise ValueError("Invalid frame index")
            for cue_number, (start, end, text) in enumerate(cues(subtitle), 1):
                counts["cues"] += 1
                first = (start * FPS + 999) // 1000
                stop = (end * FPS + 999) // 1000
                low, high = np.searchsorted(ids, [first, stop])
                if not text.strip() or not 500 <= end - start <= 15000:
                    counts["invalid_text_or_duration"] += 1
                    continue
                if high - low != stop - first:
                    counts["incomplete_frame_range"] += 1
                    continue
                clip = mask[low:high]
                body = float(clip[:, :18].mean())
                hands = float(clip[:, 18:60].mean())
                if body < .5 or hands < .5:
                    counts["insufficient_body_or_hand_coverage"] += 1
                    continue
                rows.append({
                    **{key: source[key] for key in ("sign_language", "source", "source_video_id",
                        "revision", "split", "signer_id", "license", "permission_status")},
                    "id": f"{video}:{cue_number}", "french_caption": text,
                    "subtitle_language": "fr", "transcription_source": "released VTT; provenance unknown",
                    "start_ms": start, "end_ms": end, "pose_path": str(path), "pose_sha256": digest,
                    "pose_row_slice": [int(low), int(high)], "fps": FPS,
                    "observed_body_fraction": body, "observed_hand_fraction": hands,
                    "alignment_status": "timestamp overlap only; signing alignment unreviewed",
                })
                splits[split] += 1
    return rows, {"candidate_counts": dict(counts), "retained_by_original_split": dict(splits),
                  "recordings": len({r["source_video_id"] for r in rows}), "gold_glosses": 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Path("data/poses/private/raw"))
    parser.add_argument("--poses", type=Path, default=Path("data/poses/private/windows-v1"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows, report = prepare(args.raw, args.poses)
    with args.output.open("x") as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
