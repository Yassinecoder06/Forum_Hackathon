"""Audit returned reviewer worksheets and export reviewed intervals for reprocessing."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


IDENTITY = ("item", "split", "caption", "source_video_id", "start_ms", "end_ms")
VISUAL = {"bonne", "partielle", "illisible"}
MEANING = {"complète", "partielle", "absente", "indéterminable"}
EVIDENCE = {"squelette", "vidéo source"}
RATINGS = ("visual_quality", "meaning_alignment", "start_offset_ms", "end_offset_ms", "evidence", "notes")


def worksheet(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        if not set(IDENTITY + RATINGS).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path.name}: expected worksheet columns")
        rows = {}
        for row in reader:
            item = row["item"]
            if not item or item in rows or any(row[key] is None for key in IDENTITY + RATINGS):
                raise ValueError(f"{path.name}: missing fields or duplicate item")
            rows[item] = row
        return rows


def collect(reference, returned):
    if set(reference) != set(returned):
        raise ValueError("Returned worksheet must contain the same item IDs as the original")
    counts = {}
    candidates = []
    for item, original in reference.items():
        row = returned[item]
        if any(row[key] != original[key] for key in IDENTITY):
            raise ValueError(f"{item}: original cue fields were modified")
        stats = counts.setdefault(row["split"], Counter())
        stats["items"] += 1
        visual, meaning, start, end, evidence, notes = (row[key].strip() for key in RATINGS)
        if not any((visual, meaning, start, end, evidence, notes)):
            stats["unreviewed"] += 1
            continue
        if visual not in VISUAL or meaning not in MEANING or evidence not in EVIDENCE:
            raise ValueError(f"{item}: provide an allowed visual rating, meaning rating and evidence")
        stats["reviewed"] += 1
        stats[f"meaning_{meaning}"] += 1
        stats[f"evidence_{evidence}"] += 1
        if not start and not end:
            continue
        if not start or not end or evidence != "vidéo source":
            raise ValueError(f"{item}: both offsets require source-video evidence")
        try:
            corrected_start = int(row["start_ms"]) + int(start)
            corrected_end = int(row["end_ms"]) + int(end)
        except ValueError as error:
            raise ValueError(f"{item}: offsets must be integer milliseconds") from error
        if not 0 <= corrected_start < corrected_end:
            raise ValueError(f"{item}: corrected interval must be positive and ordered")
        stats["timing_reviewed"] += 1
        if meaning == "complète":
            candidates.append({
                **{key: original[key] for key in IDENTITY},
                "reviewed_start_ms": corrected_start,
                "reviewed_end_ms": corrected_end,
                "review_evidence": evidence,
                "visual_quality": visual,
                "meaning_alignment": meaning,
                "pose_status": "re-extraction and coverage verification required",
            })
            stats["interval_candidates"] += 1
    return counts, candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("original", type=Path)
    parser.add_argument("returned", type=Path)
    parser.add_argument("--output", type=Path, help="Private JSONL for reviewed interval candidates")
    args = parser.parse_args()
    counts, candidates = collect(worksheet(args.original), worksheet(args.returned))
    if args.output:
        with args.output.open("x", encoding="utf-8") as file:
            for row in candidates:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"by_split": counts, "interval_candidates": len(candidates)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
