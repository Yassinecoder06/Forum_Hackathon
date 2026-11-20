"""Validate a returned France-LSF worksheet against the frozen source records."""
import argparse
from collections import Counter
import csv
from datetime import date
import json
from pathlib import Path

from training.prepare_fsl_review import prepare


KEYS = ("pair_id", "pair_sha256", "document", "annotation_scheme")
FIELDS = ("decision", "reviewer_role", "reviewed_at", "notes")
TIMING = ("start_ms", "end_ms")
DECISIONS = {"accept", "reject", "needs_context", "needs_alignment", "video_unavailable"}


def validate(queue, worksheet):
    expected = {row["pair_id"]: row for row in queue}
    seen = set()
    decisions = []
    with worksheet.open(encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source, delimiter="\t")
        if tuple(reader.fieldnames or ()) != (*KEYS, *TIMING, *FIELDS):
            raise ValueError("Unexpected worksheet header")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed worksheet row near line {reader.line_num}")
            pair_id = row["pair_id"]
            original = expected.get(pair_id)
            if original is None or pair_id in seen:
                raise ValueError(f"Unknown or duplicate pair ID: {pair_id}")
            seen.add(pair_id)
            if any(row[key] != original[key] for key in KEYS):
                raise ValueError(f"Source identity differs for {pair_id}")
            interval = original["interval_ms"] or ["", ""]
            if [row[key] for key in TIMING] != list(map(str, interval)):
                raise ValueError(f"Source interval differs for {pair_id}")
            decision = {key: row[key].strip() for key in FIELDS}
            if not any(decision.values()):
                continue
            if decision["decision"] not in DECISIONS or not decision["reviewer_role"]:
                raise ValueError(f"Missing reviewer role or invalid decision for {pair_id}")
            reviewed_at = date.fromisoformat(decision["reviewed_at"])
            if reviewed_at.isoformat() != decision["reviewed_at"] or reviewed_at > date.today():
                raise ValueError(f"Expected a review date in YYYY-MM-DD for {pair_id}")
            if not decision["notes"]:
                raise ValueError(f"Missing review notes for {pair_id}")
            decisions.append({**{key: row[key] for key in KEYS}, **decision})
    if seen != set(expected):
        raise ValueError(f"Worksheet is missing {len(set(expected) - seen)} original rows")
    return decisions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worksheet", type=Path)
    parser.add_argument("--data", type=Path, default=Path("data/text_to_gloss"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = prepare(args.data)
    decisions = validate(queue, args.worksheet)
    result = {
        "source_rows": len(queue),
        "undecided_rows": len(queue) - len(decisions),
        "decision_counts": dict(Counter(row["decision"] for row in decisions)),
        "catalog_updates": 0,
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump({**result, "decisions": decisions}, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
