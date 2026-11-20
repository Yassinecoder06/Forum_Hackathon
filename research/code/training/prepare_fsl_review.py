"""Build a private France-LSF review packet from the frozen training split."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from training.catalog import parse_stk, sha


def source_tokens(segment):
    result = []
    for token in segment["tokens"]:
        misc = dict(item.split("=", 1) for item in token[9].split("|") if "=" in item)
        start, end = misc.get("AlignBegin", ""), misc.get("AlignEnd", "")
        interval = None
        if start.isdecimal() and end.isdecimal() and int(end) > int(start):
            interval = [int(start), int(end)]
        result.append({"form": token[1], "features": token[5], "misc": misc, "interval_ms": interval})
    return result


def prepare(root):
    source = root / "raw"
    dataset = root / "v002"
    manifest = json.loads((dataset / "dataset_manifest.json").read_text())
    stk = source / "stk/fsl_stkautogramm-ud-test.conllu"
    sud_files = sorted((source / "sud").glob("*.conllu"))
    if {p.name for p in sud_files} != {n for n in manifest["raw_files"] if n.endswith(".conllu")}:
        raise ValueError("Unexpected SUD source file inventory")
    parent = json.loads((root / "v001/dataset_manifest.json").read_text())
    if sha(stk) != parent["source_sha256"]:
        raise ValueError("UD source checksum mismatch")
    for path in sud_files:
        if sha(path) != manifest["raw_files"][path.name]:
            raise ValueError(f"SUD source checksum mismatch in {path.name}")
    train_path = dataset / "train.jsonl"
    if sha(train_path) != manifest["files"]["train.jsonl"]:
        raise ValueError("Training split checksum mismatch")

    segments = {}
    for path in [stk, *sud_files]:
        for segment in parse_stk(path):
            identity = ("sud:" if path != stk else "") + segment["segment_id"]
            if identity in segments:
                raise ValueError(f"Duplicate source identity {identity}")
            segments[identity] = (path, segment)

    queue = []
    for line in train_path.read_text().splitlines():
        pair = json.loads(line)
        if (pair["language_variant"], pair["split"], pair["linguistic_review"]) != (
            "LSF_FR", "train", "PENDING_LOCAL_REVIEW"
        ):
            raise ValueError("Unexpected language, split, or review state")
        included = []
        for identity in pair["source_segment_ids"]:
            path, segment = segments[identity]
            if segment["recording"] != pair["recording"] or segment["document"] != pair["document"]:
                raise ValueError(f"Recording or document differs for {identity}")
            included.append({
                "id": identity,
                "source_file": str(path.relative_to(root)),
                "source_metadata": segment["metadata"],
                "tokens": source_tokens(segment),
            })
        tokens = [token for segment in included for token in segment["tokens"]]
        if " ".join(token["form"] for token in tokens) != pair["gloss_sequence"]:
            raise ValueError(f"Source FORM does not reproduce {pair['pair_id']}")
        spans = [token["interval_ms"] for token in tokens if token["interval_ms"]]
        queue.append({
            **pair,
            "pair_sha256": hashlib.sha256(line.encode()).hexdigest(),
            "interval_ms": [min(span[0] for span in spans), max(span[1] for span in spans)] if spans else None,
            "complete_token_timing": len(spans) == len(tokens),
            "source_segments": included,
            "media_status": "LOCAL_VIDEO_UNVERIFIED",
        })
    if len(queue) != manifest["counts"]["train"]:
        raise ValueError("Training row count differs from the frozen manifest")
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/text_to_gloss"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    queue = prepare(args.data)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=False)
    with (args.output / "source_evidence.jsonl").open("w") as evidence:
        for row in queue:
            evidence.write(json.dumps(row, ensure_ascii=False) + "\n")
    columns = ["pair_id", "pair_sha256", "document", "annotation_scheme", "start_ms", "end_ms",
               "decision", "reviewer_role", "reviewed_at", "notes"]
    with (args.output / "decisions.tsv").open("w", newline="") as decisions:
        writer = csv.writer(decisions, delimiter="\t")
        writer.writerow(columns)
        for row in queue:
            writer.writerow([*(row[key] for key in columns[:4]), *(row["interval_ms"] or ["", ""]),
                             "", "", "", ""])
    print(json.dumps({"rows": len(queue), "complete_timings": sum(r["complete_token_timing"] for r in queue),
                      "verified_local_videos": 0, "review_decisions": 0}))


if __name__ == "__main__":
    main()
