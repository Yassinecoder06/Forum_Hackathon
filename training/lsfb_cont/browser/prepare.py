"""Prepare a private LSFB-CONT browse page from existing local review excerpts."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    payload = re.search(r'<script[^>]+id="payload"[^>]*>(.*?)</script>',
                        (args.review / "index.html").read_text(), re.S)
    if payload is None:
        raise ValueError("Expected the LSFB review export")
    clips = args.output / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows = []
    for row in json.loads(payload[1]):
        if not row.get("clip"):
            continue
        source = args.review / "clips" / Path(row["clip"]).name
        name = f"{len(rows) + 1:03}.mp4"
        shutil.copyfile(source, clips / name)
        rows.append({
            "clip": f"clips/{name}", "split": row["split"], "french_text": row["french_text"],
            "interval_ms": row["interval_ms"], "clip_start_ms": row["clip_start_ms"],
            "hand_tiers": {tier: [{key: event.get(key) for key in ("start_ms", "end_ms", "gloss", "lemma")}
                                  for event in events] for tier, events in row["hand_tiers"].items()},
        })
    for name in ("index.html", "app.js", "style.css"):
        shutil.copyfile(Path(__file__).with_name(name), args.output / name)
    (args.output / "data.json").write_text(json.dumps(rows, ensure_ascii=False))
    print(dict(Counter(row["split"] for row in rows)))


if __name__ == "__main__":
    main()
