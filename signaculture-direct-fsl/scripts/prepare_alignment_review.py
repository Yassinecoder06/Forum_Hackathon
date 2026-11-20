"""Render a private, recording-balanced worksheet for a fluent France-LSF reviewer."""
import argparse
from collections import defaultdict
import csv
from pathlib import Path
import sys
import shutil

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from render_alignment_worksheet import render as render_worksheet
from scripts.visualize_poses import render
from src.dataset import load


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--additional", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    groups = load(args.root / args.manifest, 24, args.root)
    if args.additional:
        groups["new_test"] = load(args.root / args.additional, 24, args.root, required=("test",))["test"]
    args.output.mkdir(parents=True, exist_ok=True)
    if any((args.output / name).exists() for name in ("review.csv", "reference.csv")):
        parser.error("Choose a new output directory to preserve the existing worksheet")
    fields = ["item", "split", "caption", "source_video_id", "start_ms", "end_ms", "visual_quality",
              "meaning_alignment", "start_offset_ms", "end_offset_ms", "evidence", "notes"]
    with (args.output / "review.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for split, examples in groups.items():
            recordings = defaultdict(list)
            for example in examples:
                recordings[example[0]["source_video_id"]].append(example)
            for index, samples in enumerate(recordings.values(), 1):
                samples.sort(key=lambda example: example[0]["start_ms"])
                for cue, sample in enumerate((samples[i] for i in sorted({len(samples)//4, 3*len(samples)//4})), 1):
                    row, xy, mask, duration = sample
                    item = f"{split}-{index:02d}-{cue}"
                    render(args.output / f"{item}.mp4", [("Trajectoire observée", xy, mask)], len(xy)/duration)
                    caption = row["french_caption"]
                    if caption.lstrip().startswith(("=", "+", "-", "@")):
                        caption = "'" + caption
                    writer.writerow(dict(item=item, split=split, caption=caption,
                                         source_video_id=row["source_video_id"], start_ms=row["start_ms"],
                                         end_ms=row["end_ms"]))
    reference = args.output / "reference.csv"
    shutil.copyfile(args.output / "review.csv", reference)
    render_worksheet(reference, args.output / "index.html")


if __name__ == "__main__":
    main()
