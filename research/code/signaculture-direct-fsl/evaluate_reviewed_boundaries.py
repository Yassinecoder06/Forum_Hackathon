"""Compare identical predictions against original and reviewed held-out pose intervals."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import average, restore, score
from scripts.prepare_reviewed_intervals import records
from src.dataset import load, load_rows
from src.text_encoder import encode, french_features


def paired_scores(model, state, training, original, corrected, cache):
    if not corrected:
        return {"paired_cues": 0, "status": "awaiting reviewed held-out intervals"}
    ids = [row["id"] for row, *_ in corrected]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate reviewed cue")
    mask = np.stack([item[2] for item in training])
    counts = mask.sum(axis=0)
    support = counts > 0
    template = np.stack([item[1] for item in training]).sum(axis=0) / counts.clip(min=1)[..., None]
    template_duration = float(np.mean([item[3] for item in training]))
    texts = {row["id"]: row["french_caption"] for row, *_ in corrected}
    if state["config"].get("encoder") == "frozen_french":
        if cache:
            if any(cache["texts"].get(cue) != text for cue, text in texts.items()):
                raise ValueError("Cached captions differ from the reviewed cue set")
            inputs = torch.stack([cache["features"][cue] for cue in ids])
            if inputs.shape != (len(ids), 48, 768) or not inputs.isfinite().all():
                raise ValueError("Invalid cached French features")
        else:
            inputs = french_features(list(texts.values()), "cpu")
    else:
        inputs = torch.tensor([encode(text, state["vocab"]) for text in texts.values()])
    with torch.inference_mode():
        predicted, seconds = model(inputs)
        predicted = predicted.numpy()
        seconds = seconds.exp().clamp(.25, 15).numpy()
    metrics = defaultdict(lambda: defaultdict(list))
    for i, (row, new_xy, new_mask, new_duration) in enumerate(corrected):
        old_row, old_xy, old_mask, old_duration = original[row["id"]]
        preserved = ("source_video_id", "split", "pose_path", "pose_sha256", "french_caption")
        if any(row[key] != old_row[key] for key in preserved):
            raise ValueError("Reviewed and original cue provenance differs")
        common = support & old_mask & new_mask
        cohort = row["evaluation_cohort"]
        if cohort not in ("val", "test", "new_test"):
            raise ValueError("Reviewed evaluation requires a held-out cohort")
        for boundary, target, duration in (("subtitle", old_xy, old_duration),
                                           ("reviewed", new_xy, new_duration)):
            for method, motion, estimate in (("model", predicted[i], float(seconds[i])),
                                             ("phase_template", template, template_duration)):
                metrics[cohort][f"{method}_{boundary}"].append(
                    score(motion, target, common, common, estimate, duration))
    return {"paired_cues": len(ids), "by_cohort": {
        cohort: {"cues": len(values["model_reviewed"]),
                 "metrics": {name: average(rows) for name, rows in values.items()}}
        for cohort, values in metrics.items()
    }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reviewed", type=Path)
    parser.add_argument("--original", required=True, type=Path)
    parser.add_argument("--additional", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=Path(".."))
    parser.add_argument("--french-features", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, state = restore(args.checkpoint)
    if hashlib.sha256(args.original.read_bytes()).hexdigest() != state["manifest_sha256"]:
        raise ValueError("Original manifest differs from the checkpoint manifest")
    frames = state["config"]["frames"]
    original = load(args.original, frames, args.root)
    additional = load(args.additional, frames, args.root, required=("test",))
    occupied = {row["source_video_id"] for split in original.values() for row, *_ in split}
    if additional["train"] or additional["val"] or any(
        row["source_video_id"] in occupied for row, *_ in additional["test"]
    ):
        raise ValueError("Additional recordings must be unseen test data")
    indexed = {row["id"]: item for item in [*original["val"], *original["test"], *additional["test"]]
               for row in [item[0]]}
    expected_cohort = {item[0]["id"]: cohort for cohort, samples in
                       (("val", original["val"]), ("test", original["test"]), ("new_test", additional["test"]))
                       for item in samples}
    rows = [row for row in records(args.reviewed) if row["split"] != "train"]
    for row in rows:
        if row.get("alignment_status") != "source-video reviewed meaning and timing" or (
            expected_cohort.get(row["id"]) != row["evaluation_cohort"]
        ):
            raise ValueError("Expected reviewed held-out provenance")
    groups = load_rows(rows, frames, args.root)
    cache = torch.load(args.french_features, map_location="cpu", weights_only=True) if args.french_features else None
    report = paired_scores(model, state, original["train"], indexed, groups["val"] + groups["test"], cache)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
