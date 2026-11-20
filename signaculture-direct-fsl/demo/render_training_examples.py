"""Render observed and predicted motion for captions in the checkpoint's train split."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from evaluate import restore
from scripts.visualize_poses import render
from src.dataset import load


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--features", required=True, type=Path)
    p.add_argument("--root", type=Path, default=Path(".."))
    p.add_argument("--output", type=Path, default=Path("outputs/training-demo"))
    args = p.parse_args()
    torch.set_num_threads(2)
    model, state = restore(args.checkpoint)
    if hashlib.sha256(args.manifest.read_bytes()).hexdigest() != state["manifest_sha256"]:
        raise ValueError("Manifest differs from the checkpoint's training data")
    train = load(args.manifest, state["config"]["frames"], args.root)["train"]
    by_id = {item[0]["id"]: item for item in train}
    cached = torch.load(args.features, map_location="cpu", weights_only=True)
    texts = cached["texts"]
    if (state["config"].get("encoder") != "frozen_french" or not texts
            or cached["features"].keys() != texts.keys()
            or any(cue not in by_id or by_id[cue][0]["french_caption"] != text for cue, text in texts.items())):
        raise ValueError("Expected exact checkpoint training captions and frozen-encoder features")
    features = torch.stack([cached["features"][cue] for cue in texts])
    if features.shape != (len(texts), 48, 768) or not features.isfinite().all():
        raise ValueError("Invalid encoder features")
    with torch.inference_mode():
        predicted, seconds = model(features)
        predicted, seconds = predicted.numpy(), seconds.exp().clamp(.25, 15).numpy()
    args.output.mkdir(parents=True, exist_ok=True)
    samples = {}
    for index, (cue, caption) in enumerate(texts.items(), 1):
        _, xy, mask, observed = by_id[cue]
        sample = f"train_{index}"
        render(args.output / f"{sample}.mp4",
               [("Observed training pose", xy, mask), ("Checkpoint prediction", predicted[index-1], mask)],
               len(xy) / observed)
        samples[sample] = {"caption": caption, "observed_seconds": observed,
                           "generated_seconds": float(seconds[index-1])}
    (args.output / "samples.json").write_text(json.dumps(samples, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
