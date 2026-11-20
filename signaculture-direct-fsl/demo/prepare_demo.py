"""Prepare the private static site's assets from existing experiment outputs."""
import json
from pathlib import Path
import shutil
import subprocess

site = Path(__file__).resolve().parent
outputs = site.parent / "outputs"
assets = site / "assets"
assets.mkdir(exist_ok=True)
results = json.loads((outputs / "retrieval/evaluation.json").read_text())
demo = json.loads((outputs / "expanded/demo_001/prediction_metadata.json").read_text())
payload = {"samples": {}, "scores": {}}
training_dir = outputs / "training-demo"
training = json.loads((training_dir / "samples.json").read_text()) if training_dir.exists() else {}
for split in ("test", "new_test", "demo", *training):
    if split == "demo":
        source = outputs / "expanded/demo_001/predicted_skeleton.mp4"
        label = "Generated skeleton for an exploratory French input"
        caption = demo["text"]
        note = f"Model-only prediction · {demo['duration_seconds']:.2f} s predicted. Signing accuracy for this new sentence is unverified."
    elif split in training:
        source = training_dir / f"{split}.mp4"
        caption = training[split]["caption"]
        label = "Observed training skeleton and checkpoint prediction"
        note = ("This exact subtitle and its observed pose were in the training data. "
                f"Both views use the observed playback duration. The checkpoint predicted "
                f"{training[split]['generated_seconds']:.2f} s.")
    else:
        source = outputs / f"retrieval/{split}_comparison.mp4"
        label = "Observed pose, model prediction, average training motion, and blended caption retrieval"
        caption = results[split]["demo"]["caption"]
        note = ("Four views share the observed playback duration. "
                f"The model independently predicted {results[split]['demo']['generated_seconds']:.2f} s.")
        payload["scores"][split] = {name: data["joint"] for name, data in results[split]["methods"].items()}
    video = assets / f"{split}.mp4"
    poster = assets / f"{split}.jpg"
    shutil.copyfile(source, video)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", "0.20",
                    "-i", str(video), "-frames:v", "1", "-update", "1", str(poster)], check=True)
    payload["samples"][split] = {"video": f"assets/{video.name}", "poster": f"assets/{poster.name}",
                                 "label": label, "caption": caption, "note": note}
(assets / "results.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
