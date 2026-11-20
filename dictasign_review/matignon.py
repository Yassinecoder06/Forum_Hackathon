"""Run the extracted notebook 07 code on the same samples as the model comparison."""
import argparse
import importlib.util
import json
from pathlib import Path

import cv2
import numpy as np

from .extract_pose import ROOT, WORK

BODY = [11, 12, 14, 16, 13, 15, 23, 24, 25, 27, 26, 28]


def points(array):
    return [p[:2].tolist() if np.isfinite(p[:2]).all() else None for p in array]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extractor", type=Path)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("matignon_notebook_extractor", args.extractor)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cv2.setNumThreads(1)
    results = WORK / "results"
    destination = results / "matignon-full-frame"
    destination.mkdir(parents=True, exist_ok=True)
    for path in sorted((results / "mmpose").glob("*.json")):
        reference = json.loads(path.read_text())
        target = destination / path.name
        if target.is_file():
            cached = json.loads(target.read_text())
            if (cached.get("model") == "matignon-07"
                and [f["time_ms"] for f in cached["frames"]] == [f["time_ms"] for f in reference["frames"]]):
                continue
        source = ROOT / ".runtime" / "captures" / path.stem / "source.mp4"
        cap = cv2.VideoCapture(str(source))
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        if min(fps, count, width, height) <= 0:
            raise ValueError(f"Cannot decode {source}")
        # Match the notebook's worker resize rule, with the crop set to the full image.
        scale = float(np.clip(560 / width, 1.0, 2.0))
        xyz, _, _, _, _ = module.extract(str(source), box=None, dur_s=(count + .25) / fps, scale=scale)
        frames = []
        for sample in reference["frames"]:
            time = sample["time_ms"]
            index = round(time * fps / 1000)
            if index >= len(xyz):
                raise ValueError(f"Missing source frame {index} in {source}")
            raw = xyz[index].astype(float)
            body = points(raw[:23])
            frame = {"time_ms": time, "body": [body[i] if i < len(body) else None for i in BODY],
                     "face": points(raw[65:])}
            shoulder = np.linalg.norm((raw[11, :2] - raw[12, :2]) * [width / height, 1])
            for side, start in (("left", 23), ("right", 44)):
                hand = raw[start:start + 21]
                if np.isfinite(hand).all():
                    frame[side] = points(hand)
                    if np.isfinite(shoulder) and shoulder >= .035:
                        frame[f"{side}_depth"] = (-(hand[:, 2] - hand[0, 2]) * .34 * (width / height) / shoulder).tolist()
            frames.append(frame)
        output = {"model": "matignon-07", "checkpoint": f"Notebook 07 · MediaPipe {module.mp.__version__}",
                  "device": "cpu", "source_fps": reference["source_fps"],
                  "sample_interval_ms": reference["sample_interval_ms"], "frames": frames}
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(output, separators=(",", ":"), allow_nan=False))
        temporary.replace(target)
        print(f"{path.stem}: {len(frames)} matched frames", flush=True)


if __name__ == "__main__":
    main()
