"""Resumable matched-frame coverage benchmark for the local capture library."""

import argparse
from collections import Counter
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import cv2
import numpy as np

from .extract_pose import INVENTORY, ROOT, WORK, normalized_groups, sapiens_predictor

OUTPUT = ROOT / ".runtime/all-video-cv-benchmark"
CAPTURES = ROOT / ".runtime/captures"
SAPIENS = WORK / "models/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2"
STRIDE = 2


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
    temporary.replace(path)


def manifest():
    inventory = {
        str(uuid5(NAMESPACE_URL, entry["video_url"])): entry
        for entry in json.loads(INVENTORY.read_text())["entries"]
    }
    rows = []
    for video in sorted(CAPTURES.glob("*/source.mp4")):
        with video.open("rb") as stream:
            fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
        entry = inventory.get(video.parent.name)
        rows.append({
            "id": video.parent.name, "source": str(video), "sha256": fingerprint,
            "group": "Dicta-Sign LSF" if entry else "Other local captures",
            "concept": entry["concept_fr"] if entry else "",
        })
    write(OUTPUT / "manifest.json", rows)
    print(f"{len(rows)} files, {len({row['sha256'] for row in rows})} unique videos", flush=True)


def video_info(source):
    cap = cv2.VideoCapture(source)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if not cap.isOpened() or not np.isfinite(fps) or min(fps, width, height) <= 0:
            raise ValueError("Video or frame rate is unreadable")
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if np.isfinite(count) and 0 < count < 10_000_000:
            count = int(count)
        else:
            count = 0
            while cap.grab():
                count += 1
        if not count:
            raise ValueError("No decodable frames")
        return fps, width, count
    finally:
        cap.release()


def samples(source):
    cap = cv2.VideoCapture(source)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Video or frame rate is unreadable")
        index = 0
        while cap.grab():
            if index % STRIDE == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    raise ValueError(f"Cannot retrieve frame {index}")
                yield index, round(1000 * index / fps), frame
            index += 1
    finally:
        cap.release()


def notebook_predictor(path):
    spec = importlib.util.spec_from_file_location("matignon_notebook_extractor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def infer(row):
        fps, width, expected = video_info(row["source"])
        scale = float(np.clip(560 / width, 1., 2.))
        _, found, _, _, _ = module.extract(row["source"], box=None, scale=scale,
                                           dur_s=(expected + .25) / fps, log=lambda *_: None)
        return [[i, round(i * 1000 / fps), bool(left), bool(right), bool(body), bool(face)]
                for i, (body, left, right, face) in enumerate(found) if i % STRIDE == 0]

    return infer


def sapiens_stage():
    import torch

    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("Benchmark needs the same BF16 CUDA path as the 100-clip run")
    torch._C._jit_set_autocast_mode(False)
    predictor = sapiens_predictor(SAPIENS, "cuda", WORK / "sapiens/lite/demo")

    def infer(row):
        frames = []
        for index, time, image in samples(row["source"]):
            height, width = image.shape[:2]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                xy, confidence = predictor(image)
            pose = normalized_groups(xy, confidence.reshape(-1), width, height, "sapiens")
            frames.append([index, time, pose["left"], pose["right"], all(pose["body"][:6])])
        return frames

    return infer


def hamer_stage():
    from .hamer import load_predictor

    predict, _ = load_predictor(WORK / "models/hamer", WORK / "hamer-source", low_memory=True)

    def infer(row):
        source = json.loads((OUTPUT / "sapiens" / f"{row['sha256']}.json").read_text())
        if source.get("error"):
            raise ValueError(source["error"])
        frames = []
        for n, (index, time, image) in enumerate(samples(row["source"])):
            i, t, left, right, body_complete = source["frames"][n]
            if (i, t) != (index, time):
                raise ValueError("Sapiens and HaMeR sample indices differ")
            hints = {
                key: [{"x": p[0], "y": p[1]} for p in hand if p]
                for key, hand in (("LLM", left), ("RLM", right))
            }
            hands = predict(image, hints)
            mesh = tuple(side in hands for side in ("left", "right"))
            usable = tuple(present or len(hints[key]) == 21 for present, key in zip(mesh, ("LLM", "RLM")))
            frames.append([index, time, *usable, *mesh, body_complete])
        if len(frames) != len(source["frames"]):
            raise ValueError("Sapiens and HaMeR frame counts differ")
        return frames

    return infer


def run(stage, rows, infer):
    folder = OUTPUT / stage
    folder.mkdir(exist_ok=True)
    unique = {row["sha256"]: row for row in rows}
    start = perf_counter()
    for index, (fingerprint, row) in enumerate(unique.items(), 1):
        destination = folder / f"{fingerprint}.json"
        if destination.is_file():
            continue
        try:
            frames = infer(row)
            if not frames:
                raise ValueError("No decoded samples")
            result = {"frames": frames}
        except (ValueError, OSError, IndexError, cv2.error) as exc:
            result = {"error": str(exc)}
        write(destination, result)
        if index % 25 == 0 or index == len(unique):
            print(f"{stage}: {index}/{len(unique)} unique files, {perf_counter() - start:.0f}s", flush=True)


def report(rows):
    records = []
    groups = {}
    seen = set()
    for row in rows:
        result = dict(row)
        data = []
        for stage in ("notebook", "hamer"):
            path = OUTPUT / stage / f"{row['sha256']}.json"
            data.append(json.loads(path.read_text()) if path.is_file() else {"error": "pending"})
        notebook, combined = data
        if "error" in notebook or "error" in combined:
            result["status"] = f"Notebook: {notebook.get('error', 'OK')}; Sapiens + HaMeR: {combined.get('error', 'OK')}"
        elif [f[:2] for f in notebook["frames"]] != [f[:2] for f in combined["frames"]]:
            result["status"] = "Frame sample mismatch"
        else:
            a, b = notebook["frames"], combined["frames"]
            result.update(status="paired", frames=len(a),
                          notebook_left=sum(f[2] for f in a), notebook_right=sum(f[3] for f in a),
                          notebook_both=sum(f[2] and f[3] for f in a),
                          sapiens_hamer_left=sum(f[2] for f in b), sapiens_hamer_right=sum(f[3] for f in b),
                          sapiens_hamer_both=sum(f[2] and f[3] for f in b),
                          hamer_mesh_both=sum(f[4] and f[5] for f in b),
                          notebook_body_detected=sum(f[4] for f in a),
                          sapiens_complete_upper_body=sum(f[6] for f in b))
            categories = ["All captures", row["group"]]
            if row["sha256"] not in seen:
                categories.append("Unique videos")
                seen.add(row["sha256"])
            for group in categories:
                counts = groups.setdefault(group, Counter())
                counts.update({key: value for key, value in result.items() if isinstance(value, int)})
                counts["clips"] += 1
                difference = result["sapiens_hamer_both"] - result["notebook_both"]
                counts["clip_win_combined" if difference > 0 else "clip_win_notebook" if difference < 0 else "clip_tie"] += 1
        records.append(result)
    pending = sum("pending" in r["status"] for r in records)
    summary = {
        "inventory_files": len(rows), "unique_videos": len({r["sha256"] for r in rows}),
        "stride": STRIDE, "pending_files": pending,
        "excluded_files": sum(r["status"] != "paired" and "pending" not in r["status"] for r in records),
        "groups": groups,
    }
    write(OUTPUT / "summary.json", summary)
    with (OUTPUT / "per-video.csv").open("w", newline="") as file:
        fields = list(dict.fromkeys(key for row in records for key in row))
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    print(json.dumps(summary, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("manifest", "notebook", "sapiens", "hamer", "report"))
    parser.add_argument("--extractor", type=Path, default=ROOT / ".runtime/matignon-cv-review/extract_poses.py")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True)
    cv2.setNumThreads(1)
    if args.stage == "manifest":
        manifest()
        return
    rows = json.loads((OUTPUT / "manifest.json").read_text())
    if args.stage == "report":
        report(rows)
        return
    factories = {"notebook": lambda: notebook_predictor(args.extractor), "sapiens": sapiens_stage, "hamer": hamer_stage}
    run(args.stage, rows, factories[args.stage]())


if __name__ == "__main__":
    main()
