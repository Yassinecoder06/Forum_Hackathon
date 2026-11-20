import hashlib
import json
from pathlib import Path
import re

import numpy as np


def tokens(text):
    return re.findall(r"[^\W_]+", text.casefold())


def load(manifest, frames, root, required=("train", "val", "test")):
    rows = (json.loads(line) for line in Path(manifest).read_text().splitlines() if line.strip())
    return load_rows(rows, frames, root, required)


def load_rows(rows, frames, root, required=()):
    groups = {split: [] for split in ("train", "val", "test")}
    ownership = {}
    cache = {}
    for row in rows:
        if (row["source"], row["sign_language"], row["subtitle_language"]) != (
            "SignerX/SignVerse-2M", "fsl", "fr"
        ) or not row["permission_status"].startswith("derived_pose_metadata_noncommercial_research_only"):
            raise ValueError("Expected permitted SignVerse France-LSF with French subtitles")
        video, split = row["source_video_id"], row["split"]
        if ownership.setdefault(video, split) != split:
            raise ValueError("Recording leakage")
        path = Path(root) / row["pose_path"]
        key = path, row["pose_sha256"]
        if key not in cache:
            if hashlib.sha256(path.read_bytes()).hexdigest() != key[1]:
                raise ValueError("Pose checksum mismatch")
            with np.load(path, allow_pickle=False) as data:
                cache[key] = data["normalized_xy"], data["mask"]
        xy, mask = cache[key]
        a,b = row["pose_row_slice"]
        if xy.shape[1:] != (128,2) or mask.shape != xy.shape[:2] or not 0 <= a < b <= len(xy):
            raise ValueError("Unexpected pose schema or cue length")
        phase = np.rint(np.linspace(a,b-1,frames)).astype(int)
        valid = mask[phase].astype(bool) & np.isfinite(xy[phase]).all(-1)
        groups[split].append((row, np.where(valid[...,None],xy[phase],0).astype("float32"), valid, (b-a)/row["fps"]))
    if any(not groups[split] for split in required):
        raise ValueError("Missing required recording-disjoint split")
    return groups
