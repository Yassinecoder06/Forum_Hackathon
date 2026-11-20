"""Compare text retrieval with a caption-independent phase template."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


def load_samples(manifest):
    cache = {}
    samples = []
    for line in manifest.read_text().splitlines():
        row = json.loads(line)
        if row["sign_language"] != "fsl" or row["subtitle_language"] != "fr":
            raise ValueError("Expected France-LSF with released French subtitles")
        path = Path(row["pose_path"])
        if path not in cache:
            if hashlib.sha256(path.read_bytes()).hexdigest() != row["pose_sha256"]:
                raise ValueError("Pose checksum mismatch")
            with np.load(path, allow_pickle=False) as data:
                cache[path] = (data["normalized_xy"], data["mask"])
        xy, mask = cache[path]
        a, b = row["pose_row_slice"]
        if not 0 <= a < b <= len(xy):
            raise ValueError("Pose slice out of bounds")
        phase = np.rint(np.linspace(a, b - 1, 32)).astype(int)
        samples.append((row, xy[phase], mask[phase]))
    return samples


def evaluate(samples):
    groups = {k: {s[0]["source_video_id"] for s in samples if s[0]["split"] == k}
              for k in ("train", "val", "test")}
    if any(not v for v in groups.values()) or sum(map(len, groups.values())) != len(set.union(*groups.values())):
        raise ValueError("Expected three nonempty, recording-disjoint splits")
    training = [s for s in samples if s[0]["split"] == "train"]
    train_xy = np.stack([s[1] for s in training])
    train_mask = np.stack([s[2] for s in training])
    counts = train_mask.sum(axis=0)
    template = np.where(train_mask[..., None], train_xy, 0).sum(axis=0) / np.maximum(counts[..., None], 1)

    def tokens(row):
        return Counter(re.findall(r"\w+", row["french_caption"].casefold()))

    train_tokens = [tokens(s[0]) for s in training]
    df = Counter(token for row in train_tokens for token in row)
    idf = {token: 1 + math.log((1 + len(training)) / (1 + n)) for token, n in df.items()}

    def vector(row):
        weights = {t: n * idf[t] for t, n in row.items() if t in idf}
        norm = math.sqrt(sum(w * w for w in weights.values())) or 1
        return {t: w / norm for t, w in weights.items()}

    vectors = [vector(row) for row in train_tokens]
    report = {"training_samples": len(training), "phase_samples": 32, "results": {}}
    for split in ("val", "test"):
        held_out = [s for s in samples if s[0]["split"] == split]
        sums = Counter()
        for row, xy, valid in held_out:
            query = vector(tokens(row))
            similarity = [sum(w * v.get(t, 0) for t, w in query.items()) for v in vectors]
            best = max(range(len(similarity)), key=similarity.__getitem__)
            if similarity[best] > 0:
                retrieval = np.where(train_mask[best, ..., None], train_xy[best], template)
            else:
                retrieval = template
                sums["caption_no_training_overlap"] += 1
            common = valid & (counts > 0)
            for name, region in {"all": slice(None), "hands": slice(18, 60)}.items():
                mask = common[:, region]
                sums[f"{name}_valid_reference"] += int(valid[:, region].sum())
                sums[f"{name}_scored"] += int(mask.sum())
                for method, prediction in (("template", template), ("caption_retrieval", retrieval)):
                    dist = np.linalg.norm(prediction[:, region] - xy[:, region], axis=-1)
                    sums[f"{method}_{name}_error_sum"] += float(dist[mask].sum())
        report["results"][split] = {
            "samples": len(held_out), "recordings": len(groups[split]), "counts": dict(sums),
            "mean_normalized_2d_error": {
                method: {name: sums[f"{method}_{name}_error_sum"] / sums[f"{name}_scored"]
                         for name in ("all", "hands")}
                for method in ("template", "caption_retrieval")
            },
        }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(load_samples(args.manifest))
    report["manifest_sha256"] = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    with args.output.open("x") as output:
        json.dump(report, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
