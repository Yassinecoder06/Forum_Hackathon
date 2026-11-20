"""Audit exact FORM-to-local-motion label coverage without semantic substitution."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
import unicodedata

from fastapi import HTTPException

from backend.app.main import asset, registry

PLAIN_FORM = re.compile(r"\[([^\[\]]+)\]")


def key(value):
    return unicodedata.normalize("NFC", value).strip().casefold()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/parallel"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--queue", type=Path, help="Private prioritized rights and coverage review queue")
    args = parser.parse_args()
    manifest = json.loads((args.data / "manifest.json").read_text())
    root, glosses, motions = registry()
    real = {motion.motion_id: motion for motion in motions if not motion.synthetic}
    by_label = defaultdict(list)
    for gloss in glosses:
        if gloss.motion_id in real:
            by_label[key(gloss.gloss)].append(gloss)

    checked = {}
    missing = defaultdict(Counter)
    result = {"method": "casefolded exact canonical gloss label; plain bracketed UD FORM only",
              "registry_real_motions": len(real), "splits": {}}
    for split in ("train", "val"):
        path = args.data / f"fsl_ud_feasibility_{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"][split]:
            raise ValueError("Frozen split checksum mismatch")
        counts = Counter()
        for line in path.read_text().splitlines():
            row = json.loads(line)
            if row["sign_language"] != "fsl" or row["annotation_convention"] != "UD-FSL-STK-FORM":
                raise ValueError("Unexpected language or annotation convention")
            counts["examples"] += 1
            labels = []
            for form in row["gloss_sequence"]:
                matched = PLAIN_FORM.fullmatch(form)
                if matched:
                    labels.append(key(matched[1]))
                else:
                    counts["other_form_occurrences"] += 1
            counts["plain_form_occurrences"] += len(labels)
            found = cleared = approved = 0
            for label in labels:
                choices = []
                for gloss in by_label[label]:
                    motion = real[gloss.motion_id]
                    if motion.motion_id not in checked:
                        try:
                            asset(root, motion)
                            checked[motion.motion_id] = True
                        except HTTPException:
                            checked[motion.motion_id] = False
                    if checked[motion.motion_id]:
                        choices.append((gloss, motion))
                found += bool(choices)
                rights_cleared = any(m.rights_status == "cleared" for _, m in choices)
                cleared += rights_cleared
                if not choices:
                    missing[label][f"{split}_no_valid_motion"] += 1
                if not rights_cleared:
                    missing[label][f"{split}_no_rights_cleared_motion"] += 1
                approved += any(g.review_status == m.animation_review == "approved" and
                                m.rights_status == "cleared" for g, m in choices)
            counts["exact_label_available"] += found
            counts["exact_label_rights_cleared"] += cleared
            counts["exact_label_rights_and_review_approved"] += approved
            counts["examples_all_plain_forms_available"] += bool(labels) and found == len(labels)
        result["splits"][split] = dict(counts)
    result["candidate_motion_files_checked"] = len(checked)
    result["candidate_motion_files_valid"] = sum(checked.values())
    if args.queue:
        inventory = defaultdict(list)
        commons = json.loads((root / "data/source_inventory/commons_lsf.json").read_text())
        for entry in commons["entries"]:
            if entry["language_variant"] == "LSF_FR" and entry["eligible_for_motion"]:
                inventory[key(entry["concept_fr"])].append({
                    "source": "Commons", "match": "French concept label",
                    "source_url": entry["source_url"], "license": entry["license"],
                })
        dicta = json.loads((root / "data/source_inventory/dictasign_lsf.json").read_text())
        for entry in dicta["entries"]:
            if entry["language"] == "fsl":
                inventory[key(entry["canonical_gloss"])].append({
                    "source": "Dicta-Sign", "match": "canonical gloss label",
                    "source_url": entry["page_url"], "rights_status": entry["rights_status"],
                })
        queue = [{"label": label, "counts": dict(counts), "inventory_candidates": inventory[label]}
                 for label, counts in sorted(missing.items(), key=lambda item: (
                     -item[1]["train_no_valid_motion"], -item[1]["train_no_rights_cleared_motion"], item[0]))]
        result["inventory_followup"] = {
            "labels_without_valid_motion": sum(any(k.endswith("_no_valid_motion") for k in row["counts"]) for row in queue),
            "labels_without_rights_cleared_motion": len(queue),
            "unavailable_labels_with_inventory_candidate": sum(bool(row["inventory_candidates"])
                and any(k.endswith("_no_valid_motion") for k in row["counts"]) for row in queue),
        }
        with args.queue.open("x") as output:
            json.dump(queue, output, ensure_ascii=False, indent=2)
            output.write("\n")
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
