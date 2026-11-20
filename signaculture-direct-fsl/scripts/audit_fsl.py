"""Audit the pinned local SignVerse metadata without downloading pose archives."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path


def audit(metadata, output):
    with (metadata/"SignVerse-2M-metadata_split.csv").open(newline="") as file:
        reader=csv.DictReader(file)
        columns=reader.fieldnames
        split=list(reader)
    if not {"sign_language","video_id","split"}<=set(columns or []):
        raise ValueError(f"Unexpected split columns: {columns}")
    with (metadata/"Sign-DWPose-2M-metadata_ori.csv").open(newline="") as file:
        raw=list(csv.reader(file))
    if any(len(row)!=2 for row in raw):
        raise ValueError("Expected original headerless ID/language table")
    languages=dict(raw)
    inventory=json.loads((metadata/"hub_inventory.json").read_text())
    shards={Path(f["name"]).name:f["size"] for f in inventory["files"] if f["name"].endswith(".tar")}
    index=json.loads((metadata/"runtime_state/archive_upload_progress.json").read_text())["archives"]
    archives={}
    for name,item in index.items():
        for video in item["folders"]:
            if name in shards:
                archives.setdefault(video,set()).add(name)
    with (metadata/"runtime_state/SignVerse-2M-metadata_processed.csv").open(newline="") as file:
        processed={r["video_id"]:r for r in csv.DictReader(file)}
    rows=[]
    for row in split:
        if row["sign_language"]!="fsl":
            continue
        video=row["video_id"]
        extra=processed.get(video,{})
        if languages.get(video)!="fsl" or extra.get("sign_language","fsl")!="fsl":
            raise ValueError("Language metadata conflict")
        mapped=archives.get(video,set()).copy()
        if extra.get("archive_name") in shards:
            mapped.add(extra["archive_name"])
        rows.append({**row,"archives":";".join(sorted(mapped)),"subtitle_tracks":extra.get("subtitle_languages",""),
                     "revision":inventory["revision"],"license":"CC-BY-NC-4.0"})
    if len({r["video_id"] for r in rows})!=len(rows):
        raise ValueError("Duplicate FSL recording")
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open("w",newline="") as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    names={name for row in rows for name in row["archives"].split(";") if name}
    return {"revision":inventory["revision"],"columns":columns,"metadata_rows":len(split),
            "original_rows":len(raw),"fsl_videos":len(rows),"splits":dict(Counter(r["split"] for r in rows)),
            "mapped_videos":sum(bool(r["archives"]) for r in rows),"mapped_shards":len(names),
            "mapped_shard_bytes":sum(shards[n] for n in names),
            "full_inventory_shard_bytes":sum(shards.values()),
            "metadata_rows_with_subtitle_track":sum(bool(r["subtitle_tracks"]) for r in rows),
            "archive_count":len(shards),"clip_count_from_metadata":None}


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--metadata",type=Path,default=Path("../data/audit/private/signverse"))
    p.add_argument("--out",type=Path,default=Path("data/manifests/fsl_only.csv"))
    a=p.parse_args()
    print(json.dumps(audit(a.metadata,a.out),indent=2))
