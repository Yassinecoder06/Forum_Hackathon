"""List verified pinned archive objects for a small recording-disjoint shortlist."""
import argparse
import csv
import json
from pathlib import Path


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--manifest",type=Path,default=Path("data/manifests/fsl_only.csv"))
    p.add_argument("--inventory",type=Path,default=Path("../../../data/audit/private/signverse/hub_inventory.json"))
    p.add_argument("--per-split",type=int,choices=range(1,11),default=3)
    a=p.parse_args()
    inventory=json.loads(a.inventory.read_text())
    files={Path(f["name"]).name:f for f in inventory["files"] if f["name"].endswith(".tar")}
    with a.manifest.open(newline="") as file:
        rows=list(csv.DictReader(file))
    for split in ("train","val","test"):
        eligible=[]
        for row in rows:
            if row["sign_language"]!="fsl" or row["revision"]!=inventory["revision"]:
                raise ValueError("Manifest provenance mismatch")
            if row["split"]!=split or not row["archives"]:
                continue
            best=min((files[n] for n in row["archives"].split(";")),key=lambda f:f["size"])
            eligible.append((best["size"],row["video_id"],best["name"]))
        for size,video,name in sorted(eligible)[:a.per_split]:
            print(json.dumps({"split":split,"video_id":video,"archive":name,"bytes":size,
                              "subtitle_track_verified":False}))
