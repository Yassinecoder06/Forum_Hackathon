"""Select a bounded split-specific FSL sample from one locally cached official TAR."""
import argparse
import csv
import json
from pathlib import Path, PurePosixPath
import re
import tarfile

POS_LIMIT=16*1024**2
VTT_LIMIT=512*1024
DISK_LIMIT=256*1024**2


def stage(archive, manifest, output, limit, split="train", max_archive_bytes=DISK_LIMIT):
    if archive.stat().st_size>max_archive_bytes:
        raise ValueError("Archive exceeds configured byte budget")
    with manifest.open(newline="") as file:
        selected={r["video_id"]:r for r in csv.DictReader(file)
                  if r["sign_language"]=="fsl" and r["split"]==split
                  and archive.name in r["archives"].split(";")}
    existing={path.name for path in output.iterdir() if path.is_dir()} if output.exists() else set()
    files={}
    with tarfile.open(archive,"r:") as tar:
        for member in tar:
            path=PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or member.issym() or member.islnk():
                raise ValueError("Unsafe archive member")
            if not member.isfile() or not path.parts:
                continue
            video=path.parts[0]
            if not re.fullmatch(r"[\w-]{11}",video,flags=re.ASCII) or video not in selected or video in existing:
                continue
            if path.name=="poses.npz" and 0<member.size<=POS_LIMIT:
                kind="pose"
            elif path.name==f"{video}.fr.vtt" and 0<member.size<=VTT_LIMIT:
                kind="subtitle"
            else:
                continue
            by_video=files.setdefault(video,{})
            if kind in by_video:raise ValueError("Duplicate record member")
            by_video[kind]=member
        candidates=[]
        for video,items in files.items():
            if set(items)!={"pose","subtitle"}:continue
            with tar.extractfile(items["subtitle"]) as file:
                vtt=file.read()
            if not vtt.decode("utf-8-sig").startswith("WEBVTT") or not 1<=vtt.count(b" --> ")<=200:
                continue
            candidates.append((video,items["pose"].size,vtt))
        used=0
        accepted=0
        for video,size,vtt in sorted(candidates,key=lambda x:x[1]):
            if accepted==limit: break
            if used+size+len(vtt)>64*1024**2:continue
            row=selected[video]
            directory=output/video
            directory.mkdir(parents=True,exist_ok=False)
            with tar.extractfile(files[video]["pose"]) as file:
                (directory/"poses.npz").write_bytes(file.read())
            (directory/f"{video}.fr.vtt").write_bytes(vtt)
            (directory/"manifest.json").write_text(json.dumps({
                **row,"source_video_id":video,"source":"SignerX/SignVerse-2M","signer_id":None,
                "permission_status":"derived_pose_metadata_noncommercial_research_only; original_video_rights_not_conveyed",
                "archive":archive.name
            },indent=2))
            accepted+=1
            used+=size+len(vtt)
    return {"split":split,"eligible_recordings":len(selected),"bounded_french_candidates":len(candidates),
            "staged_recordings":accepted,"staged_bytes":used}


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("archive",type=Path)
    p.add_argument("manifest",type=Path)
    p.add_argument("output",type=Path)
    p.add_argument("--limit",type=int,choices=range(1,9),default=8)
    p.add_argument("--split",choices=("train","val","test"),default="train")
    p.add_argument("--max-archive-bytes",type=int,default=DISK_LIMIT)
    a=p.parse_args()
    print(json.dumps(stage(a.archive,a.manifest,a.output,a.limit,a.split,a.max_archive_bytes),indent=2))
