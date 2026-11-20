"""Validate and resample existing SignaCulture SignVerse cue data for the pilot."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dataset import load


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(".."))
    p.add_argument("--frames",type=int,default=32)
    a=p.parse_args()
    groups=load(a.root/"data/manifests/signverse_fsl_caption_pose.jsonl",a.frames,a.root)
    print(json.dumps({k:{"cues":len(v),"recordings":len({r["source_video_id"] for r,*_ in v}),
                         "mean_valid_fraction":float(sum(t[2].mean() for t in v)/len(v))}
                      for k,v in groups.items()},indent=2))
