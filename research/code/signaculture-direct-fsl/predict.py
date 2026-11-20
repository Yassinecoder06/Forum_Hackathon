import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from evaluate import restore
from scripts.visualize_poses import render
from src.dataset import load, tokens
from src.text_encoder import encode, french_features


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--text",required=True)
    p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/best.pt"))
    p.add_argument("--root",type=Path,default=Path(".."))
    p.add_argument("--manifest", default="data/manifests/signverse_fsl_caption_pose.jsonl")
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    torch.set_num_threads(2)
    model,state=restore(a.checkpoint)
    vocab=state["vocab"]
    config=state["config"]
    manifest=a.root/a.manifest
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=state["manifest_sha256"]:
        raise ValueError("Prediction support manifest differs from checkpoint")
    training=load(manifest,config["frames"],a.root)["train"]
    support=np.stack([x[2] for x in training]).any(0)
    text=french_features([a.text],"cpu") if config.get("encoder")=="frozen_french" else torch.tensor([encode(a.text,vocab)])
    with torch.inference_mode():
        prediction,duration=model(text)
    xy=prediction[0].numpy()
    seconds=float(duration.exp().clamp(.25,15)[0])
    valid=support & np.isfinite(xy).all(-1)
    fps=len(xy)/seconds
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(a.output/"predicted_poses.npz",normalized_xy=xy,mask=valid,fps=fps)
    for suffix in ("gif","mp4"):
        render(a.output/f"predicted_skeleton.{suffix}",[("Experimental 2D prediction",xy,valid)],fps)
    unfamiliar=sorted(set(tokens(a.text))-set(vocab))
    metadata={"text":a.text,"sign_language":"fsl","duration_seconds":seconds,"unknown_tokens":unfamiliar,
              "checkpoint":str(a.checkpoint),"manifest_sha256":state["manifest_sha256"],
              "warning":"Weak subtitle supervision; signing accuracy and out-of-domain text are unverified."}
    (a.output/"prediction_metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
    quality={"frames":len(xy),"nonfinite_coordinates":int((~np.isfinite(xy)).sum()),
             "training_supported_fraction":float(valid.mean()),
             "mean_phase_displacement":float(np.linalg.norm(np.diff(xy,axis=0),axis=-1).mean())}
    (a.output/"quality_report.json").write_text(json.dumps(quality,indent=2)+"\n")


if __name__=="__main__":main()
