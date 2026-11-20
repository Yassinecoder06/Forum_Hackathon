import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.visualize_poses import render
from src.dataset import load
from src.model import Model
from src.retrieval import CaptionIndex, trajectory
from src.text_encoder import encode, french_features


def restore(path):
    state = torch.load(path,map_location="cpu",weights_only=True)
    cfg=state["config"]
    model=Model(len(state["vocab"])+2,cfg["width"],cfg["frames"],cfg.get("encoder","learned"),
                cfg.get("residual_scale",0.0))
    model.load_state_dict(state["model"])
    return model.eval(),state


def dtw(prediction,target,valid,support):
    size=len(target)
    distance=np.full((size+1,size+1),np.inf)
    length=np.zeros_like(distance)
    distance[0,0]=0
    for i in range(1,size+1):
        for j in range(1,size+1):
            mask=valid[j-1]&support[i-1]
            if not mask.any(): continue
            cost=np.linalg.norm(prediction[i-1,mask]-target[j-1,mask],axis=-1).mean()
            prev=min(((i-1,j),(i,j-1),(i-1,j-1)),key=lambda q:distance[q])
            distance[i,j]=distance[prev]+cost
            length[i,j]=length[prev]+1
    return distance[-1,-1]/length[-1,-1] if np.isfinite(distance[-1,-1]) else None


def score(pred, target, mask, support, seconds, observed_seconds):
    distance=np.linalg.norm(pred-target,axis=-1)
    output={}
    for name,region in {"joint":slice(None),"wrist": [4,7,18,39],"hand":slice(18,60)}.items():
        valid=mask[:,region]
        output[name]=float(distance[:,region][valid].mean()) if valid.any() else None
    for order,name in [(1,"velocity"),(2,"acceleration")]:
        valid=np.logical_and.reduce([mask[i:len(mask)-order+i] for i in range(order+1)])
        error=np.linalg.norm(np.diff(pred,n=order,axis=0)-np.diff(target,n=order,axis=0),axis=-1)
        output[name]=float(error[valid].mean()) if valid.any() else None
    output["dtw"]=dtw(pred,target,mask,support)
    output["duration_mae_seconds"]=abs(seconds-observed_seconds)
    output["frame_count_mae_at_24fps"]=abs(round(24*seconds)-round(24*observed_seconds))
    output["invalid_coordinate_fraction"]=float((~np.isfinite(pred)).mean())
    output["mean_displacement"]=float(np.linalg.norm(np.diff(pred,axis=0),axis=-1).mean())
    output["static"]=float(output["mean_displacement"]<.01)
    return output


def average(rows):
    result = {}
    for key in rows[0]:
        values = [row[key] for row in rows if row[key] is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


def mismatch_indices(samples):
    for i, (row, *_) in enumerate(samples):
        candidates = [j for j, (other, *_) in enumerate(samples)
                      if other["source_video_id"] != row["source_video_id"]
                      and other["french_caption"] != row["french_caption"]]
        if not candidates:
            raise ValueError("No mismatched caption from a different recording")
        rng = np.random.default_rng(17 + i)
        yield rng.choice(candidates, size=min(5, len(candidates)), replace=False)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(".."))
    p.add_argument("--manifest", default="data/manifests/signverse_fsl_caption_pose.jsonl")
    p.add_argument("--evaluation-manifest", type=Path, help="Additional test-only manifest relative to root")
    p.add_argument("--french-features", type=Path, help="Precomputed frozen-encoder features for all evaluation captions")
    p.add_argument("--checkpoint",type=Path,default=Path("checkpoints/best.pt"))
    p.add_argument("--support-manifest", type=Path, help="Use a shared landmark mask from a subset of training data")
    p.add_argument("--metrics-only", action="store_true", help="Skip rendering comparison videos")
    p.add_argument("--out",type=Path,default=Path("outputs"))
    a=p.parse_args()
    torch.set_num_threads(2)
    model,state=restore(a.checkpoint)
    manifest=a.root/a.manifest
    if hashlib.sha256(manifest.read_bytes()).hexdigest()!=state["manifest_sha256"]:
        raise ValueError("Evaluation manifest differs from training manifest")
    groups=load(manifest,state["config"]["frames"],a.root)
    if a.evaluation_manifest:
        external=load(a.root/a.evaluation_manifest,state["config"]["frames"],a.root,required=("test",))
        rows=external["test"]
        occupied={r["source_video_id"] for samples in groups.values() for r,*_ in samples}
        if external["train"] or external["val"] or any(r["source_video_id"] in occupied for r,*_ in rows):
            raise ValueError("Additional manifest must contain only previously unseen test recordings")
        groups["new_test"]=rows
    splits=[split for split in groups if split!="train"]
    train=groups["train"]
    masks=np.stack([t[2] for t in train])
    counts=masks.sum(axis=0)
    support=counts>0
    if a.support_manifest:
        reference=load(a.root/a.support_manifest,state["config"]["frames"],a.root,required=("train",))["train"]
        training_rows={row["id"]: row for row, *_ in train}
        if any(training_rows.get(row["id"]) != row for row, *_ in reference):
            raise ValueError("Shared support must come from unchanged training rows")
        support=np.stack([item[2] for item in reference]).any(axis=0)
    template=np.stack([t[1] for t in train]).sum(axis=0)/counts.clip(min=1)[...,None]
    duration=float(np.mean([t[3] for t in train]))
    index = CaptionIndex([row["french_caption"] for row, *_ in train])
    a.out.mkdir(parents=True,exist_ok=True)
    report={}
    frozen = state["config"].get("encoder") == "frozen_french"
    if frozen:
        rows=[r for split in splits for r,*_ in groups[split]]
        texts={r["id"]:r["french_caption"] for r in rows}
        if a.french_features:
            saved=torch.load(a.french_features,map_location="cpu",weights_only=True)
            features=saved["features"]
            if saved["texts"]!=texts or features.keys()!=texts.keys():
                raise ValueError("Cached feature captions differ from evaluation manifest")
            if any(t.shape!=(48,768) or not t.isfinite().all() for t in features.values()):
                raise ValueError("Invalid frozen-encoder features")
        else:
            features=dict(zip(texts,french_features(list(texts.values()),"cpu")))
    for split in splits:
        metrics=defaultdict(list)
        samples=groups[split]
        ids=torch.stack([features[row["id"]] for row,*_ in samples]) if frozen else torch.tensor(
            [encode(row["french_caption"],state["vocab"]) for row,*_ in samples])
        with torch.inference_mode():
            poses,seconds=model(ids)
            poses=poses.numpy();seconds=seconds.exp().clamp(.25,15).numpy()
        retrieval_preview = []
        by_recording=defaultdict(lambda: defaultdict(list))
        for i, ((row,target,mask,observed), controls) in enumerate(zip(samples,mismatch_indices(samples))):
            common=mask&support
            scores={
                "model":score(poses[i],target,common,support,float(seconds[i]),observed),
                "phase_template":score(template,target,common,support,duration,observed),
                "mismatched_caption":average([score(poses[j],target,common,support,float(seconds[j]),observed)
                                              for j in controls]),
            }
            for method, limit, prior in (("retrieval_top1", 1, 0), ("retrieval_shrunk_top3", 3, .5)):
                indices = index.neighbors(row["french_caption"], limit)
                motion, predicted_seconds = trajectory(train, indices, template, duration, prior)
                scores[method] = score(motion,target,common,support,predicted_seconds,observed)
                if i == 0 and method == "retrieval_shrunk_top3":
                    retrieval_preview.append(("Caption retrieval + template", motion, support))
            for method, values in scores.items():
                metrics[method].append(values)
                by_recording[row["source_video_id"]][method].append(values)
        diversity=float(np.mean(np.std(poses,axis=0)[support]))
        report[split]={"examples":len(samples),"recordings":len(by_recording),
                       "model_across_cue_coordinate_std":diversity,
                       "methods":{method:average(rows) for method,rows in metrics.items()},
                       "by_recording":[{"recording":f"{split}-{i:02d}",
                                        "examples":len(rows["model"]),
                                        "methods":{method:average(values) for method,values in rows.items()}}
                                       for i,rows in enumerate(by_recording.values(),1)]}
        if split in ("test","new_test") and not a.metrics_only:
            row,xy,valid,observed=samples[0]
            tracks=[("Observed • weak alignment",xy,valid&support),("Generated",poses[0],support),
                    ("Training phase template",template,support), *retrieval_preview]
            for suffix in ("gif","mp4"):
                render(a.out/f"{split}_comparison.{suffix}",tracks,state["config"]["frames"]/observed)
            report[split]["demo"]={"caption":row["french_caption"],"observed_seconds":observed,
                            "generated_seconds":float(seconds[0])}
    (a.out/"evaluation.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
