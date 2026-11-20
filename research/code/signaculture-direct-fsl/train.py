import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import yaml

from src.dataset import load, tokens
from src.losses import objective
from src.model import Model
from src.text_encoder import encode, french_features


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--root",type=Path,default=Path(".."))
    p.add_argument("--manifest",default="data/manifests/signverse_fsl_caption_pose.jsonl")
    p.add_argument("--config",default="configs/poc.yaml")
    p.add_argument("--out",type=Path,default=Path("checkpoints"))
    p.add_argument("--french-features", type=Path, help="Cached frozen encoder outputs keyed by cue ID")
    p.add_argument("--resume",action="store_true")
    args=p.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text())
    torch.manual_seed(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this pilot")
    device="cuda"
    print(torch.cuda.get_device_name(),torch.cuda.mem_get_info(),flush=True)
    manifest=args.root/args.manifest
    digest=hashlib.sha256(manifest.read_bytes()).hexdigest()
    groups=load(manifest,cfg["frames"],args.root)
    words=sorted({w for row,*_ in groups["train"] for w in tokens(row["french_caption"])})
    vocab={word:i+2 for i,word in enumerate(words)}
    mode=cfg.get("encoder","learned")
    residual_scale = cfg.get("residual_scale", 0.0)
    model=Model(len(vocab)+2,cfg["width"],cfg["frames"],mode,residual_scale).to(device)
    if residual_scale:
        totals = np.stack([item[1] for item in groups["train"]]).sum(axis=0)
        counts = np.stack([item[2] for item in groups["train"]]).sum(axis=0)
        model.phase_template.copy_(torch.from_numpy(totals / counts.clip(min=1)[..., None]))
    print("parameters",sum(p.numel() for p in model.parameters()), "splits",{k:len(v) for k,v in groups.items()},flush=True)

    features={}
    if mode=="frozen_french":
        rows=[record[0] for split in ("train","val") for record in groups[split]]
        texts = {row["id"]: row["french_caption"] for row in rows}
        if args.french_features:
            cached = torch.load(args.french_features, map_location="cpu", weights_only=True)
            if any(cached["texts"].get(cue) != text for cue, text in texts.items()):
                raise ValueError("Cached captions differ from training or validation data")
            features = {cue: cached["features"][cue] for cue in texts}
            if any(item.shape != (48, 768) or not item.isfinite().all() for item in features.values()):
                raise ValueError("Invalid cached encoder features")
        else:
            features = dict(zip(texts, french_features(list(texts.values()), device)))

    def batch(samples):
        rows, xy, mask, seconds=zip(*samples)
        text=torch.stack([features[r["id"]] for r in rows]).to(device) if features else torch.tensor(
            [encode(r["french_caption"],vocab) for r in rows],device=device)
        return (text,
                torch.as_tensor(np.stack(xy),device=device),torch.as_tensor(np.stack(mask),device=device),
                torch.tensor(seconds,dtype=torch.float32,device=device))

    def loss(data):
        ids,xy,mask,sec=data
        with torch.autocast("cuda",enabled=torch.cuda.is_bf16_supported(),dtype=torch.bfloat16):
            pose,duration=model(ids)
        return objective(pose.float(),duration.float(),xy,mask,sec)

    train=batch(groups["train"])
    val=batch(groups["val"])
    smoke=tuple(t[:2] for t in train)
    smoke_text=torch.randn_like(smoke[0]) if features else torch.randint(0,len(vocab)+2,smoke[0].shape,device=device)
    smoke=(smoke_text,torch.randn_like(smoke[1]),
           torch.ones_like(smoke[2]),torch.ones_like(smoke[3]))
    for _ in range(10):
        model.zero_grad(set_to_none=True)
        value=loss(smoke)
        assert torch.isfinite(value)
        value.backward()
    probe=tuple(t[:2] for t in train)
    diagnostic=Model(len(vocab)+2,cfg["width"],cfg["frames"],mode).to(device)
    original,model=model,diagnostic
    diagnostic_opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    model.eval()
    with torch.no_grad(): before=float(loss(probe))
    model.train()
    for _ in range(35):
        diagnostic_opt.zero_grad(set_to_none=True)
        loss(probe).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        diagnostic_opt.step()
    model.eval()
    with torch.no_grad(): after=float(loss(probe))
    print("real two-cue overfit", before,after,flush=True)
    model=original
    opt=torch.optim.AdamW(model.parameters(),lr=cfg["learning_rate"])
    args.out.mkdir(parents=True,exist_ok=True)
    last=args.out/"last.pt"
    best=float("inf")
    start=0
    if args.resume:
        state=torch.load(last,map_location=device,weights_only=True)
        if state["manifest_sha256"]!=digest or state["config"]!=cfg:
            raise ValueError("Resume configuration or dataset mismatch")
        model.load_state_dict(state["model"]); opt.load_state_dict(state["opt"])
        best,start=state["best"],state["step"]
        torch.set_rng_state(state["rng"].cpu())
        torch.cuda.set_rng_state(state["cuda_rng"].cpu())
    for i in range(start,cfg["steps"]):
        model.train()
        indices=torch.randint(len(train[0]),(cfg["batch"],),device=device)
        opt.zero_grad(set_to_none=True)
        value=loss(tuple(t[indices] for t in train))
        if not torch.isfinite(value): raise RuntimeError("Non-finite loss")
        value.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        opt.step()
        if (i+1)%10 and i+1!=cfg["steps"]: continue
        model.eval()
        with torch.no_grad(): valid=float(loss(val))
        print(json.dumps({"step":i+1,"loss":float(value.detach()),"validation":valid,
                          "gpu_peak_bytes":torch.cuda.max_memory_allocated()}),flush=True)
        improved=valid<best
        best=min(valid,best)
        state={"model":model.state_dict(),"opt":opt.state_dict(),"vocab":vocab,"config":cfg,
               "step":i+1,"best":best,"manifest_sha256":digest,"rng":torch.get_rng_state(),
               "cuda_rng":torch.cuda.get_rng_state()}
        torch.save(state,last)
        if improved: torch.save(state,args.out/"best.pt")


if __name__=="__main__":
    main()
