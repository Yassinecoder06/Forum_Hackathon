"""Extract 2D pose previews for the 100 local Dicta-Sign review clips."""

import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT / ".runtime" / "pose-comparison"
INVENTORY = ROOT / "data" / "source_inventory" / "dictasign_lsf.json"

COCO_BODY = [5, 6, 8, 10, 7, 9, 11, 12, 13, 15, 14, 16]
GOLIATH_BODY = [5, 6, 8, 41, 7, 62, 9, 10, 11, 13, 12, 14]


def finger_order(wrist, first_tip):
    return [wrist] + [
        first_tip + finger * 4 + joint for finger in range(5) for joint in (3, 2, 1, 0)
    ]


def normalized_groups(points, scores, width, height, model):
    if model != "sapiens":
        groups = {
            "body": COCO_BODY,
            "left": range(91, 112),
            "right": range(112, 133),
            "face": range(23, 91),
        }
    else:
        groups = {
            "body": GOLIATH_BODY,
            "left": finger_order(62, 42),
            "right": finger_order(41, 21),
            "face": range(70, 308),
        }
    return {
        name: [
            [
                round(float(points[i][0] / width), 5),
                round(float(points[i][1] / height), 5),
                round(float(scores[i]), 4),
            ]
            if np.isfinite(points[i]).all() and np.isfinite(scores[i]) and scores[i] >= 0.3
            else None
            for i in indices
        ]
        for name, indices in groups.items()
    }


def mmpose_predictor(checkpoint, device):
    from rtmlib import RTMPose

    session = RTMPose(
        str(checkpoint), model_input_size=(192, 256), backend="onnxruntime", device=device
    )

    def predict(image):
        # The checkpoint's deployment pipeline normalizes RGB input.
        points, scores = session(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return points[0], scores[0]

    return predict


def sapiens_predictor(checkpoint, device, upstream):
    import torch

    sys.path.insert(0, str(upstream))
    from pose_utils import get_udp_warp_matrix, udp_decode

    device = torch.device(device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    model = torch.jit.load(str(checkpoint), map_location=device).eval()
    input_width, input_height = 768, 1024
    mean = torch.tensor([123.5, 116.5, 103.5], device=device)[:, None, None]
    std = torch.tensor([58.5, 57.0, 57.5], device=device)[:, None, None]

    @torch.inference_mode()
    def predict(image):
        height, width = image.shape[:2]
        center = np.array([width / 2, height / 2], dtype=np.float32)
        scale = np.array([width, height], dtype=np.float32) * 1.25
        aspect = input_width / input_height
        scale[0] = max(scale[0], scale[1] * aspect)
        scale[1] = scale[0] / aspect
        warp = get_udp_warp_matrix(center, scale, 0, (input_width, input_height))
        crop = cv2.warpAffine(image, warp, (input_width, input_height))
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        normalized = (torch.from_numpy(rgb).to(device).permute(2, 0, 1) - mean) / std
        heatmaps = model(normalized.unsqueeze(0))[0].float().cpu().numpy()
        if heatmaps.shape[0] != 308:
            raise ValueError(f"Expected 308 Goliath heatmaps, got {heatmaps.shape}")
        heatmap_height, heatmap_width = heatmaps.shape[1:]
        points, scores = udp_decode(
            heatmaps, (input_width - 1, input_height - 1), (heatmap_width, heatmap_height)
        )
        inverse = cv2.invertAffineTransform(warp)
        return cv2.transform(points.astype(np.float32), inverse)[0], scores[0]

    return predict


def extract(source, predict, model, stride):
    video = cv2.VideoCapture(str(source))
    frames = []
    index = 0
    try:
        fps = video.get(cv2.CAP_PROP_FPS)
        if not video.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Cannot read source FPS")
        while True:
            success, image = video.read()
            if not success:
                break
            if index % stride == 0:
                points, scores = predict(image)
                expected = 308 if model == "sapiens" else 133
                if points.shape != (expected, 2) or scores.size != expected:
                    raise ValueError(f"Unexpected {model} output shapes: {points.shape}, {scores.shape}")
                height, width = image.shape[:2]
                frames.append({
                    "time_ms": round(index * 1000 / fps),
                    **normalized_groups(points, scores.reshape(-1), width, height, model),
                })
            index += 1
    finally:
        video.release()
    if not frames:
        raise ValueError("No source frames decoded")
    return {"source_fps": fps, "sample_interval_ms": stride * 1000 / fps, "frames": frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("mmpose", "sapiens", "sdpose"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--label", required=True, help="Displayed checkpoint name")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--precision", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--limit", type=int, choices=range(1, 101), default=100)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--sapiens-demo", type=Path, default=WORK / "sapiens" / "lite" / "demo")
    args = parser.parse_args()
    checkpoint_exists = args.checkpoint.is_dir() if args.model == "sdpose" else args.checkpoint.is_file()
    if args.stride < 1 or not checkpoint_exists:
        parser.error("A local checkpoint and positive stride are required")
    if args.precision == "bf16" and (args.model == "mmpose" or args.device != "cuda"):
        parser.error("BF16 is supported for Sapiens and SDPose on CUDA")
    if args.model == "sdpose":
        bundle = hashlib.sha256()
        for name in sorted(args.checkpoint.rglob("*")):
            if not name.is_file() or name.suffix not in (".safetensors", ".json", ".txt"):
                continue
            with name.open("rb") as file:
                bundle.update(name.relative_to(args.checkpoint).as_posix().encode() + b"\0")
                bundle.update(hashlib.file_digest(file, "sha256").digest())
        sha256 = bundle.hexdigest()
    else:
        with args.checkpoint.open("rb") as file:
            sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    cv2.setNumThreads(1)
    if args.model == "mmpose":
        predictor = mmpose_predictor(args.checkpoint, args.device)
    elif args.model == "sapiens":
        predictor = sapiens_predictor(args.checkpoint, args.device, args.sapiens_demo)
    else:
        from .sdpose import predictor as sdpose_predictor

        sys.path.insert(0, str(args.sapiens_demo))
        from pose_utils import udp_decode

        predictor = sdpose_predictor(args.checkpoint, args.device, args.precision, udp_decode)

    precision_context = nullcontext()
    if args.model == "sapiens" and args.precision == "bf16":
        import torch

        if not torch.cuda.is_bf16_supported(including_emulation=False):
            parser.error("This GPU does not have native BF16 support")
        torch._C._jit_set_autocast_mode(False)
        precision_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16)

    with precision_context:
        return extract_inventory(args, predictor, sha256)


def extract_inventory(args, predictor, sha256):
    output = WORK / "results" / args.model
    output.mkdir(parents=True, exist_ok=True)
    entries = json.loads(INVENTORY.read_text())["entries"]
    entries = [entry for entry in entries if entry["language"] == "fsl"][:args.limit]
    failed = 0
    for index, entry in enumerate(entries, start=1):
        capture_id = str(uuid5(NAMESPACE_URL, entry["video_url"]))
        source = ROOT / ".runtime" / "captures" / capture_id / "source.mp4"
        path = output / f"{capture_id}.json"
        if path.is_file():
            old = json.loads(path.read_text())
            if (old.get("frames") and old.get("sha256") == sha256
                    and old.get("stride") == args.stride
                    and old.get("precision", "fp32") == args.precision):
                continue
        started = perf_counter()
        record = {
            "model": args.model, "checkpoint": args.label, "device": args.device,
            "sha256": sha256, "stride": args.stride, "precision": args.precision,
            "roi": "whole source frame",
        }
        try:
            record.update(extract(source, predictor, args.model, args.stride))
        except Exception as error:
            failed += 1
            record.update(frames=[], error=f"Extraction failed: {error}")
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, separators=(",", ":"), allow_nan=False))
        temporary.replace(path)
        print(f"{index}/{len(entries)} {entry['concept_fr']}: {len(record['frames'])} frames "
              f"in {perf_counter() - started:.2f}s {record.get('error', '')}", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
