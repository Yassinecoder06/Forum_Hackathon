"""Whole-body pose predictors (MMPose DWPose, Sapiens) and landmark normalization."""

from pathlib import Path
import sys

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
