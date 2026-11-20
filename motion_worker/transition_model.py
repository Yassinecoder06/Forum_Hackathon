"""LSFB exp15 learned transition model shared by the Studio motion worker and previews."""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / ".models/lsfb-transition/transition_model_exp15.pt"
POINTS = [(i, name) for i, name in enumerate((
    "head", None, None, None, None, None, None, None, None, None, None,
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist",
)) if name]
for side, start in (("left", 23), ("right", 44)):
    POINTS.append((start, f"{side}_wrist"))
    POINTS.extend((start + finger_index * 4 + joint, f"{side}_{finger}_{joint}")
                  for finger_index, finger in enumerate(("thumb", "index", "middle", "ring", "pinky"))
                  for joint in range(1, 5))


def context(clip, last):
    frames = clip["frames"]
    t = np.array([f["time_ms"] for f in frames])
    time = (t[-1] - np.arange(5, -1, -1) * 40 if last else t[0] + np.arange(6) * 40)
    data = np.zeros((6, 157, 3), dtype=np.float32)
    for index, name in POINTS:
        samples = [(stamp, frame["joints"][name]) for stamp, frame in zip(t, frames) if name in frame["joints"]]
        if not samples:
            wrist = f"{name.split('_')[0]}_wrist"
            samples = [(stamp, frame["joints"][wrist]) for stamp, frame in zip(t, frames)
                       if wrist in frame["joints"]]
        if samples:
            stamps, vec = zip(*samples)
            data[:, index] = np.stack([np.interp(time, stamps, axis) for axis in np.asarray(vec).T], axis=-1)
    return data


def checkpoint():
    weights = torch.load(MODEL, weights_only=True, map_location="cpu")
    width = weights["cfg"]["hidden"]
    layers = []
    previous = 5653
    for _ in range(weights["cfg"]["layers"]):
        layers.extend((nn.Linear(previous, width), nn.GELU(), nn.Dropout(weights["cfg"]["dropout"])))
        previous = width
    layers.append(nn.Linear(previous, 5652))
    net = nn.Sequential(*layers)
    net.load_state_dict(weights["state"], strict=True)
    return net.eval()


@torch.inference_mode()
def predict(net, a, b, duration):
    x = np.concatenate((context(a, True), context(b, False)))
    shoulders = x[:, 11:13]
    center = np.median(shoulders.mean(axis=1), axis=0)
    scale = np.median(np.linalg.norm(shoulders[:, 0, :2] - shoulders[:, 1, :2], axis=1))
    if scale < .03:
        raise ValueError("Insufficient shoulder span")
    mapped = list({index for index, _ in POINTS})
    # LSFB expects Y up and raw MediaPipe Z. Native rig depth points toward camera.
    x[:, mapped] = (x[:, mapped] - center) / scale * np.array([1, 1, -1])
    gap = duration / 40
    condition = torch.from_numpy(np.append(x.ravel(), np.float32(gap / 20)))
    residual = net(condition).reshape(12, 157, 3).numpy()
    time = np.linspace(0, 1, 14)[1:-1, None, None]
    blend = time * time * (3 - 2 * time)
    pred = x[5] * (1 - blend) + x[6] * blend + residual
    prediction = pred * np.array([1, 1, -1]) * scale + center
    if not np.isfinite(prediction).all():
        raise ValueError("The checkpoint produced a nonfinite bridge")
    return prediction


def learned_motion(motion, net):
    """Keep the Studio samples and replace only the generated bridge joints."""
    from itertools import groupby

    result = deepcopy(motion)
    groups = [list(frames) for _, frames in groupby(
        result["frames"], key=lambda frame: frame.get("playback_origin") == "transition"
    )]
    joint_index = {name: index for index, name in POINTS}
    for idx in range(1, len(groups) - 1):
        bridge = groups[idx]
        if bridge[0].get("playback_origin") != "transition":
            continue
        before, after = groups[idx - 1], groups[idx + 1]
        start, end = before[-1]["time_ms"], after[0]["time_ms"]
        if not 80 <= end - start <= 800:
            continue
        pred = predict(net, {"frames": before}, {"frames": after}, end - start)
        left, right = before[-1]["joints"], after[0]["joints"]
        for frame in bridge:
            progress = (frame["time_ms"] - start) / (end - start)
            for name in frame["joints"].keys() & joint_index.keys() & left.keys() & right.keys():
                track = np.vstack((left[name], pred[:, joint_index[name]], right[name]))
                frame["joints"][name] = [
                    float(np.interp(progress, np.linspace(0, 1, 14), axis)) for axis in track.T
                ]
    return result
