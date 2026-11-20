"""Private host-side CUDA worker shared by Docker Studio capture and composition."""
from copy import deepcopy
from functools import lru_cache
import gc
import os
from pathlib import Path
from threading import Lock
from uuid import UUID

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException

from .pose_models import WORK, normalized_groups, sapiens_predictor
from .hamer import load_predictor
from .transition_model import checkpoint, learned_motion

app = FastAPI()
gpu_lock = Lock()
bridge_lock = Lock()
torch.set_num_threads(2)
cv2.setNumThreads(1)
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
BODY_NAMES = ("left_shoulder", "right_shoulder", "right_elbow", "right_wrist", "left_elbow", "left_wrist")


@app.post("/bridge")
def bridge(motion: dict):
    with bridge_lock:
        return learned_motion(motion, bridge_model())


bridge_model = lru_cache(maxsize=1)(checkpoint)


def source_frames(source, by_time):
    capture = cv2.VideoCapture(str(source))
    fps = capture.get(cv2.CAP_PROP_FPS)
    try:
        if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Vidéo illisible pour Sapiens")
        samples = {round(time * fps / 1000): time for time in by_time}
        for index in range(max(samples) + 1):
            if not capture.grab():
                break
            if index in samples:
                ok, image = capture.retrieve()
                if ok:
                    yield samples[index], image
    finally:
        capture.release()


def sharpen(source, motion):
    output = deepcopy(motion)
    frames = output["frames"]
    samples = {frame["time_ms"]: frame for frame in frames if frame.get("playback_origin") != "endpoint_hold"}
    counts = {"sapiens_frames": 0, "left_hamer_frames": 0, "right_hamer_frames": 0}
    span_history = []
    poses = {}
    sapiens = sapiens_predictor(
        WORK / "models/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2", "cuda", WORK / "sapiens/lite/demo"
    )
    for time, image in source_frames(source, samples):
        height, width = image.shape[:2]
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            points, scores = sapiens(image)
        poses[time] = normalized_groups(points, scores.reshape(-1), width, height, "sapiens")
    if not poses:
        raise ValueError("Aucune image exploitable pour Sapiens.")
    # The two full models and checkpoint deserialization cannot coexist on
    # this host. Keep only the small landmark array between the passes.
    del sapiens
    gc.collect()
    torch.cuda.empty_cache()
    hamer, _ = load_predictor(WORK / "models/hamer", WORK / "hamer-source", low_memory=True)
    for time, image in source_frames(source, poses):
        frame = samples[time]
        height, width = image.shape[:2]
        pose = poses[time]
        left, right = pose["body"][:2]
        if left is None or right is None:
            continue
        aspect = width / height
        span = np.hypot((left[0] - right[0]) * aspect, left[1] - right[1])
        if span < .035:
            continue
        span_history.append(span)
        scale = .4 / float(np.median(span_history[-9:]))
        center = (np.asarray(left[:2]) + right[:2]) / 2

        def project(point, z):
            return [(point[0] - center[0]) * aspect * scale, 1.45 + (center[1] - point[1]) * scale, z]

        joints = frame["joints"]
        replaced = {name: project(point, .26 if name.endswith("wrist") else .08)
                    for name, point in zip(BODY_NAMES, pose["body"]) if point}
        for side in ("left", "right"):
            wrist = f"{side}_wrist"
            if wrist in replaced:
                delta = np.subtract(replaced[wrist], joints[wrist])
                for name in list(joints):
                    if name.startswith(f"{side}_") and name not in BODY_NAMES:
                        joints[name] = (np.asarray(joints[name]) + delta).tolist()
        joints.update(replaced)
        joints["neck"] = [0., 1.45, .08]
        if pose["face"][108]:
            joints["head"] = project(pose["face"][108], .08)
        counts["sapiens_frames"] += 1
        hints = {key: [{"x": point[0], "y": point[1]} for point in pose[side] if point]
                 for side, key in (("left", "LLM"), ("right", "RLM"))}
        hands = hamer(image, hints)
        for side in ("left", "right"):
            if side not in hands:
                continue
            wrist = f"{side}_wrist"
            joints[wrist] = project(hands[side][0], .26)
            for finger, name in enumerate(FINGERS):
                for offset in range(1, 5):
                    k = 4 * finger + offset
                    joints[f"{side}_{name}_{offset}"] = project(hands[side][k], .26 + hands[f"{side}_depth"][k])
            frame["hand_tracking"][side] = {"observed": True, "source": "hamer-sapiens"}
            counts[f"{side}_hamer_frames"] += 1
    del hamer
    torch.cuda.empty_cache()
    if not counts["sapiens_frames"]:
        raise ValueError("Sapiens n’a détecté aucune pose exploitable.")
    observed = [frame for frame in frames if frame.get("playback_origin") != "endpoint_hold"]
    for frame, edge in ((frames[0], observed[0]), (frames[-1], observed[-1])):
        if frame.get("playback_origin") == "endpoint_hold":
            frame.update(joints=deepcopy(edge["joints"]), hand_tracking=deepcopy(edge["hand_tracking"]))
    output["quality"].update(counts)
    sampled = output["quality"]["sampled_frames"]
    for side in ("left", "right"):
        seen = sum(frame["hand_tracking"][side]["observed"] for frame in observed)
        output["quality"][f"{side}_hand_coverage"] = round(seen / sampled, 3)
    output["tracked_joints"] = sorted(set().union(*(f["joints"] for f in frames)))
    output["capture_method"] = "sapiens-0.3b-hamer-with-mediapipe-fallback"
    output["warnings"].append("Sapiens + HaMeR pour la pose. MediaPipe conserve le visage et les points indisponibles.")
    return output


@app.post("/capture/{job_id}")
def capture(job_id: UUID, motion: dict):
    source = Path(os.environ["SIGNACULTURE_HOST_CAPTURE_STORAGE"]) / "captures" / str(job_id) / "source.mp4"
    if not source.is_file():
        raise HTTPException(404, "Capture absente du volume partagé")
    with gpu_lock:
        return sharpen(source, motion)
