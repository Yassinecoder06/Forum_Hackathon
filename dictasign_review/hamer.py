"""Run HaMeR on hand crops guided by timestamp-matched MediaPipe detections."""

import argparse
from contextlib import nullcontext
import gzip
import hashlib
import json
from pathlib import Path
import sys
import subprocess
from time import perf_counter
from uuid import NAMESPACE_URL, uuid5

import cv2
import numpy as np

from .extract_pose import INVENTORY, ROOT, WORK

CHECKPOINT_SHA = "e5cc06f294d88a92dee24e603480aab04de532b49f0e08200804ee7d90e16f53"
MANO_SHA = "45d60aa3b27ef9107a7afd4e00808f307fd91111e1cfa35afd5c4a62de264767"


def load_predictor(weights, source, *, low_memory=False):
    import torch

    for filename, expected in [("hamer.ckpt", CHECKPOINT_SHA), ("MANO_RIGHT.pkl", MANO_SHA)]:
        with (weights / filename).open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != expected:
                raise ValueError(f"Unexpected checksum for {filename}")
    sys.path.insert(0, str(source))
    from hamer.configs import get_config
    from hamer.models import HAMER
    from hamer.utils.renderer import cam_crop_to_full

    cfg = get_config(str(weights / "model_config.yaml"))
    cfg.defrost()
    cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS", None)
    cfg.MANO.MODEL_PATH = str(weights)
    cfg.MANO.DATA_DIR = str(weights)
    cfg.MANO.MEAN_PARAMS = str(weights / "mano_mean_params.npz")
    cfg.freeze()
    # This is the pinned official checkpoint, verified above.
    if low_memory:
        model = HAMER(cfg, init_renderer=False)
        state = torch.load(weights / "hamer.ckpt", mmap=True, weights_only=False, map_location="cpu")
        model.load_state_dict(state["state_dict"], assign=True, strict=True)
        del state
    else:
        model = HAMER.load_from_checkpoint(
            str(weights / "hamer.ckpt"), cfg=cfg, init_renderer=False, weights_only=False, map_location="cpu"
        )
    model = model.to("cuda").eval()
    torch.set_float32_matmul_precision("high")
    mean = np.asarray(cfg.MODEL.IMAGE_MEAN, dtype=np.float32)
    std = np.asarray(cfg.MODEL.IMAGE_STD, dtype=np.float32)

    @torch.inference_mode()
    def predict(image, frame):
        h, w = image.shape[:2]
        sides, patches, centers, sizes = [], [], [], []
        for side, key in [("left", "LLM"), ("right", "RLM")]:
            points = np.array([
                (p["x"] * w, p["y"] * h) for p in frame.get(key) or []
                if p and np.isfinite([p["x"], p["y"]]).all()
            ])
            if len(points) < 10:
                continue
            low, high = points.min(axis=0), points.max(axis=0)
            center = (low + high) / 2
            extent = high - low
            if extent.max() < 3:
                continue
            size = 2.5 * max(extent[0] / .75, extent[1])
            factor = 256 / size
            affine = np.array([[factor, 0, 128 - center[0] * factor],
                               [0, factor, 128 - center[1] * factor]], dtype=np.float32)
            patch = cv2.warpAffine(image, affine, (256, 256), flags=cv2.INTER_LINEAR)
            if side == "left":
                patch = patch[:, ::-1]
            patches.append(((patch[:, :, ::-1] / 255.0 - mean) / std).transpose(2, 0, 1))
            sides.append(side)
            centers.append(center)
            sizes.append(size)
        if not patches:
            return {}
        batch = torch.tensor(np.stack(patches), device="cuda", dtype=torch.float32)
        output = model({"img": batch})
        camera = output["pred_cam"].clone()
        joints = output["pred_keypoints_3d"].clone()
        vertices = output["pred_vertices"].clone()
        mirror = torch.tensor([1 if s == "right" else -1 for s in sides], device="cuda")
        camera[:, 1] *= mirror
        joints[:, :, 0] *= mirror[:, None]
        vertices[:, :, 0] *= mirror[:, None]
        to_tensor = lambda values: torch.as_tensor(np.asarray(values), dtype=torch.float32, device="cuda")
        focal = cfg.EXTRA.FOCAL_LENGTH / 256 * max(w, h)
        translation = cam_crop_to_full(camera, to_tensor(centers), to_tensor(sizes),
                                       to_tensor([[w, h]] * len(sides)), focal)
        camera_joints = joints + translation[:, None, :]
        pixels = camera_joints[:, :, :2] / camera_joints[:, :, 2:] * focal + to_tensor([w / 2, h / 2])
        normalized = (pixels / to_tensor([w, h])).cpu().numpy()
        depth = (joints[:, :1, 2] - joints[:, :, 2]).cpu().numpy()
        camera_vertices = vertices + translation[:, None, :]
        xy = (camera_vertices[:, :, :2] / camera_vertices[:, :, 2:] * focal
              + to_tensor([w / 2, h / 2])) / to_tensor([w, h])
        mesh = torch.cat([xy, camera_vertices[:, :, 2:]], dim=-1).cpu().numpy()
        result = {}
        for i, side in enumerate(sides):
            if normalized[i].shape != (21, 2) or not np.isfinite(normalized[i]).all() or not np.isfinite(depth[i]).all():
                continue
            result[side] = normalized[i].round(5).tolist()
            result[f"{side}_depth"] = depth[i].round(5).tolist()
            if np.isfinite(mesh[i]).all() and (mesh[i, :, 2] > 0).all():
                result[f"{side}_mesh"] = mesh[i].astype(float).round(5).tolist()
        return result

    return predict, model.mano.faces.tolist()


def estimate_head_rotation(face, width, height, sapiens=False):
    # Nose, chin, image-left and image-right eye corners, mouth corners.
    indices = (108, 7, 51, 27, 118, 119) if sapiens else (1, 152, 33, 263, 61, 291)
    if not face or any(index >= len(face) or not face[index] for index in indices):
        return None
    pixels = np.array([face[index][:2] for index in indices], dtype=np.float64) * [width, height]
    eye_span = np.linalg.norm(pixels[2] - pixels[3])
    if not np.isfinite(pixels).all() or eye_span < 8:
        return None
    template = np.array([(0, 0, -22), (0, 63, 4), (-43, -32, 0), (43, -32, 0),
                         (-28, 28, -2), (28, 28, -2)], dtype=np.float64)
    focal = max(width, height)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, translation = cv2.solvePnP(template, pixels, camera, None, flags=cv2.SOLVEPNP_SQPNP)
    if not ok or translation[2, 0] <= 0:
        return None
    projected, _ = cv2.projectPoints(template, rotation, translation, camera, None)
    error = np.linalg.norm(projected[:, 0] - pixels, axis=1).mean()
    matrix, _ = cv2.Rodrigues(rotation)
    if error > .14 * eye_span or matrix[2, 2] < .35:
        return None
    # Camera coordinates are X right, Y down, Z away from the camera.
    basis = np.diag([1, -1, -1])
    return (basis @ matrix @ basis).round(6).ravel().tolist()


def holistic_frame(mp, detector, image, time_ms, fallback):
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), time_ms)
    hints = {
        key: [{"x": point.x, "y": point.y} for point in landmarks]
        for key, landmarks in (("LLM", result.left_hand_landmarks), ("RLM", result.right_hand_landmarks))
    }
    body_indices = (11, 12, 14, 16, 13, 15, 23, 24, 25, 27, 26, 28)
    body = [
        [point.x, point.y] if (point := result.pose_landmarks[index]).visibility >= .5 else None
        for index in body_indices
    ] if result.pose_landmarks else []
    recovered = []
    if body and body[0] and body[1]:
        shoulder_span = np.linalg.norm(np.subtract(body[0], body[1]))
        for side, key, wrist_index in (("left", "LLM", 5), ("right", "RLM", 3)):
            points = fallback.get(key) or []
            if hints[key] or len(points) != 21 or not points[0] or not body[wrist_index]:
                continue
            wrist = [points[0]["x"], points[0]["y"]]
            if np.linalg.norm(np.subtract(wrist, body[wrist_index])) <= .35 * shoulder_span:
                hints[key] = points
                recovered.append(side)
    return hints, {
        "body": body,
        "face": [[p.x, p.y] for p in result.face_landmarks],
        "face_blendshapes": {
            category.category_name: round(category.score, 5)
            for category in (result.face_blendshapes or [])
            if category.category_name and np.isfinite(category.score)
        },
        "fallback_hands": recovered,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, choices=range(1, 101), default=100)
    crops = parser.add_mutually_exclusive_group()
    crops.add_argument("--sapiens", action="store_true", help="Use cached Sapiens 0.3B landmarks for body and hand crops")
    crops.add_argument("--holistic-model", type=Path, help="Run installed MediaPipe Holistic on source frames and write a separate preview")
    args = parser.parse_args()
    cv2.setNumThreads(1)
    predict, mesh_faces = load_predictor(WORK / "models" / "hamer", WORK / "hamer-source")
    result_name = "hamer-sapiens" if args.sapiens else "hamer-latest" if args.holistic_model else "hamer"
    destination = WORK / "results" / result_name
    checkpoint_label = "HaMeR · MediaPipe hand crops"
    versions = {}
    if args.holistic_model:
        import mediapipe as mp

        options = mp.tasks.vision.HolisticLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(args.holistic_model)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            output_face_blendshapes=True,
        )
        revision = subprocess.check_output(["git", "-C", str(WORK / "hamer-source"), "rev-parse", "HEAD"], text=True).strip()
        versions = {"mediapipe_version": mp.__version__, "hamer_revision": revision, "preview_format": 2}
        checkpoint_label = f"MediaPipe {mp.__version__} + HaMeR {revision[:7]}"
    destination.mkdir(parents=True, exist_ok=True)
    entries = json.loads(INVENTORY.read_text())["entries"]
    failed = 0
    for i, entry in enumerate([e for e in entries if e["language"] == "fsl"][:args.limit], start=1):
        identifier = str(uuid5(NAMESPACE_URL, entry["video_url"]))
        path = destination / f"{identifier}.json"
        baseline_model = "sapiens" if args.sapiens else "mmpose"
        baseline = json.loads((WORK / "results" / baseline_model / path.name).read_text())
        if args.sapiens:
            versions = {"sapiens_sha256": baseline["sha256"], "sapiens_precision": baseline["precision"]}
            checkpoint_label = "Sapiens 0.3B + HaMeR"
        if path.is_file():
            cached = json.loads(path.read_text())
            if cached.get("mesh_faces") and cached.get("frames") and all(cached.get(k) == v for k, v in versions.items()):
                continue
        folder = ROOT / ".runtime" / "captures" / identifier
        fps = baseline["source_fps"]
        sample_times = {round(frame["time_ms"] * fps / 1000): frame["time_ms"] for frame in baseline["frames"]}
        if args.sapiens:
            sapiens_frames = {frame["time_ms"]: frame for frame in baseline["frames"]}
        else:
            with gzip.open(folder / "animics-landmarks.json.gz", "rt") as source:
                hints = {round(frame["time_ms"] * fps / 1000): frame for frame in json.load(source)}
        video = cv2.VideoCapture(str(folder / "source.mp4"))
        frames = []
        started = perf_counter()
        try:
            with (mp.tasks.vision.HolisticLandmarker.create_from_options(options)
                  if args.holistic_model else nullcontext()) as detector:
                for index in range(max(sample_times) + 1):
                    ok, image = video.read()
                    if not ok:
                        break
                    if index not in sample_times:
                        continue
                    time_ms = sample_times[index]
                    frame = {"time_ms": time_ms}
                    if args.sapiens:
                        pose = sapiens_frames[time_ms]
                        frame.update(body=pose.get("body"), face=pose.get("face"))
                        crop_hints = {
                            key: [{"x": point[0], "y": point[1]} for point in pose.get(side) or [] if point]
                            for key, side in (("LLM", "left"), ("RLM", "right"))
                        }
                    elif detector is not None:
                        crop_hints, holistic = holistic_frame(mp, detector, image, time_ms, hints.get(index, {}))
                        frame.update(holistic)
                    else:
                        crop_hints = hints.get(index, {})
                    hands = predict(image, crop_hints)
                    # HaMeR can reject a hand while the body model still has a
                    # usable 21-point hand. Keep that hand instead of leaving a
                    # hole that the avatar fills with its bind pose. Never
                    # replace a hand HaMeR did return: a wrist offset is not
                    # evidence that the tracked hand is wrong.
                    tracked = {side for side in ("left", "right") if hands.get(side)}
                    restored = [side for side in frame.get("fallback_hands", []) if side not in tracked]
                    for side, key in (("left", "LLM"), ("right", "RLM")):
                        if side in tracked:
                            continue
                        points = crop_hints.get(key)
                        if not isinstance(points, list) or len(points) != 21:
                            continue
                        try:
                            hand = [[float(point["x"]), float(point["y"])] for point in points]
                        except (KeyError, TypeError, ValueError):
                            continue
                        if not np.isfinite(hand).all():
                            continue
                        hands[side] = hand
                        hands[f"{side}_depth"] = [0.0] * 21
                        if side not in restored:
                            restored.append(side)
                    if restored:
                        hands["fallback_hands"] = restored
                    else:
                        frame.pop("fallback_hands", None)
                    frame.update(hands)
                    if frame.get("face"):
                        height, width = image.shape[:2]
                        frame["head_rotation"] = estimate_head_rotation(frame["face"], width, height, args.sapiens)
                    frames.append(frame)
        finally:
            video.release()
        if len(frames) != len(sample_times):
            failed += 1
            print(f"{i}: incomplete source decoding", flush=True)
            continue
        result = {
            "model": "hamer", "checkpoint": checkpoint_label, "device": "cuda",
            "sha256": CHECKPOINT_SHA, "mano_sha256": MANO_SHA,
            "source_fps": fps, "sample_interval_ms": baseline["sample_interval_ms"],
            "mesh_faces": mesh_faces, "frames": frames, **versions,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False))
        temporary.replace(path)
        print(f"{i}/{args.limit} {entry['concept_fr']}: {len(frames)} frames in {perf_counter()-started:.2f}s", flush=True)
    return int(failed > 0)


if __name__ == "__main__":
    raise SystemExit(main())
