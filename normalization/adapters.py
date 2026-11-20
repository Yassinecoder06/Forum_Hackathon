"""Adapters for existing native motion and explicitly aligned MediaPipe world landmarks."""
import json
from pathlib import Path

import torch
from torch.utils.data import Dataset

from .config import Layout

FINGERS = ("thumb", "index", "middle", "ring", "pinky")
SIDES = ("left", "right")
NATIVE_BODY = ("hips", "neck", "head", "left_shoulder", "left_elbow", "left_wrist",
               "right_shoulder", "right_elbow", "right_wrist")
NATIVE_NAMES = NATIVE_BODY + tuple(
    f"{side}_{finger}_{joint}" for side in SIDES for finger in FINGERS for joint in range(1, 5)
)
INDEX = {name: i for i, name in enumerate(NATIVE_NAMES)}
FINGER_CHAINS = tuple(
    (INDEX[f"{side}_wrist"], *(INDEX[f"{side}_{finger}_{i}"] for i in range(1, 5)))
    for side in SIDES for finger in FINGERS
)
NATIVE_LAYOUT = Layout(
    names=NATIVE_NAMES,
    parents=(-1, 0, 1, 1, 3, 4, 1, 6, 7) + tuple(
        parent for chain in FINGER_CHAINS for parent in chain[:-1]
    ),
    shoulders=(3, 6),
    lower_torso=(0,),
    palms=tuple(tuple(INDEX[f"{side}_{key}"] for key in ("wrist", "index_1", "middle_1", "pinky_1")) for side in SIDES),
    fingers=FINGER_CHAINS,
    spatial_pairs=(("wrist_to_wrist", 5, 8), ("left_wrist_to_shoulder", 5, 3), ("right_wrist_to_shoulder", 8, 6),
                   ("left_palm_base_to_neck", 5, 1), ("right_palm_base_to_neck", 8, 1)),
    contact_pairs=tuple(
        (f"{left}_tip_to_{right}_tip", INDEX[f"left_{left}_4"], INDEX[f"right_{right}_4"])
        for left in FINGERS for right in FINGERS
    ),
)


def native_motion(record):
    """Decode a SignaCulture MotionFile as one batch. Units are approximate rig units."""
    if record.get("rig_version") != "signaculture-v1" or not record.get("frames"):
        raise ValueError("Expected nonempty signaculture-v1 motion")
    frames = record["frames"]
    pose = torch.full((1, len(frames), len(NATIVE_NAMES), 3), torch.nan)
    for time, frame in enumerate(frames):
        for name, xyz in frame["joints"].items():
            if name in INDEX:
                pose[0, time, INDEX[name]] = torch.as_tensor(xyz)
    confidence = torch.isfinite(pose).all(dim=-1).float()
    for time, frame in enumerate(frames):
        for side in SIDES:
            if (
                (frame.get("hand_tracking") or {}).get(side, {}).get("observed") is False
                or (frame.get("playback_quality") or {}).get(f"{side}_hand_rejected") is True
            ):
                confidence[0, time, [INDEX[n] for n in NATIVE_NAMES if n.startswith(f"{side}_") and n not in NATIVE_BODY]] = 0
        if frame.get("playback_origin") == "endpoint_hold":
            # Preserve the display endpoint but keep it below the contact confidence gate.
            confidence[0, time] *= .6
    result = dict(pose=pose, times=torch.tensor([[f["time_ms"] / 1000 for f in frames]], dtype=torch.float64),
                  confidence=confidence, face=[f.get("face") for f in frames])
    if any(f.get("head_rotation") is not None for f in frames):
        empty = torch.full((3, 3), torch.nan)
        result["head_rotation"] = torch.stack([
            torch.as_tensor(f["head_rotation"], dtype=pose.dtype).reshape(3, 3)
            if f.get("head_rotation") is not None else empty for f in frames
        ])[None]
    return result


def mediapipe_world(body, left, right, *, hand_to_body_axes, hand_to_body_scale):
    """Align wrist-local hands to body-world wrists into native layout [B,T,49,3].

    body is [B,T,33,3]; hands are [B,T,21,3], with pose axes already conventional
    x-right/y-up/z-toward-camera. Explicitly supply calibrated hand-to-body
    rotations and positive isotropic scales, each as a left/right pair.
    Missing points may be NaN. SDK world depth remains monocular.
    """
    if body.ndim != 4 or body.shape[-2:] != (33, 3):
        raise ValueError("Expected MediaPipe pose world [B,T,33,3]")
    if len(hand_to_body_axes) != 2 or len(hand_to_body_scale) != 2:
        raise ValueError("Supply separate left and right calibration")
    hips = body[..., [23, 24], :].mean(dim=-2)
    neck = body[..., [11, 12], :].mean(dim=-2)
    packed = [hips[..., None, :], neck[..., None, :], body[..., [0, 11, 13, 15, 12, 14, 16], :]]
    for hand, wrist, axes, scale in zip((left, right), (15, 16), hand_to_body_axes, hand_to_body_scale):
        axes = torch.as_tensor(axes, device=body.device, dtype=body.dtype)
        eye = torch.eye(3, device=body.device, dtype=body.dtype)
        if (
            hand.shape != (*body.shape[:2], 21, 3) or axes.shape != (3, 3) or not 0 < scale < float("inf")
            or not torch.allclose(axes.T @ axes, eye, atol=1e-4) or not torch.allclose(axes.det(), eye[0, 0], atol=1e-4)
        ):
            raise ValueError("Invalid hand calibration or landmark shape")
        local = hand[..., 1:, :] - hand[..., :1, :]
        packed.append(body[..., wrist:wrist + 1, :] + scale * (local @ axes.T))
    return torch.cat(packed, dim=-2)


class NativeMotionDataset(Dataset):
    """Lazy per-clip normalizer for manifest paths. Returns unpadded sequences."""

    def __init__(self, paths, normalizer):
        if normalizer.config.layout != NATIVE_LAYOUT or normalizer.config.coordinate_system != "rig_3d_xyz":
            raise ValueError("Use the native rig layout and coordinate system")
        self.paths, self.normalizer = tuple(Path(path) for path in paths), normalizer

    @classmethod
    def from_transition_manifest(cls, manifest, split, normalizer):
        """Use the existing source-video split without fitting any dataset statistics."""
        if split not in {"train", "val", "test"}:
            raise ValueError("Unknown transition split")
        clips = json.loads(Path(manifest).read_text())["clips"]
        return cls((row["path"] for row in clips if row["split"] == split), normalizer)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        source = native_motion(json.loads(path.read_text()))
        with torch.no_grad():
            features = self.normalizer(**source)
        return {
            "path": str(path), "xyz": features["canonical_motion"][0],
            "mask": features["valid"][0], "observed": features["observed"][0],
            "times": features["times"][0], "face": features["face"],
        }
