"""Measure Noa bind offsets and export through the existing position-based avatar API."""
import json
import struct
from pathlib import Path

import torch

from .adapters import FINGERS, INDEX, NATIVE_LAYOUT, SIDES
from .geometry import basis
from .skeleton import Skeleton


def _node_matrix(node):
    if "matrix" in node:
        return torch.tensor(node["matrix"], dtype=torch.float64).reshape(4, 4).T
    x, y, z, w = node.get("rotation", (0, 0, 0, 1))
    matrix = torch.eye(4, dtype=torch.float64)
    matrix[:3, :3] = torch.tensor([
        [1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w)],
        [2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w)],
        [2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y)],
    ], dtype=matrix.dtype) * torch.tensor(node.get("scale", (1, 1, 1)), dtype=matrix.dtype)
    matrix[:3, 3] = torch.tensor(node.get("translation", (0, 0, 0)), dtype=matrix.dtype)
    return matrix


def noa_skeleton(path, *, rotation_limit=None):
    """Load actual Noa bind transforms; fail loudly for a missing named rig node."""
    with Path(path).open("rb") as stream:
        magic, version, _ = struct.unpack("<4sII", stream.read(12))
        length, kind = struct.unpack("<I4s", stream.read(8))
        if (magic, version, kind) != (b"glTF", 2, b"JSON"):
            raise ValueError("Expected a glTF 2 GLB")
        gltf = json.loads(stream.read(length))
    nodes = gltf["nodes"]
    parents = {child: index for index, node in enumerate(nodes) for child in node.get("children", [])}
    names = {nodes[index].get("name"): index for skin in gltf["skins"] for index in skin["joints"]}

    def world(index):
        result = _node_matrix(nodes[index])
        while index in parents:
            index = parents[index]
            result = _node_matrix(nodes[index]) @ result
        return result[:3, 3]

    bone_names = ["Hips", "Neck", "Head", "LeftArm", "LeftForeArm", "LeftHand",
                  "RightArm", "RightForeArm", "RightHand"]
    bone_names.extend(f"{side.title()}Hand{finger.title()}{n}" for side in SIDES for finger in FINGERS for n in range(1, 5))
    if missing := set(bone_names) - names.keys():
        raise ValueError(f"Missing Noa rig bones: {sorted(missing)}")
    rest = torch.stack([world(names[name]) for name in bone_names])
    origin = rest[0]
    axis = rest[3] - rest[6]
    scale = axis.norm()
    rotation = basis(axis, (rest[3] + rest[6]) / 2 - origin)
    skeleton = Skeleton(NATIVE_LAYOUT, ((rest - origin) @ rotation / scale).float(), rotation_limit)
    return skeleton, float(scale)


def export_motion(result, source, shoulder_width, *, batch=0):
    """Export a position target for the existing Noa IK. Keeps the source time grid."""
    if not torch.isfinite(result["positions"]).all():
        raise ValueError("Retargeted positions are nonfinite")
    positions = result["positions"][batch].cpu() * shoulder_width
    positions += positions.new_tensor([0, 1, 0])
    valid = result["valid"][batch].cpu()
    if not valid[:, :9].all():
        raise ValueError("Native motion export needs all body joints; inspect the missing-joint mask")
    if positions.shape[0] != len(source["frames"]):
        raise ValueError("Source timestamps do not match retargeted frames")
    frames = []
    observed = result["observed"][batch].cpu()
    hands = {
        side: [INDEX[f"{side}_wrist"], *(INDEX[f"{side}_{finger}_{n}"] for finger in FINGERS for n in range(1, 5))]
        for side in SIDES
    }
    for index, frame in enumerate(source["frames"]):
        item = {key: frame[key] for key in ("time_ms", "face", "face_blendshapes", "playback_quality", "playback_origin") if key in frame}
        if result.get("head_valid") is not None and result["head_valid"][batch, index]:
            item["head_rotation"] = result["head_rotation"][batch, index].flatten().tolist()
        item["joints"] = {
            name: positions[index, j].tolist() for j, name in enumerate(NATIVE_LAYOUT.names)
            if valid[index, j]
        }
        item["hand_tracking"] = {}
        prior_tracking = frame.get("hand_tracking") or {}
        for side, indices in hands.items():
            complete = bool(observed[index, indices].all())
            metadata = prior_tracking.get(side, {})
            if complete:
                source_name = metadata.get("source", "native-position")
            elif valid[index, indices].all():
                source_name = "normalization-short-gap"
            else:
                source_name = "normalization-sparse"
            item["hand_tracking"][side] = {**metadata, "observed": complete, "source": source_name}
        frames.append(item)
    return {
        **source, "frames": frames, "joints": list(NATIVE_LAYOUT.names),
        "capture_method": "anatomy-normalized-native-target",
    }
