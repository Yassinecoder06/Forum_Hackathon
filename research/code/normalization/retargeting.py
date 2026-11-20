"""Fixed-length inverse kinematics for an explicitly supplied reference skeleton."""
from dataclasses import dataclass

import torch

from .skeleton import bone_features, palm_and_finger_features, rotation_matrix
from .temporal import derivatives


@dataclass(frozen=True)
class IKWeights:
    position: float = 1.
    rotation: float = .02
    contact: float = 8.
    handshape: float = .12
    smooth: float = .00001
    jointlimits: float = .2


def weighted_mean(value, weight):
    return (value * weight).sum() / weight.sum().clamp_min(1)


@torch.enable_grad()
def retarget(output, skeleton, *, weights=IKWeights(), steps=40, learning_rate=.03):
    """Fit [B,T,J-1,3] local axis-angle parameters via differentiable FK.

    The optimizer itself is offline. Gradients for model training can flow
    through Skeleton.forward_kinematics and the normalizer directly.
    """
    if steps < 0 or learning_rate <= 0:
        raise ValueError("Expected nonnegative steps and a positive learning rate")
    target = output["canonical_motion"].detach()
    valid = output["valid"]
    position_weight = torch.where(output["interpolated"], .25, output["confidence"].detach()) * valid
    # Give each wrist its own weight so densely sampled fingers don't dominate placement.
    wrists = [wrist for wrist, *_ in skeleton.layout.palms]
    if wrists:
        position_weight[..., wrists] *= 4
    bone_valid = output["bone_valid"]
    _, initial = skeleton.reconstruct(target, valid)
    initial = initial.detach()
    limits = skeleton.rotation_limit
    if limits is not None:
        limits = limits.to(target)
        initial = initial.clamp(-limits, limits)
    root = target[..., 0, :]
    local = initial.clone().requires_grad_()
    optimizer = torch.optim.Adam([local], lr=learning_rate)
    pairs = skeleton.layout.contact_pairs
    first, second = [a for _, a, _ in pairs], [b for _, _, b in pairs]
    contacts = output["contacts"]
    hand_bones = sorted({index - 1 for chain in skeleton.layout.fingers for index in chain[1:]})
    target_rotation = rotation_matrix(initial)
    final = {}
    for iteration in range(steps + 1):
        optimizer.zero_grad()
        points = skeleton.forward_kinematics(root, local)
        direction, _, _ = bone_features(points, torch.ones_like(valid), skeleton.layout, 1e-6)
        position = weighted_mean((points - target).square().sum(dim=-1), position_weight)
        rotation = weighted_mean((rotation_matrix(local) - target_rotation).square().sum(dim=(-1, -2)), bone_valid)
        relative = points[..., first, :] - points[..., second, :]
        contact = weighted_mean((relative - contacts["relative"].detach()).square().sum(dim=-1), contacts["active"] * contacts["confidence"].detach())
        direction_error = (direction - output["bone_direction"].detach()).square().sum(dim=-1)
        handshape = weighted_mean(direction_error[..., hand_bones], bone_valid[..., hand_bones])
        palm = palm_and_finger_features(points, torch.ones_like(valid), skeleton.layout, 1e-6)
        handshape = handshape + weighted_mean(
            (palm["palm_rotation"] - output["palm_rotation"].detach()).square().sum(dim=(-1, -2)), output["palm_valid"],
        )
        dynamics = derivatives(points, valid, output["times"].detach())
        smooth = weighted_mean(
            (dynamics["acceleration"] - output["acceleration"].detach()).square().sum(dim=-1),
            dynamics["acceleration_valid"],
        )
        jointlimits = (local.abs() - limits).clamp_min(0).square().mean() if limits is not None else local.sum() * 0
        final = dict(position=position, rotation=rotation, contact=contact, handshape=handshape, smooth=smooth, jointlimits=jointlimits)
        objective = sum(getattr(weights, key) * value for key, value in final.items())
        if iteration == steps:
            break
        objective.backward()
        torch.nn.utils.clip_grad_norm_([local], 5.)
        optimizer.step()
        if limits is not None:
            with torch.no_grad():
                local.clamp_(-limits, limits)
    return {
        "positions": points.detach(), "local_rotation": local.detach(),
        "losses": {name: value.detach() for name, value in final.items()},
        "contact_distance": relative.detach().norm(dim=-1),
        "valid": valid, "observed": output["observed"], "times": output["times"], "face": output["face"],
        "head_rotation": output.get("head_rotation"), "head_valid": output.get("head_valid"),
    }
