"""Right-handed torso frames and invertible clip-relative transforms."""
import torch
from torch.nn.functional import avg_pool1d


def unit(v, eps=1e-6):
    return v / v.norm(dim=-1, keepdim=True).clamp_min(eps)


def basis(lateral, upward, eps=1e-6):
    x = unit(lateral, eps)
    z = unit(torch.cross(x, unit(upward, eps), dim=-1), eps)
    y = torch.cross(z, x, dim=-1)
    return torch.stack((x, y, z), dim=-1)


def geometry(pose, confidence, config):
    """Inputs [B,T,J,3], [B,T,J]. Returns raw torso-local and clip-reference coordinates."""
    left, right = config.layout.shoulders
    lower = list(config.layout.lower_torso)
    center = pose[..., lower, :].mean(dim=-2)
    lateral = pose[..., left, :] - pose[..., right, :]
    upward = (pose[..., left, :] + pose[..., right, :]) * .5 - center
    width = lateral.norm(dim=-1)
    valid = confidence[..., [left, right, *lower]].amin(dim=-1) >= config.min_confidence
    valid = valid & (width > config.epsilon) & (upward.norm(dim=-1) > config.epsilon)
    valid = valid & (torch.cross(unit(lateral), unit(upward), dim=-1).norm(dim=-1) > .05)
    weights = torch.where(valid, confidence[..., [left, right, *lower]].amin(dim=-1), 0)
    scale = width.masked_fill(~valid, torch.inf).sort(dim=1).values
    count = valid.sum(dim=1)
    scale = scale.gather(1, ((count - 1).clamp_min(0) // 2)[:, None]).squeeze(1)
    scale = torch.where(count > 0, scale, 1)
    first = valid.to(torch.int64).argmax(dim=1)
    batch = torch.arange(pose.shape[0], device=pose.device)
    eye = torch.eye(3, dtype=pose.dtype, device=pose.device).expand(pose.shape[0], 3, 3)
    ref_rotation = torch.where((count > 0)[:, None, None], basis(lateral[batch, first], upward[batch, first]), eye)
    ref_center = torch.where((count > 0)[:, None], center[batch, first], 0)
    vectors = torch.cat((center, unit(lateral), unit(upward)), dim=-1)
    radius = config.frame_radius
    window = 2 * radius + 1
    pooled = avg_pool1d((vectors * weights[..., None]).transpose(1, 2), window, stride=1, padding=radius).transpose(1, 2)
    pooled_weight = avg_pool1d(weights[:, None], window, stride=1, padding=radius).transpose(1, 2)
    smoothed_center, x, y = (pooled / pooled_weight.clamp_min(config.epsilon)).split(3, dim=-1)
    usable = (pooled_weight[..., 0] > 0) & (torch.cross(x, y, dim=-1).norm(dim=-1) > .05)
    rotation = torch.where(usable[..., None, None], basis(x, y), ref_rotation[:, None])
    center = torch.where(usable[..., None], smoothed_center, ref_center[:, None])
    local = torch.matmul(pose - center[..., None, :], rotation) / scale[:, None, None, None]
    reference = torch.matmul(pose - ref_center[:, None, None, :], ref_rotation[:, None]) / scale[:, None, None, None]
    return local, reference, {
        "center": center, "rotation": rotation, "scale": scale,
        "reference_center": ref_center, "reference_rotation": ref_rotation,
        "torso_observed": valid, "calibrated": count > 0, "local_frame_supported": usable,
        "torso_translation": torch.matmul((center - ref_center[:, None])[:, None], ref_rotation[:, None]).squeeze(1) / scale[:, None, None],
        "torso_rotation": ref_rotation[:, None].transpose(-1, -2) @ rotation,
    }


def invert(points, transforms, *, torso_local=True):
    """Recover shared-space coordinates. Masks are carried by the caller."""
    center = transforms["center"] if torso_local else transforms["reference_center"][:, None]
    rotation = transforms["rotation"] if torso_local else transforms["reference_rotation"][:, None]
    return (points * transforms["scale"][:, None, None, None]) @ rotation.transpose(-1, -2) + center[..., None, :]
