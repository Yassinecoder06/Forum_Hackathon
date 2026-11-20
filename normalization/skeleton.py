"""Bone directions, direction-derived local rotations, and fixed-length kinematics."""
import torch

from .geometry import basis, unit


def bone_features(points, mask, layout, eps):
    parents = list(layout.parents[1:])
    vectors = points[..., 1:, :] - points[..., parents, :]
    lengths = vectors.norm(dim=-1)
    valid = mask[..., 1:] & mask[..., parents] & (lengths > eps)
    directions = torch.where(valid[..., None], unit(vectors, eps), 0)
    return directions, lengths, valid


def palm_and_finger_features(points, mask, layout, eps):
    palms, valid_palms = [], []
    for wrist, index, middle, pinky in layout.palms:
        across = points[..., index, :] - points[..., pinky, :]
        along = points[..., middle, :] - points[..., wrist, :]
        rotation = basis(across, along, eps)
        valid = mask[..., [wrist, index, middle, pinky]].all(dim=-1) & (rotation[..., 2].norm(dim=-1) > .5)
        palms.append(torch.where(valid[..., None, None], rotation, 0))
        valid_palms.append(valid)
    angles, valid_angles = [], []
    for chain in layout.fingers:
        for first, middle, last in zip(chain, chain[1:], chain[2:]):
            a = points[..., middle, :] - points[..., first, :]
            b = points[..., last, :] - points[..., middle, :]
            valid = mask[..., [first, middle, last]].all(dim=-1) & (a.norm(dim=-1) > eps) & (b.norm(dim=-1) > eps)
            cosine = (unit(a) * unit(b)).sum(dim=-1)
            sine = torch.cross(unit(a), unit(b), dim=-1).norm(dim=-1)
            angle = torch.atan2(torch.where(valid, sine, 0), torch.where(valid, cosine, 1))
            angles.append(torch.where(valid, angle, 0))
            valid_angles.append(valid)
    return {
        "palm_rotation": torch.stack(palms, dim=-3) if palms else points.new_empty((*points.shape[:2], 0, 3, 3)),
        "palm_valid": torch.stack(valid_palms, dim=-1) if palms else mask[..., :0],
        "finger_angle": torch.stack(angles, dim=-1) if angles else points.new_empty((*points.shape[:2], 0)),
        "finger_valid": torch.stack(valid_angles, dim=-1) if angles else mask[..., :0],
    }


def skew(v):
    x, y, z = v.unbind(-1)
    zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), dim=-1).unflatten(-1, (3, 3))


def rotation_matrix(rotvec):
    """Rodrigues' formula with stable small-angle limits."""
    angle = rotvec.norm(dim=-1, keepdim=True)[..., None]
    cross = skew(rotvec)
    eye = torch.eye(3, dtype=rotvec.dtype, device=rotvec.device)
    return eye + torch.sinc(angle / torch.pi) * cross + .5 * torch.sinc(angle / (2 * torch.pi)).square() * (cross @ cross)


def shortest_rotation(a, b):
    cross = torch.cross(a, b, dim=-1)
    cosine = (a * b).sum(dim=-1, keepdim=True).clamp(-1, 1)
    fallback = torch.nn.functional.one_hot(a.abs().argmin(dim=-1), 3).to(a)
    axis = torch.where((cross.norm(dim=-1, keepdim=True) < 1e-6) & (cosine < 0), torch.cross(a, fallback, dim=-1), cross)
    # b==a is allowed, with a zero axis-angle.
    angle = torch.atan2(cross.norm(dim=-1, keepdim=True), cosine)
    return unit(axis) * angle


class Skeleton:
    """Rest points [J,3] in canonical shoulder-width units, with optional local bounds [J-1,3]."""

    def __init__(self, layout, rest, rotation_limit=None):
        if rest.shape != (len(layout.names), 3) or not torch.isfinite(rest).all():
            raise ValueError("Rest skeleton does not match the layout")
        offsets = rest[1:] - rest[list(layout.parents[1:])]
        if (offsets.norm(dim=-1) <= 1e-6).any():
            raise ValueError("Reference skeleton contains a zero-length edge")
        if rotation_limit is not None and (
            rotation_limit.shape != offsets.shape or not torch.isfinite(rotation_limit).all() or (rotation_limit < 0).any()
        ):
            raise ValueError("Expected finite nonnegative per-axis rotation limits")
        self.layout, self.rest = layout, rest
        self.offsets = offsets
        self.rotation_limit = rotation_limit

    def reconstruct(self, points, valid):
        """Fixed-length directional reconstruction; returns positions and minimal local axis-angle rotations."""
        directions, _, bone_valid = bone_features(points, valid, self.layout, 1e-6)
        offsets = self.offsets.to(points)
        roots = [points[..., 0, :]]
        identity = torch.eye(3, dtype=points.dtype, device=points.device).expand(*points.shape[:2], 3, 3)
        rotations, local = [identity], []
        for j, parent in enumerate(self.layout.parents[1:]):
            ancestor = rotations[parent]
            rest = unit(offsets[j]).expand_as(directions[..., j, :])
            direction = (ancestor.transpose(-1, -2) @ directions[..., j, :, None]).squeeze(-1)
            direction = torch.where(bone_valid[..., j, None], direction, rest)
            axis_angle = shortest_rotation(rest, direction)
            world = ancestor @ rotation_matrix(axis_angle)
            roots.append(roots[parent] + world @ offsets[j])
            rotations.append(world)
            local.append(axis_angle)
        return torch.stack(roots, dim=-2), torch.stack(local, dim=-2) if local else points[..., :0, :]

    def forward_kinematics(self, root, local):
        """Forward kinematics with rotations indexed by the child of each edge."""
        offsets = self.offsets.to(root)
        local_matrices = rotation_matrix(local)
        positions = [root]
        rotations = [torch.eye(3, dtype=root.dtype, device=root.device).expand(*root.shape[:-1], 3, 3)]
        for j, parent in enumerate(self.layout.parents[1:]):
            world = rotations[parent] @ local_matrices[..., j, :, :]
            positions.append(positions[parent] + world @ offsets[j])
            rotations.append(world)
        return torch.stack(positions, dim=-2)
