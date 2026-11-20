"""Opt-in, batched sign pose normalization in explicit shared XYZ coordinates."""
import torch
from torch import nn

from .config import Config, Layout
from .contacts import detect_contacts, pair_features
from .geometry import geometry, invert
from .skeleton import Skeleton, bone_features, palm_and_finger_features
from .temporal import derivatives, temporal

__all__ = ["Config", "Layout", "SignPoseNormalizer", "Skeleton", "invert"]


class SignPoseNormalizer(nn.Module):
    """Normalize [B,T,J,3] shared XYZ with optional [B,T,J] confidence.

    Supply timestamps in seconds as [B,T]. Head orientations [B,T,3,3] are
    in shared coordinates; facial features are passed through unchanged.
    An upstream detector adapter must align origins, axes and units.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, pose, *, times, confidence=None, face=None, head_rotation=None, skeleton=None):
        config = self.config
        if pose.ndim != 4 or pose.shape[-2:] != (len(config.layout.names), 3) or min(pose.shape[:2]) < 1:
            raise ValueError("Expected nonempty pose [B,T,J,3] matching the layout")
        if pose.dtype not in (torch.float32, torch.float64):
            raise ValueError("Use float32 or float64 coordinates")
        if times.shape != pose.shape[:2] or not torch.isfinite(times).all():
            raise ValueError("Expected finite timestamps [B,T]")
        times = times.to(pose)
        if (times[:, 1:] <= times[:, :-1]).any():
            raise ValueError("Timestamps must increase strictly")
        if confidence is None:
            confidence = torch.ones_like(pose[..., 0])
        if confidence.shape != pose.shape[:-1]:
            raise ValueError("Confidence must be [B,T,J]")
        confidence = confidence.to(pose)
        confidence = torch.where(
            torch.isfinite(pose).all(dim=-1) & torch.isfinite(confidence),
            confidence.clamp(0, 1), 0,
        )
        pose = torch.where(torch.isfinite(pose), pose, 0)
        torso, reference, transforms = geometry(pose, confidence, config)
        canonical, valid, trusted, gap, outlier = temporal(reference, confidence, times, config)
        calibrated = transforms["calibrated"][:, None, None]
        valid, trusted, gap = valid & calibrated, trusted & calibrated, gap & calibrated
        canonical = torch.where(valid[..., None], canonical, 0)
        direction, length, bone_valid = bone_features(canonical, valid, config.layout, config.epsilon)
        spatial = pair_features(torso, trusted & transforms["local_frame_supported"][..., None], confidence, config.layout.spatial_pairs)
        output = {
            "canonical_motion": canonical, "torso_local": torso, "geometric_motion": reference,
            "transforms": transforms, "times": times, "observed": trusted, "valid": valid,
            "interpolated": gap, "outlier": outlier, "confidence": confidence,
            "bone_direction": direction, "source_bone_length": length, "bone_valid": bone_valid,
            "spatial": spatial, "contacts": detect_contacts(reference, trusted, confidence, config),
            "face": face, **derivatives(canonical, valid, times),
            **palm_and_finger_features(canonical, valid, config.layout, config.epsilon),
        }
        output["torso_palm_rotation"] = (
            transforms["torso_rotation"].transpose(-1, -2).unsqueeze(-3) @ output["palm_rotation"]
        )
        output["torso_palm_valid"] = output["palm_valid"] & transforms["local_frame_supported"][..., None]
        if head_rotation is not None:
            if head_rotation.shape != (*times.shape, 3, 3):
                raise ValueError("Head orientation must be [B,T,3,3]")
            head_rotation = head_rotation.to(pose)
            head_valid = torch.isfinite(head_rotation).all(dim=(-1, -2)) & transforms["calibrated"][:, None]
            output["head_rotation"] = transforms["reference_rotation"][:, None].transpose(-1, -2) @ torch.where(
                head_valid[..., None, None], head_rotation, 0,
            )
            output["head_valid"] = head_valid
        if skeleton is not None:
            if skeleton.layout != config.layout:
                raise ValueError("Skeleton layout must match the input layout")
            reconstructed, rotations = skeleton.reconstruct(canonical, valid)
            output.update(anatomical_motion=reconstructed, local_rotation=rotations)
        return output
