"""Named signed distances and confidence-gated contact candidates."""
import torch


def pair_features(points, observed, confidence, pairs):
    first = [a for _, a, _ in pairs]
    second = [b for _, _, b in pairs]
    vector = points[..., first, :] - points[..., second, :]
    valid = observed[..., first] & observed[..., second]
    return {
        "names": tuple(name for name, _, _ in pairs),
        "relative": torch.where(valid[..., None], vector, 0),
        "distance": torch.where(valid, vector.norm(dim=-1), 0),
        "valid": valid,
        "confidence": torch.where(valid, torch.minimum(confidence[..., first], confidence[..., second]), 0),
    }


def detect_contacts(points, trusted, confidence, config):
    pairs = config.layout.contact_pairs
    result = pair_features(points, trusted, confidence, pairs)
    result["active"] = (
        result["valid"] & (result["confidence"] >= config.contact_min_confidence)
        & (result["distance"] <= config.contact_distance)
    )
    return result
