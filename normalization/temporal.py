"""Time-aware filtering that retains the input time grid and observation masks."""
import torch


def temporal(points, confidence, times, config):
    observed = confidence >= config.min_confidence
    outlier = torch.zeros_like(observed)
    if times.shape[1] > 2:
        duration = times[:, 2:] - times[:, :-2]
        alpha = ((times[:, 1:-1] - times[:, :-2]) / duration)[..., None, None]
        before, after = points[:, :-2], points[:, 2:]
        predicted = before + alpha * (after - before)
        error = (points[:, 1:-1] - predicted).norm(dim=-1)
        endpoint_motion = (after - before).norm(dim=-1)
        outlier[:, 1:-1] = (
            observed[:, :-2] & observed[:, 2:] & observed[:, 1:-1]
            & (duration[..., None] <= config.max_gap_seconds)
            & (confidence[:, 1:-1] < torch.minimum(confidence[:, :-2], confidence[:, 2:]))
            & (error > config.spike_distance + 2 * endpoint_motion)
        )
    trusted = observed & ~outlier
    index = torch.arange(times.shape[1], device=times.device)[None, :, None]
    prev = torch.where(trusted, index, -1).cummax(dim=1).values
    nxt = torch.where(trusted, index, times.shape[1]).flip((1,)).cummin(dim=1).values.flip((1,))
    p = prev.clamp_min(0)
    n = nxt.clamp_max(times.shape[1] - 1)
    ptime = times[..., None].expand_as(trusted).gather(1, p)
    ntime = times[..., None].expand_as(trusted).gather(1, n)
    span = ntime - ptime
    gap = ~trusted & (prev >= 0) & (nxt < times.shape[1]) & (span <= config.max_gap_seconds)
    fraction = ((times[..., None] - ptime) / span.clamp_min(config.epsilon)).clamp(0, 1)
    ppoint = points.gather(1, p[..., None].expand_as(points))
    npoint = points.gather(1, n[..., None].expand_as(points))
    repaired = torch.where(gap[..., None], torch.lerp(ppoint, npoint, fraction[..., None]), points)
    if times.shape[1] > 2:
        weight = config.smoothing * (1 - confidence[:, 1:-1])
        weight = weight * trusted[:, :-2] * trusted[:, 1:-1] * trusted[:, 2:]
        weight = weight * (duration[..., None] <= config.max_gap_seconds)
        speed_change = (repaired[:, 1:-1] - predicted).norm(dim=-1)
        weight = weight * (speed_change < config.spike_distance * .5)
        repaired = torch.cat((repaired[:, :1], torch.lerp(repaired[:, 1:-1], predicted, weight[..., None]), repaired[:, -1:]), dim=1)
    valid = trusted | gap
    return torch.where(valid[..., None], repaired, 0), valid, trusted, gap, outlier


def derivatives(points, valid, times):
    """Backward differences in seconds, with separate validity for each derivative."""
    dt = times[:, 1:] - times[:, :-1]
    velocity_mask = valid[:, 1:] & valid[:, :-1]
    velocity = (points[:, 1:] - points[:, :-1]) / dt[..., None, None]
    acceleration_mask = velocity_mask[:, 1:] & velocity_mask[:, :-1]
    acceleration = 2 * (velocity[:, 1:] - velocity[:, :-1]) / (dt[:, 1:] + dt[:, :-1])[..., None, None]
    return {
        "velocity": torch.where(velocity_mask[..., None], velocity, 0), "velocity_valid": velocity_mask,
        "acceleration": torch.where(acceleration_mask[..., None], acceleration, 0), "acceleration_valid": acceleration_mask,
    }
