def objective(pred, duration, target, mask, seconds):
    weights = mask.to(pred.dtype)
    weights[:, :, 18:60] *= 2
    position = ((pred-target).square().sum(-1)*weights).sum()/weights.sum().clamp_min(1)
    adjacent = mask[:,1:] & mask[:,:-1]
    velocity = ((pred.diff(dim=1)-target.diff(dim=1)).abs().sum(-1)*adjacent).sum()/adjacent.sum().clamp_min(1)
    return position + .2*velocity + .1*(duration-seconds.log()).square().mean()
