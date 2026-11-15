"""Deterministic one-to-one fallback assignment; no detector confidence invented."""
import itertools
import math
from copy import deepcopy


def roi_box(cx, cy, size, width, height):
    size = min(int(size), width, height)
    return max(0, min(width-size, int(cx-size/2))), max(0, min(height-size, int(cy-size/2))), size


def map_roi_hand(hand, x0, y0, size, width, height):
    mapped = deepcopy(hand)
    for point in mapped.landmark:
        point.x = (x0 + point.x*size)/width
        point.y = (y0 + point.y*size)/height
        point.z *= size/width
    return mapped


def consistent_hand(a, b, width, height):
    """Independent crop scales must agree in original pixels; not accuracy proof."""
    if len(a.landmark) != 21 or len(b.landmark) != 21:
        return False
    palm = math.hypot((a.landmark[5].x-a.landmark[17].x)*width,
                      (a.landmark[5].y-a.landmark[17].y)*height)
    if palm < 4:
        return False
    errors = [math.hypot((x.x-y.x)*width,(x.y-y.y)*height) for x,y in zip(a.landmark,b.landmark)]
    return max(errors) <= max(3.,palm*.35) and sum(errors)/21 <= max(1.5,palm*.15)


def agreed_candidates(forward, backward, width, height):
    """Keep actual forward detections corroborated in reverse; collapse repeats."""
    accepted=[]
    for hand in forward:
        if not any(consistent_hand(hand, other, width, height) for other in backward):
            continue
        if any(math.hypot(hand.landmark[0].x-other.landmark[0].x,hand.landmark[0].y-other.landmark[0].y)<.03
               and consistent_hand(hand,other,width,height) for other in accepted):
            continue
        accepted.append(hand)
    return accepted


def assign_hands(detections, wrists, existing=None, previous=None, max_distance=.18):
    """Return side->detection index. Existing Holistic sides stay locked.

    Ambiguous candidates are rejected. Temporal continuity is a small tie breaker,
    never a reason to associate a distant detection with the wrong pose wrist.
    """
    existing = existing or {}
    previous = previous or {}
    sides = [s for s in ('left', 'right') if not existing.get(s)]
    candidates = []
    for choices in itertools.product(range(-1, len(detections)), repeat=len(sides)):
        used = [i for i in choices if i >= 0]
        if len(used) != len(set(used)):
            continue
        score = 0.
        valid = True
        for side, index in zip(sides, choices):
            if index < 0:
                score += max_distance
                continue
            point = detections[index]
            distance = math.dist(point, wrists[side])
            # Do not reuse the hand already represented by a locked Holistic side.
            locked_distance = min((math.dist(point, wrists[s]) for s in wrists if existing.get(s)), default=math.inf)
            if distance >= max_distance or locked_distance + .015 < distance:
                valid = False
                break
            score += distance + .15 * min(math.dist(point, previous[side]), .1) if side in previous else distance
        if valid:
            candidates.append((score, choices))
    if not candidates:
        return {}
    candidates.sort()
    best = candidates[0][1]
    # If equally plausible side assignments disagree, abstain on those sides.
    near = [choice for score, choice in candidates if score < candidates[0][0] + .008]
    return {side: index for j, (side, index) in enumerate(zip(sides, best))
            if index >= 0 and all(choice[j] == index for choice in near)}
