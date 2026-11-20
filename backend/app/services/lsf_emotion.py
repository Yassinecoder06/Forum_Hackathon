"""Voice emotion -> HOW the LSF engines sign (speed and size of the manual signing), never WHAT is signed.

Input: emotion2vec scores of the voice (labels angry, disgusted, fearful, happy, neutral, other, sad, surprised,
unknown). The dominant emotional label and its score set a speed factor (the whole clip is retimed) and an
amplitude factor (arm and hand points spread around their clip mean). The face is deliberately NOT changed: in LSF
eyebrows, eyes and head carry grammar (questions, negation), so emotion must not override them.

The values are the heuristic starting points of sign-model/signlib.EMOTIONS (brow offsets removed). They are a
documented guess, not a model learnt from deaf signers: the emotion-transfer plan that would learn them needs
recordings with deaf signers (sign-model/EMOTION_TRANSFER_PLAN.md, on hold).
"""
from __future__ import annotations

import numpy as np

EMOTIONS = {  # speed > 1 = faster; amplitude > 1 = larger movements
    "neutral": dict(speed=1.00, amplitude=1.00),
    "happy": dict(speed=1.10, amplitude=1.10),
    "angry": dict(speed=1.15, amplitude=1.20),
    "sad": dict(speed=0.85, amplitude=0.85),
    "surprised": dict(speed=1.10, amplitude=1.15),
    "fearful": dict(speed=1.10, amplitude=0.95),
    "disgusted": dict(speed=0.95, amplitude=1.00),
}
NOT_EMOTIONS = {"neutral", "other", "unknown"}
ARM_HAND = np.r_[13:23, 23:65]           # elbows, wrists, MediaPipe hand points of the pose, both hands


def emotion_params(scores: dict[str, float] | None, strength: float = 1.0) -> dict | None:
    """Dominant label of the voice scores -> {label, confidence, speed, amplitude}; None when nothing applies."""
    if not scores:
        return None
    valid = {k: float(v) for k, v in scores.items() if k in EMOTIONS or k in NOT_EMOTIONS}
    if not valid:
        return None
    label = max(valid, key=valid.get)
    confidence = float(np.clip(valid[label], 0.0, 1.0))
    if label in NOT_EMOTIONS:
        label = "neutral"
    c = confidence * float(np.clip(strength, 0.0, 1.0))
    base, target = EMOTIONS["neutral"], EMOTIONS[label]
    return {"label": label, "confidence": round(confidence, 3),
            **{k: round(base[k] + c * (target[k] - base[k]), 4) for k in base}}


def retime(x: np.ndarray, speed: float) -> np.ndarray:
    if abs(speed - 1.0) < 1e-3 or len(x) < 2:
        return x
    n = max(2, int(round(len(x) / speed)))
    idx = np.linspace(0, len(x) - 1, n)
    lo = np.floor(idx).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (idx - lo).reshape(-1, *([1] * (x.ndim - 1)))
    return x[lo] * (1 - w) + x[hi] * w


def apply_emotion(poses: np.ndarray, params: dict | None) -> np.ndarray:
    """Retime the clip and scale arm/hand motion around its own mean. Face points are untouched."""
    if not params:
        return poses
    x = retime(np.asarray(poses, dtype=np.float32), params["speed"]).copy()
    amp = params["amplitude"]
    if abs(amp - 1.0) > 1e-3:
        arm = x[:, ARM_HAND]
        mean = np.nanmean(arm, axis=0, keepdims=True)
        x[:, ARM_HAND] = mean + amp * (arm - mean)
    return x
