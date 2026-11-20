"""Torch-free helpers for windowed emotion2vec scoring.

The worker scores overlapping windows so the avatar face can follow the voice over
time. Each window score remains an unvalidated model hypothesis.
"""
import math

WINDOW_S = 2.0
HOP_S = 1.0
PAUSE_RMS = 0.01  # Same pause threshold as the acoustic frames in audio_analysis.


def window_bounds(n_samples, rate, window_s=WINDOW_S, hop_s=HOP_S):
    """Sample ranges of overlapping windows; the last window always reaches the end."""
    if n_samples <= 0 or rate <= 0:
        return []
    size = max(1, round(window_s * rate))
    hop = max(1, round(hop_s * rate))
    if n_samples <= size:
        return [(0, n_samples)]
    bounds = [(start, start + size) for start in range(0, n_samples - size + 1, hop)]
    if bounds[-1][1] < n_samples:
        bounds.append((n_samples - size, n_samples))
    return bounds


def rms(samples):
    if len(samples) == 0:
        return 0.0
    return math.sqrt(sum(float(v) * float(v) for v in samples) / len(samples))


def build_timeline(bounds, rate, scores):
    """Pair window bounds with normalized score dicts (None = skipped silent window)."""
    if len(bounds) != len(scores):
        raise ValueError('Each emotion window needs exactly one score entry')
    timeline = []
    for (start, end), values in zip(bounds, scores):
        if values is None:
            continue
        start_ms = round(start / rate * 1000)
        end_ms = max(start_ms + 1, round(end / rate * 1000))
        timeline.append({'start_ms': start_ms, 'end_ms': end_ms, 'scores': values})
    return timeline
