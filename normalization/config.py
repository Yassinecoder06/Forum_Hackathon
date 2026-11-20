"""Joint layout and processing settings, independent of any detector."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Layout:
    names: tuple[str, ...]
    parents: tuple[int, ...]
    shoulders: tuple[int, int]  # anatomical left, right
    lower_torso: tuple[int, ...]
    palms: tuple[tuple[int, int, int, int], ...] = ()  # wrist, index, middle, pinky
    spatial_pairs: tuple[tuple[str, int, int], ...] = ()
    contact_pairs: tuple[tuple[str, int, int], ...] = ()
    fingers: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        size = len(self.names)
        if not size or len(set(self.names)) != size or len(self.parents) != size:
            raise ValueError("Expected unique names and one parent per joint")
        if self.parents[0] != -1 or any(not 0 <= p < j for j, p in enumerate(self.parents[1:], 1)):
            raise ValueError("Expected one root followed by parent-before-child joints")
        indices = (*self.shoulders, *self.lower_torso,
                   *(j for group in (*self.palms, *self.fingers) for j in group),
                   *(j for _, a, b in (*self.spatial_pairs, *self.contact_pairs) for j in (a, b)))
        if not self.lower_torso or self.shoulders[0] == self.shoulders[1] or any(not 0 <= j < size for j in indices):
            raise ValueError("Invalid landmark indices")


@dataclass(frozen=True)
class Config:
    layout: Layout
    coordinate_system: str  # shared XYZ with consistent axes and units
    min_confidence: float = .5
    frame_radius: int = 2
    max_gap_seconds: float = .15
    smoothing: float = .15
    spike_distance: float = .3  # shoulder widths
    contact_distance: float = .12
    contact_min_confidence: float = .7
    epsilon: float = 1e-6

    def __post_init__(self):
        if self.coordinate_system not in {"shared_3d", "rig_3d_xyz", "aligned_monocular_xyz"}:
            raise ValueError("Align all landmarks into shared XYZ first; image-plane data is unsupported")
        if (
            self.frame_radius < 0 or self.epsilon <= 0 or self.max_gap_seconds < 0
            or not 0 < self.min_confidence <= 1 or not 0 <= self.smoothing <= 1
            or not 0 <= self.contact_min_confidence <= 1
            or self.contact_distance <= 0 or self.spike_distance <= 0
        ):
            raise ValueError("Invalid normalization thresholds")
