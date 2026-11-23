"""Finger depth in the converter: learned C0 depth when given, else bone-length lift + depth-sign repair.

Authored geometric hands only; no interpreter data."""
import numpy as np
import pytest

from backend.app.services import lsf_pose_convert as C
from backend.tests.test_lsf_pose_convert import fixture

BONES = (0.04, 0.025, 0.02)                     # proximal, middle, distal phalanx (metres)


def hand3d(curl_deg, side="left"):
    """One hand (21, 3), wrist at 0, fingers up (+y), back of the hand towards the camera: curling a finger
    (positive flexion) moves it away from the camera (-z). Mirrored in x for the right hand."""
    s = 1.0 if side == "left" else -1.0
    h = np.zeros((21, 3))
    a = np.radians(curl_deg)
    for k, f in enumerate(("index", "middle", "ring", "pinky")):
        mcp = np.array([s * (0.03 - 0.02 * k), 0.09, 0.0])
        chain = C.FINGER_CHAINS[f]
        h[chain[0]] = mcp
        p, ang = mcp, 0.0
        for i, length in enumerate(BONES):
            ang += a
            p = p + length * np.array([0.0, np.cos(ang), -np.sin(ang)])
            h[chain[i + 1]] = p
    d = np.array([s * 0.6, 0.8, 0.0])               # straight thumb
    for i, j in enumerate(C.FINGER_CHAINS["thumb"]):
        h[j] = np.array([s * 0.02, 0.02, 0.0]) + 0.03 * i * d
    return h


@pytest.mark.parametrize("side", ["left", "right"])
def test_authored_hand_is_within_limits_and_lift_artifact_is_not(side):
    h = hand3d(50, side)[None]
    assert C.finger_angles(h, side)["index_pip"][0] == pytest.approx(50, abs=1e-6)
    assert C.finger_violations(h, side) == 0
    bent_back = h.copy()
    bent_back[..., 2] *= -1                          # what an always-towards-the-camera lift does to this hand
    assert C.finger_violations(bent_back, side) > 0.5


@pytest.mark.parametrize("side", ["left", "right"])
def test_sign_fix_repairs_depth_only(side):
    h = np.stack([hand3d(c, side) for c in (0, 20, 50, 70)])
    bent_back = h.copy()
    bent_back[..., 2] *= -1
    fixed = C.hand_sign_fix(bent_back, side)
    # greedy, bone by bone (as in sign-model hand_kin): a strong curl is fully repaired, a mild one may stay
    assert C.finger_violations(fixed, side) < C.finger_violations(bent_back, side)
    assert C.finger_violations(fixed[3:], side) == 0
    assert all(C.finger_violations(fixed[i:i + 1], side) <= C.finger_violations(bent_back[i:i + 1], side) for i in range(4))
    assert np.allclose(fixed[..., :2], bent_back[..., :2])          # x/y unchanged
    for f, chain in C.FINGER_CHAINS.items():                       # bone lengths unchanged
        pts = [0, *chain]
        for a, b in zip(pts[:-1], pts[1:]):
            assert np.allclose(np.linalg.norm(fixed[:, b] - fixed[:, a], axis=-1),
                               np.linalg.norm(bent_back[:, b] - bent_back[:, a], axis=-1))
    assert np.allclose(C.hand_sign_fix(h, side), h)                # a plausible hand is left alone


def clip_with_hand(T=40):
    """Converter fixture whose left hand alternates flat (full bone lengths visible) and curled 60 degrees."""
    x = fixture(T=T)
    for t in range(T):
        h = hand3d(0 if (t // 5) % 2 == 0 else 60) / C.SHOULDER_WIDTH_M
        x[t, C.LEFT_HAND:C.LEFT_HAND + 21] = x[t, C.LEFT_HAND] + h[:, :2]
    return x


def left_hand(motion, frame):
    j = motion["frames"][frame]["joints"]
    pts = [j["left_wrist"]] + [j[f"left_{f}_{i}"] for f in C.FINGERS for i in (1, 2, 3, 4)]
    return np.array(pts)[None]


def test_lift_with_sign_repair_has_fewer_violations_and_same_xy(monkeypatch):
    x = clip_with_hand()
    fixed = C.engine_poses_to_motion(x)
    monkeypatch.setattr(C, "hand_sign_fix", lambda h, side: h)    # the previous lift, no repair
    raw = C.engine_poses_to_motion(x)
    curled = [t for t in range(len(x)) if (t // 5) % 2]
    v_raw = np.mean([C.finger_violations(left_hand(raw, t), "left") for t in curled])
    v_fix = np.mean([C.finger_violations(left_hand(fixed, t), "left") for t in curled])
    assert v_raw > 0.3 and v_fix < 0.5 * v_raw
    for t in (0, 7):
        assert np.allclose(left_hand(fixed, t)[..., :2], left_hand(raw, t)[..., :2])


def test_learned_hand_depth_is_used_only_when_flagged():
    x = clip_with_hand()
    hz = np.zeros(x.shape[:2], np.float32)
    true = hand3d(60) / C.SHOULDER_WIDTH_M
    hz[:, C.LEFT_HAND:C.LEFT_HAND + 21] = true[:, 2]
    x3 = np.concatenate([x, hz[..., None]], axis=-1)
    m = C.engine_poses_to_motion(x3, hand_depth=True)
    h = left_hand(m, 7)[0]
    assert np.allclose(h[1:, 2] - h[0, 2], true[1:, 2] * C.SHOULDER_WIDTH_M, atol=1e-4)
    assert C.finger_violations(left_hand(m, 7), "left") == 0
    # the right hand has no learned depth (all zero): it falls back to the lift + repair
    assert m["frames"][7]["joints"]["right_index_4"] == C.engine_poses_to_motion(x, hand_depth=True)["frames"][7]["joints"]["right_index_4"]
    # without the flag a third channel is ignored (lexicon / capture depth in other units)
    assert C.engine_poses_to_motion(x3)["frames"][7]["joints"] == C.engine_poses_to_motion(x)["frames"][7]["joints"]
