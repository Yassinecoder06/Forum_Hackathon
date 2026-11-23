"""Convert LSF engine poses (157 points, normalised 2D) into a `signaculture-v1` MotionFile.

Engine pose layout (sign-model / signlib), one row per frame at 25 fps:
    0:23    upper body = MediaPipe pose landmarks 0..22 (11/12 shoulders, 13/14 elbows, 15/16 wrists)
    23:44   left hand  = MediaPipe hand landmarks 0..20 (signer's left)
    44:65   right hand
    65:157  face: lips 40, left eyebrow 10, right eyebrow 10, left eye 16, right eye 16
            (sorted MediaPipe face-mesh indices of each FaceLandmarksConnections set)
Normalisation: origin at the shoulder midpoint, unit = shoulder width, y up, x towards the
signer's left. Only x/y are used, except the learned finger depth of the C0 hand refiner
(channel 2 of the hand points, when the engine says so): other engine depth is not reliable,
so depth is re-estimated here from bone lengths.

MotionFile (frontend/src/types/contract.ts): metres, y up, z towards the camera, named joints
(`head`, `neck`, `hips`, `left_shoulder`, `left_elbow`, `left_wrist`, `left_thumb_1..4`,
`left_index_1..4`, ...), optional per-frame `face` ratios (`mouth_open`, `brow_left`,
`brow_right`) measured exactly like backend/app/workers/face_features.py.

Depth lifting (an approximation, documented in docs/pipeline/LSF_ENGINES.md):
    for each segment parent->child with 3D length L and observed 2D length d,
    z_child = z_parent + sqrt(max(L^2 - d^2, 0)), i.e. the foreshortening is always resolved
    towards the camera. L is the 95th percentile of d over the clip (a segment is seen at its
    full length when it is parallel to the image), never below an anatomical prior for the arm.
    Wrists are then kept in front of the torso and the face. Fingers: the learned depth when
    present, else the same lift; then a depth-sign repair against anatomical joint limits
    (hand_sign_fix). The avatar's runtime joint-limit clamp stays the final safety net.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

N_POINTS = 157
ENGINE_FPS = 25.0
SHOULDER_WIDTH_M = 0.36
NECK_HEIGHT_M = 1.45          # shoulder midpoint; puts the nose near 1.6 m like the captured assets
LEFT_HAND, RIGHT_HAND = 23, 44
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
POSE = {"head": 0, "left_shoulder": 11, "right_shoulder": 12, "left_elbow": 13, "right_elbow": 14,
        "left_wrist": 15, "right_wrist": 16}
CORE_JOINTS = ["head", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist",
               "right_wrist", "hips", "neck"]
FINGER_JOINTS = [f"{side}_{finger}_{i}" for side in ("left", "right") for finger in FINGERS for i in (1, 2, 3, 4)]

# Anatomical priors in shoulder widths (biacromial ~0.36-0.40 m, upper arm ~0.29 m, forearm ~0.25 m).
UPPER_ARM_PRIOR, FOREARM_PRIOR = 0.78, 0.68
# In front of the shoulder plane (metres): chest surface and face surface.
TORSO_FRONT_M, FACE_FRONT_M, HEAD_Z_M = 0.11, 0.13, 0.09

# Face points: index inside the 92-point face block for the MediaPipe face-mesh ids used by
# face_features.measure_face (13/14 inner lips, 105/159 and 334/386 brow-to-lid, 33/263 eye corners).
_LIPS = sorted([61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 185, 40, 39, 37, 0, 267, 269, 270, 409, 78, 95,
                88, 178, 87, 14, 317, 402, 318, 324, 308, 191, 80, 81, 82, 13, 312, 311, 310, 415])
_LEFT_BROW = sorted([276, 283, 282, 295, 285, 300, 293, 334, 296, 336])
_RIGHT_BROW = sorted([46, 53, 52, 65, 55, 70, 63, 105, 66, 107])
_LEFT_EYE = sorted([263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398])
_RIGHT_EYE = sorted([33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173])
_FACE_BLOCKS = [(_LIPS, 65), (_LEFT_BROW, 105), (_RIGHT_BROW, 115), (_LEFT_EYE, 125), (_RIGHT_EYE, 141)]


def face_index(mesh_id: int) -> int:
    for ids, start in _FACE_BLOCKS:
        if mesh_id in ids:
            return start + ids.index(mesh_id)
    raise KeyError(mesh_id)


# measure_face multiplies by width/height to keep the historical normalised-image scale; the
# engine coordinates are isotropic, so the 16:9 broadcast aspect is applied explicitly.
FACE_ASPECT = 16 / 9


def _interp_nan(x: np.ndarray) -> np.ndarray:
    """Linear interpolation of missing values over time (per coordinate); edges are held."""
    x = x.copy()
    flat = x.reshape(len(x), -1)
    t = np.arange(len(x))
    for j in range(flat.shape[1]):
        col = flat[:, j]
        ok = np.isfinite(col)
        if ok.all() or not ok.any():
            continue
        col[~ok] = np.interp(t[~ok], t[ok], col[ok])
    return x


def resample(x: np.ndarray, fps_in: float, fps_out: float) -> np.ndarray:
    """Linear resampling along time; keeps the first and last frame."""
    if abs(fps_in - fps_out) < 1e-6 or len(x) < 2:
        return x
    duration = (len(x) - 1) / fps_in
    n = int(round(duration * fps_out)) + 1
    t = np.linspace(0, len(x) - 1, n)
    lo = np.floor(t).astype(int)
    hi = np.minimum(lo + 1, len(x) - 1)
    w = (t - lo).reshape(-1, *([1] * (x.ndim - 1)))
    return x[lo] * (1 - w) + x[hi] * w


def _segment_length(a: np.ndarray, b: np.ndarray, prior: float = 0.0) -> float:
    """3D length estimate: the 95th percentile of the 2D length (segment seen side-on).

    With an anatomical prior the estimate is kept within [prior, 1.2 * prior], so a clip whose
    proportions are off (e.g. lexicon clips from another camera) cannot push the hands far
    forward; frames where the 2D length exceeds it simply get no extra depth."""
    d = np.linalg.norm(b - a, axis=-1)
    d = d[np.isfinite(d)]
    observed = float(np.percentile(d, 95)) if len(d) else 0.0
    return float(np.clip(observed, prior, 1.2 * prior)) if prior else observed


def _lift(parent_z: np.ndarray, a: np.ndarray, b: np.ndarray, length: float) -> np.ndarray:
    d2 = np.sum((b - a) ** 2, axis=-1)
    return parent_z + np.sqrt(np.clip(length ** 2 - d2, 0.0, None))


def face_ratios(frame2d: np.ndarray) -> dict[str, float] | None:
    """Same quantities as face_features.measure_face, from one (157, 2) normalised frame."""
    p = lambda mesh_id: frame2d[face_index(mesh_id)]  # noqa: E731
    e = p(263) - p(33)
    distance = float(np.hypot(*e))
    if not math.isfinite(distance) or distance < 1e-4:
        return None
    # y is up in engine coordinates (down in images); the projection uses absolute values.
    n = np.array([-e[1], e[0]]) / distance
    gap = lambda a, b: abs(float((p(a) - p(b)) @ n)) / distance * FACE_ASPECT  # noqa: E731
    result = {"mouth_open": gap(13, 14), "brow_left": gap(105, 159), "brow_right": gap(334, 386)}
    # Lip shape and eye opening, in eye-corner distances (isotropic engine units; y up). Relative to the
    # clip's neutral (face_calibration) they drive lip rounding/spreading, lip corners and eye opening.
    dist = lambda a, b: float(np.hypot(*(p(a) - p(b)))) / distance  # noqa: E731
    result.update(mouth_width=dist(61, 291), mouth_corner=float((p(61)[1] + p(291)[1]) / 2 - (p(13)[1] + p(14)[1]) / 2) / distance,
                  eye_open_left=dist(386, 374), eye_open_right=dist(159, 145))
    return result if all(math.isfinite(v) for v in result.values()) else None


# Neutral face channels of the B2 lexicon rest pose (engineering reference, not a linguistic neutral).
FACE_NEUTRAL_DEFAULT = {"brow_left": 0.428, "brow_right": 0.424, "mouth_width": 0.572, "mouth_corner": -0.024,
                        "eye_open_left": 0.070, "eye_open_right": 0.069}
NONMANUAL_TYPES = ("question_yn", "question_wh", "negation")
RAMP_MS = 160.0


def nonmanual_channels(t_ms: float, spans: list[dict]) -> dict[str, float]:
    """Grammatical non-manual markers at time t: brow raise (yes/no question), brow furrow (wh-question),
    head shake (negation) and a slight forward head tilt for questions. Smooth ramps in and out; kept as
    separate channels so emotion can never overwrite them. Simplified textbook markers, not validated."""
    out = {"nm_brow_raise": 0.0, "nm_brow_furrow": 0.0, "nm_head_yaw": 0.0, "nm_head_pitch": 0.0}
    for sp in spans:
        a, b, kind = float(sp["start_ms"]), float(sp["end_ms"]), sp["type"]
        if not a - RAMP_MS < t_ms < b + RAMP_MS:
            continue
        ramp = min(1.0, (t_ms - (a - RAMP_MS)) / RAMP_MS, ((b + RAMP_MS) - t_ms) / RAMP_MS)
        ramp = ramp * ramp * (3 - 2 * ramp)
        if kind == "question_yn":
            out["nm_brow_raise"] = max(out["nm_brow_raise"], 0.65 * ramp)
            out["nm_head_pitch"] += 0.10 * ramp
        elif kind == "question_wh":
            out["nm_brow_furrow"] = max(out["nm_brow_furrow"], 0.6 * ramp)
            out["nm_head_pitch"] += 0.06 * ramp
        elif kind == "negation":
            out["nm_head_yaw"] += 0.17 * ramp * math.sin(2 * math.pi * 2.2 * (t_ms - a) / 1000.0)
            out["nm_brow_furrow"] = max(out["nm_brow_furrow"], 0.25 * ramp)
    return {k: round(v, 5) for k, v in out.items()}


# ----------------------------------------------------------------------------- finger kinematics
# Port of sign-model/hand_kin.py (joint angles, anatomical limits, depth-sign repair). Hand layout (MediaPipe):
# 0 wrist; thumb 1-4 (CMC, MCP, IP, tip); index 5-8, middle 9-12, ring 13-16, pinky 17-20 (MCP, PIP, DIP, tip).
# Flexion = signed angle between parent and child bone around the palm's lateral axis (index MCP -> pinky MCP),
# positive towards the palm; FLEX_SIGN was fixed on real MediaPipe-3D signing (x towards the signer's left, y up,
# z towards the camera). Limits in degrees: active range of motion plus a 10-degree tolerance for tracking noise.
FLEX_SIGN = {"left": 1.0, "right": -1.0}
FINGER_CHAINS = {"thumb": (1, 2, 3, 4), "index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16),
                 "pinky": (17, 18, 19, 20)}
FINGER_LIMITS = {"mcp": (-30, 100), "pip": (-10, 115), "dip": (-15, 95), "thumb_mcp": (-15, 75), "thumb_ip": (-20, 90)}
SPREAD_MAX = 35.0


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def _signed_angle(p: np.ndarray, c: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Signed angle (degrees) from bone p to bone c around `axis` (all (..., 3))."""
    return np.degrees(np.arctan2((np.cross(p, c) * axis).sum(-1), (p * c).sum(-1)))


def _excess(v: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.maximum(lo - v, 0) + np.maximum(v - hi, 0)


def finger_angles(h: np.ndarray, side: str) -> dict[str, np.ndarray]:
    """h (T, 21, 3) one hand -> {joint: (T,) degrees}: flexion of every finger joint and spread between neighbours."""
    h = np.asarray(h, dtype=np.float64)
    lat = _unit(h[:, 17] - h[:, 5]) * FLEX_SIGN[side]
    out = {}
    for f, (a, b, c, d) in FINGER_CHAINS.items():
        bones = [h[:, a] - h[:, 0], h[:, b] - h[:, a], h[:, c] - h[:, b], h[:, d] - h[:, c]]
        if f == "thumb":            # the thumb flexes around the palm normal, not the lateral axis
            ax = _unit(np.cross(h[:, 5] - h[:, 0], h[:, 17] - h[:, 0])) * FLEX_SIGN[side]
            out["thumb_mcp"] = _signed_angle(bones[1], bones[2], ax)
            out["thumb_ip"] = _signed_angle(bones[2], bones[3], ax)
            continue
        out[f"{f}_mcp"] = _signed_angle(bones[0], bones[1], lat)
        out[f"{f}_pip"] = _signed_angle(bones[1], bones[2], lat)
        out[f"{f}_dip"] = _signed_angle(bones[2], bones[3], lat)
    n = _unit(np.cross(h[:, 5] - h[:, 0], h[:, 17] - h[:, 0]))
    proj = {}
    for f in ("index", "middle", "ring", "pinky"):
        v = h[:, FINGER_CHAINS[f][1]] - h[:, FINGER_CHAINS[f][0]]
        proj[f] = _unit(v - (v * n).sum(-1, keepdims=True) * n)
    for f1, f2 in (("index", "middle"), ("middle", "ring"), ("ring", "pinky")):
        out[f"spread_{f1}_{f2}"] = np.degrees(np.arccos(np.clip((proj[f1] * proj[f2]).sum(-1), -1, 1)))
    return out


def _limit_of(joint: str) -> tuple[float, float]:
    if joint.startswith("thumb"):
        return FINGER_LIMITS[joint]
    if joint.startswith("spread"):
        return (-1e9, SPREAD_MAX)
    return FINGER_LIMITS[joint.split("_")[1]]


def finger_violations(h: np.ndarray, side: str, margin: float = 0.0) -> float:
    """Share of joint-frames beyond the anatomical limits (margin: degrees of tolerance; 20 = 'severe')."""
    bad = tot = 0
    for k, v in finger_angles(h, side).items():
        lo, hi = _limit_of(k)
        m = ((v < lo - margin) | (v > hi + margin)) & np.isfinite(v)
        bad += int(m.sum())
        tot += m.size
    return bad / max(tot, 1)


def hand_sign_fix(h: np.ndarray, side: str) -> np.ndarray:
    """Depth-sign repair of a bone-length lift (sign-model hand_kin.sign_fix). The lift resolves every finger bone's
    foreshortening towards the camera, which bends curled fingers backwards on a hand seen from the back. For each
    finger bone after the palm (MCP->PIP, PIP->DIP, DIP->tip; thumb MCP->IP, IP->tip), in order, keep the lift's depth
    or mirror it, whichever breaks the limits of the two joints it touches less (ties keep the lift). x/y and bone
    lengths are unchanged. h (T, 21, 3) wrist-relative -> repaired copy."""
    h = np.asarray(h, dtype=np.float64)
    lat = _unit(h[:, 17] - h[:, 5]) * FLEX_SIGN[side]
    tax = _unit(np.cross(h[:, 5] - h[:, 0], h[:, 17] - h[:, 0])) * FLEX_SIGN[side]
    out = h.copy()
    with np.errstate(invalid="ignore"):
        for f, (a, b, c, d) in FINGER_CHAINS.items():
            chain = [0, a, b, c, d]
            bones = [h[:, chain[i + 1]] - h[:, chain[i]] for i in range(4)]
            if f == "thumb":
                axis, lims, first = tax, [None, None, FINGER_LIMITS["thumb_mcp"], FINGER_LIMITS["thumb_ip"]], 2
            else:
                axis, lims, first = lat, [None, FINGER_LIMITS["mcp"], FINGER_LIMITS["pip"], FINGER_LIMITS["dip"]], 1
            for i in range(first, 4):
                cand = []
                for s in (1.0, -1.0):
                    bi = bones[i].copy()
                    bi[:, 2] = s * bones[i][:, 2]
                    e = _excess(_signed_angle(bones[i - 1], bi, axis), *lims[i])
                    if i < 3:
                        e = e + _excess(_signed_angle(bi, bones[i + 1], axis), *lims[i + 1])
                    cand.append((e, bi))
                flip = cand[1][0] < cand[0][0] - 1e-6
                bones[i] = np.where(flip[:, None], cand[1][1], cand[0][1])
            pos = h[:, 0].copy()
            for i in range(4):
                pos = pos + bones[i]
                out[:, chain[i + 1]] = pos
    return out


def lift_to_3d(x2d: np.ndarray, hand_z: np.ndarray | None = None) -> np.ndarray:
    """(T, 157, 2) normalised -> (T, 157, 3) in metres, y up, z towards the camera.

    Body points get depth from bone lengths; face points keep the head depth. Fingers: when `hand_z` (T, 157) carries a
    learned depth for a hand (C0 hand refiner: relative to the wrist, towards the camera, shoulder widths), finger
    depth = wrist depth + hand_z, else the bone-length lift; then the depth-sign repair (hand_sign_fix) on both."""
    T = len(x2d)
    xy = x2d.astype(np.float64) * SHOULDER_WIDTH_M
    xy[..., 1] += NECK_HEIGHT_M
    z = np.zeros((T, N_POINTS))
    z[:, :] = np.nan
    z[:, 11] = z[:, 12] = 0.0
    z[:, 0] = HEAD_Z_M
    z[:, 65:] = HEAD_Z_M
    sw = SHOULDER_WIDTH_M
    for side, (s, e, w, hand) in {"left": (11, 13, 15, LEFT_HAND), "right": (12, 14, 16, RIGHT_HAND)}.items():
        # hand landmark 0 is the better wrist when present (the engines attach the hand to it)
        wrist = np.where(np.isfinite(xy[:, hand]), xy[:, hand], xy[:, w])
        xy[:, w] = wrist
        xy[:, hand] = wrist
        upper = _segment_length(xy[:, s], xy[:, e], UPPER_ARM_PRIOR * sw)
        fore = _segment_length(xy[:, e], wrist, FOREARM_PRIOR * sw)
        z[:, e] = _lift(z[:, s], xy[:, s], xy[:, e], upper)
        z[:, w] = _lift(z[:, e], xy[:, e], wrist, fore)
        # keep the hands in front of the chest and of the face (monocular depth can put them inside)
        rel = xy[:, w] - np.array([0.0, NECK_HEIGHT_M])
        in_torso = (np.abs(rel[:, 0]) < 0.6 * sw) & (rel[:, 1] < 0.1 * sw) & (rel[:, 1] > -1.8 * sw)
        in_face = (np.abs(xy[:, w, 0] - xy[:, 0, 0]) < 0.35 * sw) & (np.abs(xy[:, w, 1] - xy[:, 0, 1]) < 0.45 * sw)
        front = np.where(in_face, FACE_FRONT_M, np.where(in_torso, TORSO_FRONT_M, -np.inf))
        z[:, w] = np.maximum(z[:, w], front)
        z[:, hand] = z[:, w]
        fingers = slice(hand + 1, hand + 21)
        learned = None if hand_z is None else np.asarray(hand_z, dtype=np.float64)[:, fingers]
        if learned is not None and np.isfinite(learned).all() and np.abs(learned).max() > 1e-6:
            z[:, fingers] = z[:, [w]] + learned * sw
        else:
            # finger chains: 0 -> 1 -> 2 -> 3 -> 4 for each finger (0 = wrist)
            for k in range(5):
                chain = [0] + [1 + 4 * k + i for i in range(4)]
                for parent, child in zip(chain[:-1], chain[1:]):
                    a, b = xy[:, hand + parent], xy[:, hand + child]
                    length = _segment_length(a, b)
                    z[:, hand + child] = _lift(z[:, hand + parent], a, b, length)
        # depth-sign repair on either depth (measured on C0 + hand refiner: violated joints 9.5 -> 6.9 %)
        h = np.concatenate([xy[:, hand:hand + 21], z[:, hand:hand + 21, None]], axis=-1)
        z[:, fingers] = hand_sign_fix(h - h[:, :1], side)[:, 1:, 2] + z[:, [w]]
    out = np.concatenate([xy, z[..., None]], axis=-1)
    return out


def engine_poses_to_motion(poses: np.ndarray, *, fps_in: float = ENGINE_FPS, fps_out: float = ENGINE_FPS,
                           engine: str = "", text: str = "", face_calibration: dict[str, float] | None = None,
                           nonmanual: list[dict] | None = None, hand_depth: bool = False) -> dict[str, Any]:
    """(T, 157, 2|3) engine poses -> MotionFile dict (signaculture-v1).

    nonmanual: [{"start_ms", "end_ms", "type": question_yn | question_wh | negation}] grammatical markers baked into
    each frame's face channels (nm_*), see nonmanual_channels.
    hand_depth: channel 2 of the hand points is a learned finger depth (C0 hand refiner, metadata `hand_depth`); any
    other third channel (lexicon / capture depth in other units) is ignored."""
    x = np.asarray(poses, dtype=np.float64)
    if x.ndim != 3 or x.shape[1] != N_POINTS or x.shape[2] < 2:
        raise ValueError(f"expected (T, {N_POINTS}, 2|3) engine poses, got {x.shape}")
    if len(x) < 2:
        raise ValueError("at least two frames are needed")
    hand_z = None
    if hand_depth and x.shape[2] >= 3:
        hand_z = resample(_interp_nan(x[..., 2]), fps_in, fps_out)
    x2d = x[..., :2]
    hand_seen = {side: np.isfinite(x2d[:, start:start + 21]).all(axis=(1, 2))
                 for side, start in (("left", LEFT_HAND), ("right", RIGHT_HAND))}
    core = [0, 11, 12, 13, 14, 15, 16]
    if not np.isfinite(x2d[:, core]).any(axis=0).all():
        raise ValueError("engine output has no usable shoulders/arms")
    x2d = _interp_nan(x2d)
    x2d = resample(x2d, fps_in, fps_out)
    for side in hand_seen:
        idx = np.clip(np.round(np.arange(len(x2d)) * fps_in / fps_out).astype(int), 0, len(hand_seen[side]) - 1)
        hand_seen[side] = hand_seen[side][idx]
    xyz = lift_to_3d(x2d, hand_z)
    frames = []
    for i in range(len(xyz)):
        f = xyz[i]
        joints = {name: [round(float(v), 5) for v in f[idx]] for name, idx in POSE.items()}
        joints["hips"] = [0.0, 1.0, 0.0]
        # outer eye corners (MediaPipe face mesh 263 = left, 33 = right): head tilt/turn for the body layer
        for name, mesh_id in (("left_eye", 263), ("right_eye", 33)):
            p = f[face_index(mesh_id)]
            if np.isfinite(p).all():
                joints[name] = [round(float(v), 5) for v in p]
        joints["neck"] = [round(float(v), 5) for v in (f[11] + f[12]) / 2]
        tracking = {}
        for side, start in (("left", LEFT_HAND), ("right", RIGHT_HAND)):
            observed = bool(hand_seen[side][i]) and bool(np.isfinite(f[start:start + 21]).all())
            tracking[side] = {"observed": observed, "source": f"lsf-engine-{engine}" if engine else "lsf-engine"}
            if not observed:
                continue
            for k, finger in enumerate(FINGERS):
                for j in range(1, 5):
                    joints[f"{side}_{finger}_{j}"] = [round(float(v), 5) for v in f[start + 1 + 4 * k + (j - 1)]]
        frame = {"time_ms": round(i * 1000 / fps_out), "joints": joints, "hand_tracking": tracking,
                 "playback_origin": "source"}
        face = face_ratios(x2d[i])
        if face:
            frame["face"] = {k: round(v, 5) for k, v in face.items()}
            if nonmanual:
                frame["face"].update(nonmanual_channels(frame["time_ms"], nonmanual))
        frames.append(frame)
    motion: dict[str, Any] = {
        "rig_version": "signaculture-v1",
        "fps": fps_out,
        "duration_ms": frames[-1]["time_ms"],
        "joints": CORE_JOINTS,
        "tracked_joints": CORE_JOINTS + FINGER_JOINTS,
        "frames": frames,
        "synthetic": True,
        "capture_method": f"lsf-engine-{engine}-2d-bone-length-lift" if engine else "lsf-engine-2d-bone-length-lift",
        "warnings": [
            "Mouvement généré automatiquement : ce n'est pas une traduction LSF validée.",
            "Profondeur estimée à partir des longueurs des segments (projection 2D) : approximative.",
        ],
        "attribution": {
            "credit": f"Moteur LSF SignaCulture ({engine.upper()})" if engine else "Moteur LSF SignaCulture",
            "source_url": "https://github.com/Yassinecoder06/Forum_Hackathon/blob/main/docs/pipeline/LSF_ENGINES.md",
            "license": "research-preview",
            "changes": "Poses générées par un modèle, converties en articulations 3D approximatives.",
        },
    }
    if text:
        motion["source_text"] = text
    if face_calibration:
        motion["face_calibration"] = {k: float(face_calibration.get(k, FACE_NEUTRAL_DEFAULT[k])) for k in FACE_NEUTRAL_DEFAULT}
    if nonmanual:
        motion["nonmanual"] = [dict(sp) for sp in nonmanual]
    return motion
