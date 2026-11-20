"""Engine pose (157 points, normalised 2D) -> signaculture-v1 MotionFile conversion.

All inputs are authored geometric fixtures; no interpreter data is used."""
import math

import numpy as np
import pytest

from backend.app.services import lsf_pose_convert as C


def fixture(T=50, forearm_2d=0.5, hand_nan_frames=()):
    """Arms hanging at the sides, forearms raised towards the camera (foreshortened)."""
    x = np.full((T, C.N_POINTS, 2), np.nan, dtype=np.float32)
    x[:, 0] = [0.0, 0.55]                       # nose
    for side, sign, (s, e, w, hand) in (("left", 1, (11, 13, 15, C.LEFT_HAND)), ("right", -1, (12, 14, 16, C.RIGHT_HAND))):
        x[:, s] = [0.5 * sign, 0.0]
        x[:, e] = [0.55 * sign, -C.UPPER_ARM_PRIOR]                   # upper arm seen at full length
        x[:, w] = x[:, e] + [0.0, forearm_2d]                         # forearm foreshortened (points at the camera)
        x[:, hand] = x[:, w]
        for k in range(5):                                            # straight fingers going up, 0.1 per phalanx
            for i in range(4):
                x[:, hand + 1 + 4 * k + i] = x[:, w] + [(k - 2) * 0.05 * sign, 0.15 + 0.1 * i]
        for t in hand_nan_frames:
            x[t, hand:hand + 21] = np.nan
    # face: neutral-ish layout around the nose (eye corners 0.3 apart)
    x[:, 65:157] = x[:, [0]]
    x[:, C.face_index(33)] = [-0.15, 0.7]
    x[:, C.face_index(263)] = [0.15, 0.7]
    x[:, C.face_index(159)] = [-0.1, 0.72]
    x[:, C.face_index(105)] = [-0.1, 0.72 + 0.3 * 0.24]
    x[:, C.face_index(386)] = [0.1, 0.72]
    x[:, C.face_index(334)] = [0.1, 0.72 + 0.3 * 0.24]
    x[:, C.face_index(13)] = [0.0, 0.4]
    x[:, C.face_index(14)] = [0.0, 0.4 - 0.3 * 0.03]
    return x


def joint(motion, frame, name):
    return np.array(motion["frames"][frame]["joints"][name])


def test_face_index_layout():
    assert C.face_index(0) == 65                      # first sorted lip landmark
    assert C.face_index(415) == 65 + 39               # last lip landmark
    assert C.face_index(276) == 105                   # first left-eyebrow landmark (sorted)
    assert C.face_index(46) == 115
    assert C.face_index(249) == 125
    assert C.face_index(7) == 141
    assert C.face_index(246) == 156                   # last sorted right-eye landmark
    with pytest.raises(KeyError):
        C.face_index(1)


def test_motion_contract_and_scale():
    m = C.engine_poses_to_motion(fixture(), engine="c0", text="bonjour")
    assert m["rig_version"] == "signaculture-v1" and m["fps"] == 25
    assert m["frames"][0]["time_ms"] == 0 and m["frames"][-1]["time_ms"] == m["duration_ms"] == 1960
    times = [f["time_ms"] for f in m["frames"]]
    assert times == sorted(set(times))
    for f in m["frames"]:
        assert set(m["joints"]).issubset(f["joints"])
        assert all(len(v) == 3 and all(math.isfinite(c) for c in v) for v in f["joints"].values())
    width = np.linalg.norm(joint(m, 0, "left_shoulder") - joint(m, 0, "right_shoulder"))
    assert width == pytest.approx(C.SHOULDER_WIDTH_M, abs=1e-4)
    assert joint(m, 0, "neck")[1] == pytest.approx(C.NECK_HEIGHT_M, abs=1e-4)
    assert 1.55 < joint(m, 0, "head")[1] < 1.7
    assert joint(m, 0, "left_shoulder")[0] > 0 > joint(m, 0, "right_shoulder")[0]   # +x = signer's left
    assert m["synthetic"] is True
    assert any("pas une traduction LSF validée" in w for w in m["warnings"])


def test_depth_from_bone_lengths():
    m = C.engine_poses_to_motion(fixture(forearm_2d=0.4))
    s, e, w = (joint(m, 10, f"left_{n}") for n in ("shoulder", "elbow", "wrist"))
    sw = C.SHOULDER_WIDTH_M
    # upper arm seen side-on: no depth; its 3D length is the prior
    assert e[2] == pytest.approx(0.0, abs=1e-6)
    assert np.linalg.norm(e - s) == pytest.approx(math.hypot(0.05, C.UPPER_ARM_PRIOR) * sw, rel=1e-3)
    # forearm foreshortened: lifted towards the camera, 3D length = forearm prior
    assert w[2] > 0
    assert np.linalg.norm(w - e) == pytest.approx(C.FOREARM_PRIOR * sw, rel=1e-3)
    # fingers keep their observed 2D length (fixture is flat) and never go behind the wrist
    tip = joint(m, 10, "left_index_4")
    assert tip[2] >= w[2] - 1e-9


def test_full_length_segment_has_no_depth_and_longer_clip_length_is_capped():
    x = fixture(forearm_2d=C.FOREARM_PRIOR)
    m = C.engine_poses_to_motion(x)
    w = joint(m, 0, "left_wrist")
    assert w[2] == pytest.approx(0.0, abs=1e-6) or w[2] >= C.TORSO_FRONT_M - 1e-9
    # a clip with implausibly long 2D upper arms cannot push the elbow far forward
    x = fixture()
    x[::2, 13, 1] = -2.0
    m = C.engine_poses_to_motion(x)
    assert max(f["joints"]["left_elbow"][2] for f in m["frames"]) <= 1.2 * C.UPPER_ARM_PRIOR * C.SHOULDER_WIDTH_M + 1e-6


def test_hands_kept_in_front_of_torso_and_face():
    x = fixture(forearm_2d=C.FOREARM_PRIOR)
    x[:, 15] = x[:, C.LEFT_HAND] = [0.1, -0.5]                # wrist in front of the chest, seen without depth
    x[:, 13] = [0.4, -0.5]
    x[:, 16] = x[:, C.RIGHT_HAND] = [0.0, 0.6]                # wrist at the face
    x[:, 14] = [-0.4, 0.0]
    m = C.engine_poses_to_motion(x)
    assert joint(m, 3, "left_wrist")[2] >= C.TORSO_FRONT_M - 1e-9
    assert joint(m, 3, "right_wrist")[2] >= C.FACE_FRONT_M - 1e-9


def test_missing_hand_frames_drop_fingers_and_mark_untracked():
    m = C.engine_poses_to_motion(fixture(hand_nan_frames=(5, 6)))
    f = m["frames"][5]
    assert f["hand_tracking"]["left"]["observed"] is False
    assert "left_index_1" not in f["joints"] and "right_index_1" not in f["joints"]
    assert all(math.isfinite(v) for v in f["joints"]["left_wrist"])          # wrist interpolated from the pose
    assert m["frames"][4]["hand_tracking"]["left"]["observed"] is True
    assert "left_thumb_4" in m["frames"][4]["joints"]


def test_resampling_to_target_fps():
    m = C.engine_poses_to_motion(fixture(T=51), fps_out=12.5)
    assert m["fps"] == 12.5 and len(m["frames"]) == 26 and m["duration_ms"] == 2000
    assert m["frames"][1]["time_ms"] == 80


def test_face_ratios_match_capture_measure():
    m = C.engine_poses_to_motion(fixture(), face_calibration={"brow_left": 0.4, "brow_right": 0.41})
    face = m["frames"][0]["face"]
    assert face["brow_left"] == pytest.approx(0.24 * C.FACE_ASPECT, rel=1e-3)
    assert face["brow_right"] == pytest.approx(0.24 * C.FACE_ASPECT, rel=1e-3)
    assert face["mouth_open"] == pytest.approx(0.03 * C.FACE_ASPECT, rel=1e-3)
    assert {k: m["face_calibration"][k] for k in ("brow_left", "brow_right")} == {"brow_left": 0.4, "brow_right": 0.41}
    assert m["face_calibration"]["mouth_width"] == C.FACE_NEUTRAL_DEFAULT["mouth_width"]   # unspecified channels keep the lexicon neutral


def test_three_dimensional_input_ignores_engine_depth():
    x = fixture()
    x3 = np.concatenate([x, np.full(x.shape[:2] + (1,), 5.0, np.float32)], axis=-1)
    a = C.engine_poses_to_motion(x)
    b = C.engine_poses_to_motion(x3)
    assert a["frames"][7]["joints"] == b["frames"][7]["joints"]


@pytest.mark.parametrize("shape", [(10, 156, 2), (10, 157), (1, 157, 2)])
def test_rejects_bad_shapes(shape):
    with pytest.raises(ValueError):
        C.engine_poses_to_motion(np.zeros(shape, np.float32))
