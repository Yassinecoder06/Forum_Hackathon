"""C0 hold smoothing (engine side) and the eye-corner joints of the converter. Authored fixtures, no interpreter data."""
import numpy as np
import pytest

from backend.app.services import lsf_pose_convert as C
from backend.tests.test_lsf_pose_convert import fixture

c0 = pytest.importorskip("backend.app.engines.lsf.c0", reason="torch not installed")


def wander(T=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.cumsum(rng.normal(0, 0.05, (T, C.N_POINTS, 2)), 0).astype("float32")


def test_holds_still_the_marked_steps_without_jumps():
    x = wander()
    hold = np.array([0, 1, 1, 0, 0, 0, 1, 1, 1, 0], dtype="float32")         # 10 steps x 4 frames
    y = c0.apply_holds(x, hold, np.random.default_rng(0), scale=1.0)
    assert np.allclose(y[3], x[3]) and np.allclose(y[12], x[12])           # anchors kept: no jump in or out
    speed = lambda z, a, b: np.abs(np.diff(z[a:b], axis=0)).mean()          # noqa: E731
    assert speed(y, 3, 13) < 0.6 * speed(x, 3, 13)                         # the run is one smooth move, not wandering
    assert np.allclose(y[:4], x[:4]) and np.allclose(y[12:24], x[12:24])   # free steps untouched


def test_no_holds_is_identity_and_scale_increases_holds():
    x = wander()
    assert np.array_equal(c0.apply_holds(x, np.zeros(10, "float32"), np.random.default_rng(0)), x)
    p = np.full(10, 0.3, "float32")
    moved = lambda s: np.abs(np.diff(c0.apply_holds(x, p, np.random.default_rng(1), scale=s), axis=0)).mean()  # noqa: E731
    assert moved(2.0) < moved(1.0)


def test_converter_adds_eye_corners():
    m = C.engine_poses_to_motion(fixture())
    j = m["frames"][0]["joints"]
    assert j["left_eye"][0] > j["right_eye"][0]                              # +x = signer's left
    assert j["left_eye"][2] == pytest.approx(C.HEAD_Z_M)
