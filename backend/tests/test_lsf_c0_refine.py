"""Optional C0 refiners: random-weight models only (no private weights, no interpreter data)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="torch not installed")

from backend.app.engines.lsf import c0, config  # noqa: E402
from backend.app.engines.lsf import c0_refine as R  # noqa: E402


def fake_refiner_file(path, d_x, d_c, mode):
    torch.manual_seed(d_x)
    net = R.Net(d_x, d_c)
    torch.save({"net": net.state_dict(), "mu": torch.zeros(d_x), "sd": torch.full((d_x,), 0.01), "mode": mode}, path)
    return path


def bare_engine():
    """A C0Engine with random tokenizer weights (down 2, 2 levels like production) and no text model."""
    torch.manual_seed(0)
    eng = c0.C0Engine.__new__(c0.C0Engine)
    eng.dev, eng.down, eng.levels, eng.refiner = "cpu", 2, 2, None
    eng.models = {"body": c0.VQVAE(len(c0.BODY) * 2, down=2, levels=2).eval(), "hand": c0.VQVAE(84, down=2, levels=2).eval()}
    for m in eng.models.values():
        for vq in m.rvq.vqs:
            vq.emb.normal_(0, 1)
    return eng


def tokens(n=12, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, c0.CB, (n, 2)), rng.integers(0, c0.CB, (n, 2)), rng.random(n).astype("float32")


def test_refiner_files_are_optional(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_LSF_MODEL_DIR", str(tmp_path))
    for key in ("c0_handdec", "c0_facedec"):
        monkeypatch.delenv(config.FILES[key][0], raising=False)
        assert config.optional(key) is None
        assert config.path(key) == tmp_path / config.FILES[key][1]
    assert "c0_handdec" not in config.ENGINE_FILES["c0"]          # never required for C0
    assert bare_engine()._load_refiner("cpu") is None


def test_without_refiner_decode_is_the_tokenizer_decode(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_LSF_MODEL_DIR", str(tmp_path))
    eng = bare_engine()
    tb, th, hold = tokens()
    x = eng.decode(tb, th, None, 0)
    assert x.shape == (2 * 2 * len(tb) - 1, 157, 2)                 # 2 frames per token at 12.5 fps -> 25 fps
    # same as the refiner's own tokenizer path (x/y), i.e. the refiners start from exactly this decode
    hand = fake_refiner_file(tmp_path / "h.pt", 126, 224, "reg")
    ref = R.C0Refiner(str(hand), dev="cpu")
    for p in ref.hand[0].parameters():
        torch.nn.init.zeros_(p)
    y = ref.decode(eng.models, tb, th, 0)
    assert np.allclose(y[..., :2], x, atol=1e-5)


def test_refiner_gives_hand_depth_and_holds_still_apply(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_LSF_C0_HANDDEC", str(fake_refiner_file(tmp_path / "h.pt", 126, 224, "reg")))
    monkeypatch.setenv("SIGNACULTURE_LSF_C0_FACEDEC", str(fake_refiner_file(tmp_path / "f.pt", 184, 248, "fm")))
    eng = bare_engine()
    eng.refiner = eng._load_refiner("cpu")
    assert eng.refiner is not None and eng.refiner.face is not None
    assert eng.refiner_files == {"hand": "h.pt", "face": "f.pt"}
    tb, th, hold = tokens()
    x = eng.decode(tb, th, None, 3)
    assert x.shape == (4 * len(tb) - 1, 157, 3)
    hands = np.r_[23:65]
    assert np.abs(x[:, hands, 2]).max() > 0                          # learned finger depth on the hands
    assert np.abs(np.delete(x[..., 2], hands, axis=1)).max() == 0     # nothing else gets depth
    assert np.array_equal(eng.decode(tb, th, None, 3), x)             # seeded: same take twice
    held = eng.decode(tb, th, hold, 3)
    assert np.array_equal(held, c0.apply_holds(x, hold, np.random.default_rng(3)))


def test_face_file_alone_does_not_enable_refiners(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_LSF_MODEL_DIR", str(tmp_path))
    monkeypatch.delenv("SIGNACULTURE_LSF_C0_HANDDEC", raising=False)
    monkeypatch.setenv("SIGNACULTURE_LSF_C0_FACEDEC", str(fake_refiner_file(tmp_path / "f.pt", 184, 248, "fm")))
    assert bare_engine()._load_refiner("cpu") is None
