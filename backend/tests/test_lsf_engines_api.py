"""POST /lsf/generate and MetaHuman avatar selection, without the private model files."""
import importlib.util

import numpy as np
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.engines.lsf import config
from backend.tests.test_lsf_pose_convert import fixture

client = TestClient(main.app)


def test_missing_model_files_give_clear_503(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_LSF_MODEL_DIR", str(tmp_path))
    for env, _ in config.FILES.values():
        monkeypatch.delenv(env, raising=False)
    for engine in ("c0", "b2"):
        r = client.post("/lsf/generate", json={"text": "Bonjour à tous", "engine": engine})
        assert r.status_code == 503
        detail = r.json()["detail"]
        if importlib.util.find_spec("torch") is None:      # CI installs no PyTorch: that is reported first
            assert "PyTorch" in detail
        else:
            assert engine.upper() in detail and "LSF_ENGINES.md" in detail
    caps = client.get("/capabilities").json()
    assert caps["lsf_engines"] == {"c0": False, "b2": False}


def test_engine_paths_from_individual_env_vars(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_LSF_MODEL_DIR", str(tmp_path / "none"))
    for key, (env, _) in config.FILES.items():
        f = tmp_path / f"{key}.bin"
        f.write_bytes(b"x")
        monkeypatch.setenv(env, str(f))
    assert config.status() == {"c0": {"available": True, "missing": []}, "b2": {"available": True, "missing": []}}


def test_request_validation():
    assert client.post("/lsf/generate", json={"text": "bonjour", "engine": "gpt"}).status_code == 422
    assert client.post("/lsf/generate", json={"text": "", "engine": "b2"}).status_code == 422
    assert client.post("/lsf/generate", json={"text": "   ", "engine": "b2"}).status_code == 422
    assert client.post("/lsf/generate", json={"text": "x" * 401, "engine": "b2"}).status_code == 422


def test_generate_returns_motion_file(monkeypatch):
    calls = []

    def fake_generate(text, engine, seed=0, groups=None):
        calls.append((text, engine, seed))
        return fixture(T=30), {"engine": engine, "lemmas": ["bonjour"], "unknown": [], "face_neutral": {"brow_left": 0.43, "brow_right": 0.42}}

    monkeypatch.setattr(main.lsf_engines, "generate", fake_generate)
    r = client.post("/lsf/generate", json={"text": "  Bonjour   à tous ", "engine": "b2", "seed": 3})
    assert r.status_code == 200
    m = r.json()
    assert calls == [("Bonjour à tous", "b2", 3)]
    assert m["rig_version"] == "signaculture-v1" and m["fps"] == 25
    assert m["frames"][-1]["time_ms"] == m["duration_ms"]
    assert set(m["joints"]).issubset(m["frames"][0]["joints"])
    assert {k: m["face_calibration"][k] for k in ("brow_left", "brow_right")} == {"brow_left": 0.43, "brow_right": 0.42}
    assert set(m["face_calibration"]) >= {"mouth_width", "mouth_corner", "eye_open_left", "eye_open_right"}
    assert m["generation"]["translation_validated"] is False
    assert "face_neutral" not in m["generation"]
    assert np.isclose(np.linalg.norm(np.subtract(m["frames"][0]["joints"]["left_shoulder"], m["frames"][0]["joints"]["right_shoulder"])), 0.36)


def test_metahuman_avatar_is_opt_in(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_AVATAR_DIR", str(tmp_path))
    (tmp_path / "MetaHuman.glb").write_bytes(b"glTF")
    monkeypatch.delenv("SIGNACULTURE_AVATAR", raising=False)
    assert client.get("/avatar/model").status_code == 503          # not chosen by default
    monkeypatch.setenv("SIGNACULTURE_AVATAR", "MetaHuman")
    r = client.get("/avatar/model")
    assert r.status_code == 200 and r.content == b"glTF"
    assert client.get("/capabilities").json()["avatar"] == "MetaHuman"
