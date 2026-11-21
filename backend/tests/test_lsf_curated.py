"""Curated C0 takes: the checked demo sample is returned for its sentence, live generation for anything else."""
import json

import numpy as np

from backend.app.engines import lsf


class FakeC0:
    def __init__(self):
        self.calls = []

    def generate(self, text, seed=0):
        self.calls.append(text)
        return np.zeros((10, 157, 2), np.float32), {"tokens": 3}


def test_curated_take_wins_for_its_sentence_only(monkeypatch, tmp_path):
    take = np.full((25, 157, 2), 0.5, np.float32)
    (tmp_path / "c0").mkdir()
    np.savez(tmp_path / "c0" / "curated.npz", texts=json.dumps(["Je vous dis à la semaine prochaine."]), poses_0=take)
    monkeypatch.setenv("SIGNACULTURE_LSF_MODEL_DIR", str(tmp_path))
    monkeypatch.setattr(lsf, "_curated", None)
    model = FakeC0()
    poses, meta = lsf._generate(model, "  je vous dis à la semaine prochaine ", "c0", 0, 0.0)
    assert meta["curated"] and np.array_equal(poses, take) and not model.calls
    poses, meta = lsf._generate(model, "Je vous dis à demain.", "c0", 0, 0.0)
    assert "curated" not in meta and model.calls == ["Je vous dis à demain."]
    lsf._generate(model, "Je vous dis à la semaine prochaine.", "c0", 7, 0.0)      # another seed: live sampling
    assert len(model.calls) == 2


def test_text_key_ignores_case_spacing_apostrophes_and_final_punctuation():
    assert lsf.text_key("Qu’est-ce qui a changé  aujourd'hui ?") == lsf.text_key("qu'est-ce qui a changé aujourd'hui")
