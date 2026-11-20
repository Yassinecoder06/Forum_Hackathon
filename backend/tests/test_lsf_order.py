"""Rule-based LSF word order (lsf_order), grammatical non-manual channels and the API wiring.

The expected orders are the simplified textbook rules the module implements, not LSF ground truth."""
import math

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.services import lsf_order as O
from backend.app.services import lsf_pose_convert as C
from backend.tests.test_lsf_pose_convert import fixture

client = TestClient(main.app)


@pytest.mark.parametrize("text, glosses, markers", [
    ("Le musée est fermé demain.", ["DEMAIN", "MUSÉE", "FERMÉ"], []),
    ("Je ne veux pas boire de l'eau.", ["MOI", "EAU", "BOIRE", "VOULOIR", "PAS"], ["negation"]),
    ("Où est la gare ?", ["GARE", "OÙ"], ["question_wh"]),
    ("Est-ce que tu viens demain ?", ["DEMAIN", "TOI", "VENIR"], ["question_yn"]),
    ("Hier, j'ai mangé une pomme.", ["HIER", "MOI", "POMME", "MANGER"], []),
    ("Le gouvernement a présenté un projet de loi.", ["GOUVERNEMENT", "PROJET", "LOI", "PRÉSENTER", "FINI"], []),
])
def test_rules(text, glosses, markers):
    r = O.order(text)
    assert r["glosses"] == glosses
    assert [m for c in r["clauses"] for m in c["markers"]] == markers
    assert r["validated"] is False


def test_time_then_place_then_verb_last():
    r = O.order("La semaine prochaine, le président ira à Marseille.")
    g = r["glosses"]
    assert g.index("SEMAINE") < g.index("MARSEILLE") < g.index("PRÉSIDENT") < g.index("ALLER")


def test_nonmanual_channels_ramp_and_head_shake():
    spans = [{"start_ms": 1000, "end_ms": 2000, "type": "negation"}, {"start_ms": 3000, "end_ms": 4000, "type": "question_yn"}]
    assert C.nonmanual_channels(0, spans) == {"nm_brow_raise": 0, "nm_brow_furrow": 0, "nm_head_yaw": 0, "nm_head_pitch": 0}
    yaws = [C.nonmanual_channels(t, spans)["nm_head_yaw"] for t in range(1100, 1900, 20)]
    assert max(yaws) > 0.1 and min(yaws) < -0.1                                  # the head shakes both ways
    mid = C.nonmanual_channels(3500, spans)
    assert mid["nm_brow_raise"] == pytest.approx(0.65) and mid["nm_head_pitch"] > 0
    assert 0 < C.nonmanual_channels(3000 - C.RAMP_MS / 2, spans)["nm_brow_raise"] < 0.65   # smooth ramp, no jump


def test_converter_bakes_markers_and_new_face_channels():
    spans = [{"start_ms": 200, "end_ms": 1500, "type": "question_wh"}]
    m = C.engine_poses_to_motion(fixture(T=50), nonmanual=spans)
    f = m["frames"][20]["face"]
    assert f["nm_brow_furrow"] > 0.5 and math.isfinite(f["mouth_width"]) and math.isfinite(f["eye_open_left"])
    assert m["nonmanual"] == spans
    assert "nm_brow_furrow" not in C.engine_poses_to_motion(fixture(T=50))["frames"][20]["face"]


def test_order_endpoint():
    r = client.post("/lsf/order", json={"text": "Pourquoi tu ne viens jamais au cinéma ?"})
    assert r.status_code == 200
    body = r.json()
    assert body["glosses"][-1] == "POURQUOI" and "JAMAIS" in body["glosses"]
    assert body["clauses"][0]["markers"] == ["negation", "question_wh"]


def test_generate_passes_lsf_groups_and_marks_clauses(monkeypatch):
    seen = {}

    def fake_generate(text, engine, seed=0, groups=None):
        seen["groups"] = groups
        meta = {"engine": engine, "lemmas": ["demain", "president"], "unknown": [],
                "sign_spans": [{"start_frame": 5, "end_frame": 20, "lemma": "demain", "clause": 0},
                               {"start_frame": 25, "end_frame": 40, "lemma": "president", "clause": 0}]}
        return fixture(T=50), meta

    monkeypatch.setattr(main.lsf_engines, "generate", fake_generate)
    r = client.post("/lsf/generate", json={"text": "Le président ne viendra pas demain.", "engine": "b2", "order": "lsf"})
    assert r.status_code == 200
    m = r.json()
    assert seen["groups"] == [O.order("Le président ne viendra pas demain.")["clauses"][0]["glosses"]]
    assert m["nonmanual"] == [{"start_ms": 200, "end_ms": 1600, "type": "negation"}]       # first to last sign of the clause
    assert m["generation"]["order"] == "lsf" and "sign_spans" not in m["generation"]
    # an edited gloss line replaces the automatic order; French order sends no groups
    client.post("/lsf/generate", json={"text": "Je visite Paris.", "engine": "b2", "glosses": [["PARIS", "MOI", "VISITER"]]})
    assert seen["groups"] == [["PARIS", "MOI", "VISITER"]]
    client.post("/lsf/generate", json={"text": "Je visite Paris.", "engine": "b2", "order": "french"})
    assert seen["groups"] is None
    # markers can be switched off
    r = client.post("/lsf/generate", json={"text": "Tu viens ?", "engine": "c0", "nonmanual": False})
    assert "nonmanual" not in r.json()
