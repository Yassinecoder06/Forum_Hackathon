"""Webcam sign recognition on a synthetic lexicon (no private data): preparation, segmentation, DTW ranking, mirror
(left-handed signer), the endpoint's two input formats and its errors."""
import base64

import numpy as np
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.services import sign_recognition as R

client = TestClient(main.app)
P = R.TOTAL_POINTS


def rest_pose():
    x = np.zeros((P, 3), "float32")
    x[11], x[12] = (0.5, 0, 0), (-0.5, 0, 0)
    x[13], x[14] = (0.6, -0.6, 0), (-0.6, -0.6, 0)
    x[15], x[16] = (0.3, R.REST_Y, 0), (-0.3, R.REST_Y, 0)
    x[0] = (0, 0.5, 0)
    for o, w in ((R.LH, 15), (R.RH, 16)):
        x[o:o + 21] = x[w] + hand_shape(0)
    x[65:] = (0, 0.6, 0)
    return x


def hand_shape(kind):
    """21 hand points relative to the wrist: an open hand (0), a fist (1) or a pointing index (2)."""
    h = np.zeros((21, 3), "float32")
    for f in range(5):
        for j in range(4):
            reach = 0.06 * (j + 1)
            if kind == 1 or (kind == 2 and f != 1):
                reach = 0.05 + 0.01 * j
            h[1 + 4 * f + j] = ((f - 2) * 0.035, reach, 0)
    return h


TRAJ = {
    "bonjour": lambda t: (-0.35 - 0.3 * t, 0.55 + 0.1 * np.sin(np.pi * t)),       # from the forehead outwards
    "merci": lambda t: (-0.1, 0.4 - 0.9 * t),                                        # from the chin down
    "maison": lambda t: (-0.4 + 0.4 * t, 0.2 + 0.3 * np.sin(np.pi * t)),            # roof shape
    "ecole": lambda t: (-0.3 + 0.15 * np.sin(4 * np.pi * t), -0.2),                  # small taps
    "voiture": lambda t: (-0.4 + 0.2 * np.sin(2 * np.pi * t), -0.1 + 0.2 * np.cos(2 * np.pi * t)),
}
SHAPE = {"bonjour": 0, "merci": 0, "maison": 2, "ecole": 1, "voiture": 1}


def sign_clip(word, n=20, rest=None):
    rest = rest_pose() if rest is None else rest
    clip = np.repeat(rest[None], n, 0)
    for i, t in enumerate(np.linspace(0, 1, n)):
        x, y = TRAJ[word](t)
        clip[i, 16] = (x, y, 0)
        clip[i, R.RH:R.RH + 21] = clip[i, 16] + hand_shape(SHAPE[word])
    return clip


def templates():
    words = list(TRAJ)
    return R.Templates(words, [sign_clip(w) for w in words], ["corpus"] * len(words), rest_pose())


def with_rest(clips, hold=12, blend=8):
    """Signs joined by lowering the hands to rest (as a person signing word by word)."""
    rest = rest_pose()
    parts = [np.repeat(rest[None], hold, 0)]
    for c in clips:
        parts.append(np.stack([rest * (1 - w) + c[0] * w for w in np.linspace(0, 1, blend)]))
        parts.append(c)
        parts.append(np.stack([c[-1] * (1 - w) + rest * w for w in np.linspace(0, 1, blend)]))
        parts.append(np.repeat(rest[None], hold, 0))
    return np.concatenate(parts)


def to_raw(x, s=0.22, c=(0.5, 0.45), seed=0, noise=0.002):
    """Normalized clip -> image coordinates (x, y down) as the browser sends them, with a little tracking noise."""
    r = x.copy()
    r[..., 1] *= -1
    r[..., :2] = r[..., :2] * s + np.array(c)
    r[..., 2] *= s
    return r + np.random.default_rng(seed).normal(0, noise, r.shape).astype("float32")


def test_prepare_normalizes_like_the_corpus():
    raw = to_raw(with_rest([sign_clip("merci", 30)]), noise=0)
    x, pres = R.prepare(raw, rest_pose())
    assert np.allclose(x[:, 11, :2].mean(0), (0.5, 0), atol=0.02) and np.allclose(x[:, 12, :2].mean(0), (-0.5, 0), atol=0.02)
    assert pres.all()


def test_isolated_sign_is_recognised_first():
    rec = R.Recognizer(templates())
    for i, w in enumerate(TRAJ):
        out = rec.recognize(to_raw(with_rest([sign_clip(w, 26)]), seed=i))
        assert len(out["segments"]) == 1, w
        assert out["glosses"][0]["gloss"] == w
        assert out["segments"][0]["candidates"][0]["score"] > 0.4


def test_sequence_is_segmented_into_signs():
    rec = R.Recognizer(templates())
    out = rec.recognize(to_raw(with_rest([sign_clip("bonjour", 24), sign_clip("maison", 28), sign_clip("merci", 22)])))
    assert [g["gloss"] for g in out["glosses"]] == ["bonjour", "maison", "merci"]
    assert out["text_guess"] == "Bonjour maison merci"
    assert 0 < out["confidence"] <= 1
    assert out["segments"][0]["end_ms"] <= out["segments"][1]["start_ms"]


def test_left_handed_signer_matches_mirrored():
    rec = R.Recognizer(templates())
    out = rec.recognize(to_raw(R.mirror(with_rest([sign_clip("maison", 26)]))))
    assert out["glosses"][0]["gloss"] == "maison"


def test_missing_hands_and_irregular_timestamps():
    raw = to_raw(with_rest([sign_clip("voiture", 30)]))
    raw[::7, R.LH:R.RH + 21] = np.nan                       # hands lost now and then
    raw[:, R.LH:R.LH + 21] = np.nan                          # the left hand never seen (one-handed sign)
    t = np.cumsum(np.random.default_rng(1).uniform(28, 45, len(raw)))
    out = R.Recognizer(templates()).recognize(raw, t_ms=t)
    assert out["glosses"][0]["gloss"] == "voiture"


def test_no_signing_gives_no_gloss():
    raw = to_raw(np.repeat(rest_pose()[None], 40, 0))
    out = R.Recognizer(templates()).recognize(raw)
    assert out["glosses"] == [] and out["text_guess"] == "" and out["confidence"] == 0


def test_endpoint_accepts_base64_and_json(monkeypatch):
    monkeypatch.setattr(R, "recognizer", R.Recognizer(templates()))
    raw = to_raw(with_rest([sign_clip("ecole", 26)]))
    b64 = base64.b64encode(raw.astype("<f4").tobytes()).decode()
    r = client.post("/sign/recognize", json={"data_b64": b64, "fps": 25})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["glosses"][0]["gloss"] == "ecole" and body["vocabulary"] == 5
    assert len(body["segments"][0]["candidates"]) == 5
    frames = [[[None if np.isnan(v) else float(v) for v in p] for p in f] for f in raw[:, :, :]]
    frames[3][R.RH] = [None, None, None]
    r = client.post("/sign/recognize", json={"frames": frames, "top_k": 3})
    assert r.status_code == 200 and r.json()["glosses"][0]["gloss"] == "ecole"
    assert len(r.json()["segments"][0]["candidates"]) == 3


def test_endpoint_errors(monkeypatch):
    monkeypatch.setattr(R, "recognizer", R.Recognizer(templates()))
    assert client.post("/sign/recognize", json={}).status_code == 422
    bad = base64.b64encode(np.zeros(100, "<f4").tobytes()).decode()
    assert client.post("/sign/recognize", json={"data_b64": bad}).status_code == 422
    short = base64.b64encode(np.zeros((3, P, 3), "<f4").tobytes()).decode()
    assert client.post("/sign/recognize", json={"data_b64": short}).status_code == 422
    no_shoulders = np.full((20, P, 3), np.nan, "<f4")
    r = client.post("/sign/recognize", json={"data_b64": base64.b64encode(no_shoulders.tobytes()).decode()})
    assert r.status_code == 422 and "paules" in r.json()["detail"]


def test_missing_lexicon_gives_503(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_SIGN_TEMPLATES", str(tmp_path / "absent.npz"))
    monkeypatch.setattr(R, "recognizer", R.Recognizer())
    raw = to_raw(with_rest([sign_clip("merci", 20)]))
    r = client.post("/sign/recognize", json={"data_b64": base64.b64encode(raw.astype("<f4").tobytes()).decode()})
    assert r.status_code == 503 and "Lexique" in r.json()["detail"]
    assert R.available() is False


def test_templates_load_from_npz(tmp_path):
    words = list(TRAJ)
    clips = [sign_clip(w) for w in words]
    offs = np.cumsum([0] + [len(c) for c in clips])
    path = tmp_path / "lex.npz"
    np.savez(path, names=np.array(words), frames=np.concatenate(clips).astype("float16"), offsets=offs,
             signers=np.array(["corpus"] * len(words)), rest=rest_pose(), display=np.array(["Bonjour", "Merci", "Maison", "École", "Voiture"]))
    tp = R.Templates.from_npz(path)
    assert len(tp) == 5 and tp.label("ecole") == "École"
    out = R.Recognizer(tp).recognize(to_raw(with_rest([sign_clip("ecole", 24)])))
    assert out["glosses"][0]["label"] == "École" and out["text_guess"] == "École"


def test_vocabulary_restricts_the_candidates(monkeypatch):
    monkeypatch.setattr(R, "recognizer", R.Recognizer(templates()))
    raw = to_raw(with_rest([sign_clip("ecole", 26)]))
    b64 = base64.b64encode(raw.astype("<f4").tobytes()).decode()
    r = client.post("/sign/recognize", json={"data_b64": b64, "vocabulary": ["merci", "maison"]})
    body = r.json()
    assert r.status_code == 200 and body["vocabulary"] == 2
    assert {c["gloss"] for c in body["segments"][0]["candidates"]} == {"merci", "maison"}
    r = client.post("/sign/recognize", json={"data_b64": b64, "vocabulary": "conversation"})
    assert r.status_code == 200 and r.json()["glosses"][0]["gloss"] == "ecole"     # ecole is an everyday word


def test_personal_sign_is_learned_and_used(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_SIGN_USER_TEMPLATES", str(tmp_path / "user.npz"))
    words = ["bonjour", "merci"]
    rec = R.Recognizer(R.Templates(words, [sign_clip(w) for w in words], ["corpus"] * 2, rest_pose()))
    monkeypatch.setattr(R, "recognizer", rec)
    # the person's own sign for "maison" (a word the lexicon above lacks)
    take = base64.b64encode(to_raw(with_rest([sign_clip("maison", 26)])).astype("<f4").tobytes()).decode()
    r = client.post("/sign/templates", json={"gloss": "Maison", "data_b64": take})
    assert r.status_code == 200 and r.json() == {"gloss": "maison", "label": "Maison", "count": 1}
    again = base64.b64encode(to_raw(with_rest([sign_clip("maison", 30)]), seed=4).astype("<f4").tobytes()).decode()
    out = client.post("/sign/recognize", json={"data_b64": again, "vocabulary": ["bonjour"]}).json()
    assert out["glosses"][0]["gloss"] == "maison"                  # personal signs always count
    assert R.user_store().summary() == [{"gloss": "maison", "label": "Maison", "count": 1}]
    assert client.delete("/sign/templates/maison").json() == {"removed": 1}
    assert R.user_store().summary() == []


def test_learning_needs_a_sign(monkeypatch, tmp_path):
    monkeypatch.setenv("SIGNACULTURE_SIGN_USER_TEMPLATES", str(tmp_path / "user.npz"))
    monkeypatch.setattr(R, "recognizer", R.Recognizer(templates()))
    still = base64.b64encode(to_raw(np.repeat(rest_pose()[None], 30, 0)).astype("<f4").tobytes()).decode()
    r = client.post("/sign/templates", json={"gloss": "maison", "data_b64": still})
    assert r.status_code == 422 and "Aucun signe" in r.json()["detail"]
