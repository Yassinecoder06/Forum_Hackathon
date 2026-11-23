"""The brain (one local LLM for conversation and the AR companion): payload, reply clean-up, errors; Ollama is faked."""
from fastapi.testclient import TestClient

from backend.app import main
from backend.app.services import brain as B

client = TestClient(main.app)


class FakeOllama:
    def __init__(self, reply="Bonjour ! Je vais bien, merci. Et toi, comment vas-tu aujourd'hui ? Moi j'aime le soleil.",
                 fail=None, models=("gemma3:4b",)):
        self.reply, self.fail, self.models, self.calls = reply, fail, list(models), []

    def chat(self, payload, timeout):
        self.calls.append(payload)
        if self.fail:
            raise self.fail
        return {"message": {"role": "assistant", "content": self.reply}}

    def status(self):
        return {"running": self.fail is None, "models": self.models}


def use(monkeypatch, fake):
    monkeypatch.setattr(B, "brain", B.Brain(fake))
    return fake


def test_reply_is_short_signable_french(monkeypatch):
    fake = use(monkeypatch, FakeOllama())
    r = client.post("/brain/respond", json={"messages": [{"role": "user", "text": "Bonjour, ça va ?"}]})
    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "Bonjour ! Je vais bien, merci."            # at most two sentences
    assert body["mode"] == "conversation" and body["model"] == "gemma3:4b"
    sent = fake.calls[0]
    assert sent["messages"][0]["role"] == "system" and "LSF" in sent["messages"][0]["content"]
    assert sent["messages"][-1] == {"role": "user", "content": "Bonjour, ça va ?"}
    assert sent["stream"] is False and sent["keep_alive"] and sent["options"]["num_ctx"] <= 4096


def test_model_and_keep_alive_from_environment(monkeypatch):
    monkeypatch.setenv("SIGNACULTURE_LLM_MODEL", "qwen2.5vl:7b")
    monkeypatch.setenv("SIGNACULTURE_LLM_KEEP_ALIVE", "2m")
    fake = use(monkeypatch, FakeOllama())
    client.post("/brain/respond", json={"messages": [{"role": "user", "text": "Salut"}]})
    assert fake.calls[0]["model"] == "qwen2.5vl:7b" and fake.calls[0]["keep_alive"] == "2m"


def test_history_and_companion_image(monkeypatch):
    fake = use(monkeypatch, FakeOllama(reply="Je vois une tasse rouge sur la table."))
    msgs = [{"role": "assistant", "text": "Bienvenue."}, {"role": "user", "text": "Bonjour"},
            {"role": "assistant", "text": "Bonjour !"}, {"role": "user", "text": "Que vois-tu ?"}]
    r = client.post("/brain/respond", json={"messages": msgs, "mode": "companion", "image_b64": "data:image/jpeg;base64,QUJD"})
    assert r.status_code == 200 and r.json()["text"] == "Je vois une tasse rouge sur la table."
    sent = fake.calls[0]["messages"]
    assert "caméra" in sent[0]["content"]
    assert [m["role"] for m in sent[1:]] == ["user", "assistant", "user"]       # starts with the person
    assert sent[-1]["images"] == ["QUJD"]                                         # data: prefix removed


def test_ollama_down_gives_clear_503(monkeypatch):
    use(monkeypatch, FakeOllama(fail=B.BrainUnavailable("Modèle de langage local indisponible : démarrez Ollama")))
    r = client.post("/brain/respond", json={"messages": [{"role": "user", "text": "Bonjour"}]})
    assert r.status_code == 503 and "Ollama" in r.json()["detail"]


def test_real_client_unreachable_server_raises_brain_unavailable():
    c = B.OllamaClient("http://127.0.0.1:9")                                       # nothing listens there
    try:
        c.chat({"model": "gemma3:4b", "messages": []}, timeout=2)
    except B.BrainUnavailable as exc:
        assert "Ollama" in str(exc)
    else:
        raise AssertionError("expected BrainUnavailable")
    assert c.status()["running"] is False


def test_last_message_must_be_the_person(monkeypatch):
    use(monkeypatch, FakeOllama())
    r = client.post("/brain/respond", json={"messages": [{"role": "user", "text": "a"}, {"role": "assistant", "text": "b"}]})
    assert r.status_code == 422


def test_clean_reply_removes_lists_emojis_markdown():
    assert B.clean_reply("**Oui** 😊 !\n- le chat\n- le chien") == "Oui ! le chat le chien."
    assert B.clean_reply("Bonjour (en LSF) à toi") == "Bonjour à toi."
    long = "mot " * 120
    assert len(B.clean_reply(long)) <= B.MAX_REPLY_CHARS + 1


def test_capabilities_report_the_brain(monkeypatch):
    use(monkeypatch, FakeOllama())
    caps = client.get("/capabilities").json()
    assert caps["brain"] == {"model": "gemma3:4b", "running": True, "available": True}
    assert "sign_recognition" in caps
