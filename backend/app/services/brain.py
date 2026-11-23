"""The brain: ONE local LLM for every mode that answers (sign conversation, AR companion, later film understanding).

Ollama on this PC (http://127.0.0.1:11434 by default, SIGNACULTURE_OLLAMA_URL), model SIGNACULTURE_LLM_MODEL (default
gemma3:4b: good French, vision-capable, about 3.3 GB of VRAM). Replies are short (1-2 sentences of simple French) because
they are signed by the avatar right after: concrete words, no lists, no emojis, no markdown.

Contract (POST /brain/respond):
    in   {messages: [{role: "user"|"assistant", text}], image_b64?: str, mode: "conversation"|"companion"}
    out  {text, model, mode, ms}
The image (JPEG/PNG, base64, no data: prefix needed) is attached to the last user message (companion: what the phone sees).

VRAM: the GPU (8 GB) is shared with Whisper (about 1.6 GB) and the signing engines, so the context is kept small
(num_ctx 2048) and the model is unloaded after SIGNACULTURE_LLM_KEEP_ALIVE of inactivity (default 10 min).
Nothing leaves the machine.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import Protocol

DEFAULT_MODEL = "gemma3:4b"
MAX_TURNS = 12              # the last 12 messages are enough for a spoken/signed exchange and keep the prompt small
MAX_REPLY_CHARS = 220       # about two short sentences: the avatar signs it right after

SYSTEM_PROMPTS = {
    "conversation": (
        "Tu es SignaCulture, un compagnon chaleureux qui converse avec une personne sourde ou entendante. "
        "Ta réponse sera traduite en langue des signes française (LSF) et signée par un avatar, puis affichée en texte. "
        "Réponds toujours en français simple, en une ou deux phrases courtes (moins de 25 mots en tout). "
        "Utilise des mots concrets et courants, faciles à signer. Pas de liste, pas d'emoji, pas de mise en forme, "
        "pas de parenthèses. Tutoie ou vouvoie comme la personne. Si la phrase de la personne est courte ou incomplète "
        "(quelques signes reconnus), devine son intention avec bienveillance et réponds naturellement, "
        "éventuellement par une petite question pour continuer l'échange."
    ),
    "companion": (
        "Tu es SignaCulture, un compagnon chaleureux qui accompagne une personne grâce à la caméra de son téléphone. "
        "Une image de ce que voit la caméra peut accompagner le message. "
        "Ta réponse sera signée en langue des signes française (LSF) par un avatar, puis affichée en texte. "
        "Réponds en français simple, en une ou deux phrases courtes (moins de 25 mots en tout), avec des mots concrets "
        "(objets, couleurs, lieux, personnes que tu vois). Pas de liste, pas d'emoji, pas de mise en forme. "
        "Ne décris que ce que tu vois vraiment ; si tu ne sais pas, dis-le simplement."
    ),
}


class BrainUnavailable(RuntimeError):
    """The local LLM cannot answer (Ollama not running, model missing, timeout): the API returns 503."""


class ChatClient(Protocol):
    def chat(self, payload: dict, timeout: float) -> dict: ...


class OllamaClient:
    def __init__(self, url: str | None = None):
        self.url = (url or os.getenv("SIGNACULTURE_OLLAMA_URL", "http://127.0.0.1:11434")).rstrip("/")
        # local server: never through a system proxy
        self.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def _post(self, path: str, payload: dict, timeout: float) -> dict:
        req = urllib.request.Request(self.url + path, data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            with self.opener.open(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = ""
            try:
                detail = json.loads(exc.read().decode("utf-8")).get("error", "")
            except Exception:  # noqa: BLE001
                pass
            if exc.code == 404 or "not found" in detail:
                raise BrainUnavailable(f"Modèle de langage « {payload.get('model')} » absent : lancez « ollama pull {payload.get('model')} ».") from None
            raise BrainUnavailable(f"Le modèle de langage a répondu une erreur ({exc.code}) {detail}".strip()) from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
            raise BrainUnavailable("Modèle de langage local indisponible : démarrez Ollama (« ollama serve »).") from exc

    def chat(self, payload: dict, timeout: float) -> dict:
        return self._post("/api/chat", payload, timeout)

    def status(self, timeout: float = 4.0) -> dict:
        try:
            with self.opener.open(self.url + "/api/tags", timeout=timeout) as r:
                tags = json.loads(r.read().decode("utf-8"))
        except Exception:  # noqa: BLE001
            return {"running": False, "models": []}
        return {"running": True, "models": [m.get("name") for m in tags.get("models", [])]}


def model_name() -> str:
    return os.getenv("SIGNACULTURE_LLM_MODEL", DEFAULT_MODEL)


def keep_alive() -> str:
    return os.getenv("SIGNACULTURE_LLM_KEEP_ALIVE", "10m")


_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF‍️]")


def clean_reply(text: str, max_chars: int = MAX_REPLY_CHARS) -> str:
    """Model output -> text the avatar can sign: no markdown, emojis, list marks or line breaks; at most two sentences."""
    t = _EMOJI.sub("", text or "")
    t = re.sub(r"[*_#`>~]+", "", t)
    t = re.sub(r"^\s*(?:[-•]|\d+[.)])\s+", "", t, flags=re.M)
    t = re.sub(r"\([^)]*\)", "", t)
    t = " ".join(t.split())
    t = re.sub(r"\s+([,.])", r"\1", t).strip(" \"'«»")
    sentences = re.findall(r"[^.!?]+[.!?]*", t)
    out = ""
    for s in sentences[:2]:
        s = s.strip()
        if not s:
            continue
        if out and len(out) + 1 + len(s) > max_chars:
            break
        out = f"{out} {s}".strip()
    if len(out) > max_chars:                               # one very long sentence: cut at the last word
        out = out[:max_chars].rsplit(" ", 1)[0].rstrip(",;:") + "."
    if out and out[-1] not in ".!?":
        out += "."
    return out


def build_payload(messages: list[dict], image_b64: str | None, mode: str) -> dict:
    turns = [m for m in messages if m.get("role") in ("user", "assistant") and (m.get("text") or "").strip()][-MAX_TURNS:]
    while turns and turns[0]["role"] != "user":            # the exchange starts with the person
        turns = turns[1:]
    if not turns or turns[-1]["role"] != "user":
        raise ValueError("Le dernier message doit venir de la personne.")
    chat = [{"role": "system", "content": SYSTEM_PROMPTS[mode]}]
    chat += [{"role": m["role"], "content": m["text"].strip()[:1000]} for m in turns]
    if image_b64:
        chat[-1]["images"] = [image_b64.split(",", 1)[-1] if image_b64.startswith("data:") else image_b64]
    return {
        "model": model_name(),
        "messages": chat,
        "stream": False,
        "keep_alive": keep_alive(),
        # small context and short answers: little KV-cache VRAM next to Whisper and the engines, fast replies
        "options": {"temperature": 0.6, "top_p": 0.9, "num_ctx": 2048, "num_predict": 90},
    }


class Brain:
    def __init__(self, client: ChatClient | None = None):
        self.client = client or OllamaClient()

    def respond(self, messages: list[dict], image_b64: str | None = None, mode: str = "conversation",
                timeout: float = 120.0) -> dict:
        if mode not in SYSTEM_PROMPTS:
            raise ValueError(f"Mode inconnu : {mode}")
        payload = build_payload(messages, image_b64, mode)
        t0 = time.perf_counter()
        out = self.client.chat(payload, timeout)
        text = clean_reply(((out or {}).get("message") or {}).get("content", ""))
        if not text:
            raise BrainUnavailable("Le modèle de langage n'a rien répondu.")
        return {"text": text, "model": payload["model"], "mode": mode, "ms": round((time.perf_counter() - t0) * 1000)}

    def warm(self) -> dict:
        """Load the model in VRAM before the first question (an empty chat loads it without generating)."""
        t0 = time.perf_counter()
        self.client.chat({"model": model_name(), "messages": [], "keep_alive": keep_alive(), "stream": False}, 180.0)
        return {"model": model_name(), "ms": round((time.perf_counter() - t0) * 1000)}

    def status(self) -> dict:
        info = self.client.status() if hasattr(self.client, "status") else {"running": True, "models": []}
        name = model_name()
        have = any(m == name or m == f"{name}:latest" for m in info.get("models", []))
        return {"model": name, "running": info["running"], "available": bool(info["running"] and have)}


brain = Brain()
