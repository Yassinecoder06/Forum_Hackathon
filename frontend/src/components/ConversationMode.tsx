/// <reference types="vite/client" />
import { useEffect, useRef, useState } from "react";
import { BookOpen, Camera, CameraOff, Check, Hand, Keyboard, Loader2, Mic, MicOff, Play, RotateCcw, Send, Square, Trash2, Video } from "lucide-react";
import { request } from "../services/api";
import type { MotionFile } from "../types/contract";
import { AvatarCanvas } from "./AvatarPlayer/AvatarCanvas";
import { generateBody } from "../utils/lsfRequest";
import { CAPTURE_WORKLET, PhraseDetector, encodeWav } from "../utils/phraseDetector";
import { BODY_EDGES, HAND_EDGES, HandPause, encodeTake, type Landmark } from "../utils/signCapture";
import type { SignTracker } from "../utils/signTracker";
import { choiceText, historyMessages, type Recognition, type Turn, type Via, type Vocabulary } from "../utils/conversation";

const WATERMARK = "SignaCulture · LSF";
const MAX_TAKE_MS = 20000;
const VIA_LABEL: Record<Via, string> = { signes: "en signes", voix: "à la voix", texte: "par écrit" };
const json = (body: unknown, timeout = 60000) => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: AbortSignal.timeout(timeout),
});

type CameraState = "off" | "loading" | "ready" | "recording" | "recognizing";
type Take = { points: Float32Array[]; t: number[]; pause: HandPause; started: number; learn: string | null };
type Stage = { motion: MotionFile; text: string; started: number; turnId?: number };

/**
 * Conversation mode: the person signs in front of the webcam (or speaks, or types), the system understands it, the
 * brain (one local LLM, POST /brain/respond) answers, and the answer comes back as LSF signed by the avatar
 * (POST /lsf/generate: C0, LSF order, grammatical face) and as text. Signing: MediaPipe in the browser -> 157-point
 * corpus layout -> POST /sign/recognize (lexicon matching) -> "Vous avez signé : … ?" to confirm, pick or correct.
 * "Signes à connaître": the avatar shows a lexicon sign (B2 clip) and the person can record their own version of it.
 */
export function ConversationMode({ avatarCredit, transcription, signRecognition, brainReady }: {
  avatarCredit?: string; transcription?: boolean; signRecognition?: boolean; brainReady?: boolean;
}) {
  const [turns, setTurns] = useState<Turn[]>([]),
    [camera, setCamera] = useState<CameraState>("off"),
    [recognition, setRecognition] = useState<Recognition | null>(null),
    [choice, setChoice] = useState<number[]>([]),
    [draft, setDraft] = useState(""),
    [edited, setEdited] = useState(false),
    [typed, setTyped] = useState(""),
    [micOn, setMicOn] = useState(false),
    [micReady, setMicReady] = useState<"loading" | "ready" | "off">(transcription ? "loading" : "off"),
    [hint, setHint] = useState(""),
    [error, setError] = useState(""),
    [stage, setStage] = useState<Stage | null>(null),
    [now, setNow] = useState(0),
    [takeMs, setTakeMs] = useState(0),
    [vocab, setVocab] = useState<Vocabulary | null>(null),
    [vocabMode, setVocabMode] = useState<"conversation" | "all">("conversation"),
    [word, setWord] = useState<string | null>(null),
    [learning, setLearning] = useState(false),
    [notice, setNotice] = useState("");
  const video = useRef<HTMLVideoElement>(null),
    overlay = useRef<HTMLCanvasElement>(null),
    tracker = useRef<SignTracker | null>(null),
    stream = useRef<MediaStream | null>(null),
    take = useRef<Take | null>(null),
    loop = useRef(0),
    chain = useRef<Promise<void>>(Promise.resolve()),
    turnsRef = useRef<Turn[]>([]),
    vocabRef = useRef(vocabMode),
    stopMic = useRef<() => void>(() => {}),
    detector = useRef<PhraseDetector | null>(null);
  turnsRef.current = turns;
  vocabRef.current = vocabMode;

  // Clock for the avatar; load the brain and the signing engine while the person gets ready; release devices on leave.
  useEffect(() => {
    let id = 0;
    const tick = () => { setNow(performance.now()); id = requestAnimationFrame(tick); };
    id = requestAnimationFrame(tick);
    request("/brain/warmup", { method: "POST", signal: AbortSignal.timeout(200000) }).catch(() => {});
    if (!transcription) request("/lsf/generate", json(generateBody("Bonjour.", "c0", true, true), 280000)).catch(() => {});
    loadVocabulary();
    return () => { cancelAnimationFrame(id); closeCamera(); stopMic.current(); };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!transcription) return;
    request("/live/warmup?engine=c0", { method: "POST", signal: AbortSignal.timeout(300000) }).then(() => setMicReady("ready")).catch(() => setMicReady("off"));
  }, [transcription]);

  function loadVocabulary() {
    request<Vocabulary>("/sign/vocabulary").then(setVocab).catch(() => setVocab(null));
  }

  const patch = (id: number, p: Partial<Turn>) => setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...p } : t)));

  /** One exchange: the person's text -> brain -> reply text -> LSF signing. One at a time, in order. */
  function send(text: string, via: Via, recognitionMs?: number) {
    const clean = text.trim();
    if (!clean) return;
    setError("");
    const userId = Date.now() + Math.random(), botId = userId + 0.5;
    const user: Turn = { id: userId, role: "user", text: clean, via, ms: recognitionMs !== undefined ? { recognition: recognitionMs } : undefined };
    setTurns((prev) => [...prev, user, { id: botId, role: "assistant", text: "", pending: true }]);
    chain.current = chain.current.then(async () => {
      const history = historyMessages([...turnsRef.current.filter((t) => t.id !== userId && t.id !== botId), user]);
      const t0 = performance.now();
      let reply: string;
      try {
        reply = (await request<{ text: string }>("/brain/respond", json({ messages: history, mode: "conversation" }, 180000))).text;
      } catch (e) {
        patch(botId, { pending: false, failed: true, text: e instanceof Error ? e.message : "Réponse impossible." });
        return;
      }
      const t1 = performance.now();
      patch(botId, { text: reply, ms: { brain: t1 - t0 } });
      try {
        const motion = await request<MotionFile>("/lsf/generate", json(generateBody(reply.slice(0, 400), "c0", true, true), 280000));
        patch(botId, { pending: false, motion, ms: { brain: t1 - t0, signing: performance.now() - t1 } });
        setStage({ motion, text: reply, started: performance.now(), turnId: botId });
      } catch (e) {
        patch(botId, { pending: false, ms: { brain: t1 - t0 } });
        setError(e instanceof Error ? e.message : "Signature impossible.");
      }
    });
  }

  // ------------------------------------------------------------------ webcam + MediaPipe
  function drawOverlay(pose: Landmark[] | null, hands: (Landmark[] | null)[], recording: boolean) {
    const c = overlay.current, v = video.current;
    if (!c || !v) return;
    if (c.width !== v.videoWidth) { c.width = v.videoWidth; c.height = v.videoHeight; }
    const g = c.getContext("2d");
    if (!g) return;
    g.clearRect(0, 0, c.width, c.height);
    const line = (pts: Landmark[], edges: [number, number][], color: string, width: number) => {
      g.strokeStyle = color;
      g.lineWidth = width;
      g.beginPath();
      for (const [a, b] of edges) {
        if (!pts[a] || !pts[b]) continue;
        g.moveTo(pts[a].x * c.width, pts[a].y * c.height);
        g.lineTo(pts[b].x * c.width, pts[b].y * c.height);
      }
      g.stroke();
    };
    if (pose) line(pose, BODY_EDGES, "rgba(255,255,255,.75)", 3);
    hands.forEach((h, k) => h && line(h, HAND_EDGES, recording ? (k ? "#f0a37f" : "#ffd2a6") : "rgba(255,236,220,.9)", 2));
  }

  async function openCamera() {
    setError("");
    setCamera("loading");
    try {
      const s = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
      stream.current = s;
      if (video.current) {
        video.current.srcObject = s;
        await video.current.play();
      }
      if (!tracker.current) {
        const { SignTracker } = await import("../utils/signTracker");
        const t = new SignTracker();
        await t.init();
        tracker.current = t;
      }
    } catch (e) {
      closeCamera();
      setError(e instanceof Error && e.name === "NotAllowedError" ? "Caméra refusée : autorisez la caméra pour signer." : "Caméra ou vision indisponible.");
      return;
    }
    setCamera("ready");
    let lastTime = -1;
    const step = () => {
      const v = video.current, t = tracker.current;
      if (v && t && v.currentTime !== lastTime) {
        lastTime = v.currentTime;
        const f = t.detect(v);
        if (f) {
          const current = take.current;
          drawOverlay(f.pose, [f.left, f.right], Boolean(current));
          if (current) {
            current.points.push(f.points);
            current.t.push(f.t_ms);
            setTakeMs(f.t_ms - current.started);
            const pause = current.pause.push(f.points, f.t_ms);
            setHint(current.pause.raised ? "" : current.pause.signedMs > 0 ? "Mains baissées : fin de la prise…" : "Levez les mains devant la poitrine pour signer.");
            if (pause || f.t_ms - current.started > MAX_TAKE_MS) void finishTake();
          }
        }
      }
      loop.current = requestAnimationFrame(step);
    };
    loop.current = requestAnimationFrame(step);
  }

  function closeCamera() {
    cancelAnimationFrame(loop.current);
    take.current = null;
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
    setCamera("off");
    setLearning(false);
  }

  function startTake(learn: string | null = null) {
    setRecognition(null);
    setNotice("");
    setLearning(Boolean(learn));
    setHint(learn ? "Signez le mot une fois, puis baissez les mains." : "Signez, puis baissez les mains : la prise s’arrête seule.");
    take.current = { points: [], t: [], pause: new HandPause(), started: performance.now(), learn };
    setTakeMs(0);
    setCamera("recording");
  }

  async function finishTake() {
    const current = take.current;
    take.current = null;
    if (!current) return;
    setHint("");
    setLearning(false);
    if (current.points.length < 8) {
      setCamera("ready");
      setError("Prise trop courte : signez au moins une demi-seconde.");
      return;
    }
    const t0 = current.t[0];
    const data_b64 = encodeTake(current.points), t_ms = current.t.map((t) => t - t0);
    if (current.learn) await learnTake(current.learn, data_b64, t_ms);
    else await recognizeTake(data_b64, t_ms);
  }

  async function recognizeTake(data_b64: string, t_ms?: number[]) {
    setCamera((c) => (c === "off" ? "off" : "recognizing"));
    setError("");
    const t0 = performance.now();
    try {
      const result = await request<Recognition>("/sign/recognize", json({ data_b64, t_ms, fps: 25, top_k: 5, vocabulary: vocabRef.current }));
      showRecognition({ ...result, client_ms: performance.now() - t0 });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Reconnaissance impossible.");
    } finally {
      setCamera((c) => (c === "off" ? "off" : "ready"));
    }
  }

  async function learnTake(gloss: string, data_b64: string, t_ms?: number[]) {
    setCamera((c) => (c === "off" ? "off" : "recognizing"));
    try {
      const r = await request<{ label: string; count: number }>("/sign/templates", json({ gloss, data_b64, t_ms, fps: 25 }));
      setNotice(`Votre signe « ${r.label.toUpperCase()} » est enregistré (${r.count} prise${r.count > 1 ? "s" : ""}) : il est reconnu dès maintenant.`);
      loadVocabulary();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Enregistrement impossible.");
    } finally {
      setCamera((c) => (c === "off" ? "off" : "ready"));
    }
  }

  function showRecognition(result: Recognition) {
    const picks = result.segments.map(() => 0);
    setRecognition(result);
    setChoice(picks);
    setDraft(choiceText(result, picks));
    setEdited(false);
  }

  // Development only: feed a recorded take (or a recognition result) without a camera.
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const w = window as unknown as Record<string, unknown>;
    w.signacultureConversation = { recognizeTake, learnTake, showRecognition, send };
    return () => { delete w.signacultureConversation; };
  });

  function pick(segment: number, candidate: number) {
    const next = choice.map((c, i) => (i === segment ? candidate : c));
    setChoice(next);
    if (recognition && !edited) setDraft(choiceText(recognition, next));
  }

  function confirmSigns() {
    if (!recognition || !draft.trim()) return;
    send(draft, "signes", recognition.client_ms ?? recognition.ms);
    setRecognition(null);
  }

  /** The avatar shows one lexicon sign (B2 clip of the word), to learn it or to compare. */
  async function showSign(gloss: string, label: string) {
    setWord(gloss);
    setError("");
    try {
      const motion = await request<MotionFile>("/lsf/generate", json({ ...generateBody(label, "b2", true, false, [[gloss]]) }, 120000));
      setStage({ motion, text: label.toUpperCase(), started: performance.now() });
    } catch (e) {
      setError(e instanceof Error ? e.message : "Signe indisponible.");
    }
  }

  async function forget(gloss: string) {
    await request(`/sign/templates/${encodeURIComponent(gloss)}`, { method: "DELETE" }).catch(() => {});
    loadVocabulary();
  }

  // ------------------------------------------------------------------ microphone (same path as the live mode)
  async function startMic() {
    setError("");
    let s: MediaStream;
    try {
      s = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch {
      setError("Micro refusé ou indisponible.");
      return;
    }
    const context = new AudioContext();
    const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" }));
    await context.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const source = context.createMediaStreamSource(s);
    const node = new AudioWorkletNode(context, "signaculture-capture");
    const rate = context.sampleRate;
    const vad = new PhraseDetector({ sampleRate: rate }, (samples) => {
      const form = new FormData();
      form.append("file", new Blob([encodeWav(samples, rate)], { type: "audio/wav" }), "phrase.wav");
      const t0 = performance.now();
      request<{ text: string }>("/live/transcribe", { method: "POST", body: form })
        .then((r) => r.text.trim() && send(r.text, "voix", performance.now() - t0))
        .catch((e) => setError(e instanceof Error ? e.message : "Transcription impossible."));
    });
    detector.current = vad;
    node.port.onmessage = (e: MessageEvent<Float32Array>) => vad.push(e.data);
    source.connect(node);
    stopMic.current = () => {
      vad.flush();
      node.port.onmessage = null;
      source.disconnect();
      s.getTracks().forEach((t) => t.stop());
      void context.close();
      detector.current = null;
      stopMic.current = () => {};
      setMicOn(false);
    };
    setMicOn(true);
  }

  const motionTime = stage ? Math.min(now - stage.started, stage.motion.duration_ms) : 0;
  const recording = camera === "recording";
  const level = detector.current ? Math.min(1, detector.current.level * 12) : 0;
  const mine = new Map((vocab?.user ?? []).map((u) => [u.gloss, u.count]));
  const words = vocab ? [...vocab.conversation, ...vocab.user.filter((u) => !vocab.conversation.some((c) => c.gloss === u.gloss))] : [];
  const selected = words.find((w) => w.gloss === word);
  return (
    <div className="conversation-mode">
      <p className="film-disclaimer">
        Signez face à la caméra, parlez ou écrivez : l’avatar vous répond en langue des signes et en texte. Tout est traité sur cet ordinateur.
      </p>
      {brainReady === false && (
        <p role="status" className="research-error">Le modèle de langage local n’est pas démarré : lancez Ollama sur cet ordinateur pour converser.</p>
      )}
      {error && <p role="alert" className="research-error">{error}</p>}
      <div className="conversation-stage">
        <section className="conversation-input">
          <div className={`conversation-camera${recording ? " recording" : ""}`}>
            <video ref={video} playsInline muted aria-label="Aperçu de la caméra" />
            <canvas ref={overlay} aria-hidden="true" />
            {camera === "off" && (
              <div className="film-placeholder">
                <Hand size={34} />
                <span>Activez la caméra pour signer</span>
              </div>
            )}
            {camera === "loading" && <div className="film-placeholder"><Loader2 className="spin" size={28} /><span>Préparation de la vision…</span></div>}
            {recording && <span className="conversation-rec"><i /> {learning ? `Votre signe « ${selected?.label.toUpperCase() ?? ""} »` : "Prise"} · {(takeMs / 1000).toFixed(1)} s</span>}
            {hint && recording && <div className="film-subtitle">{hint}</div>}
          </div>
          <div className="film-actions">
            {camera === "off" ? (
              <button className="primary-button" disabled={signRecognition === false} onClick={() => void openCamera()}>
                <Camera size={16} /> {signRecognition === false ? "Reconnaissance des signes indisponible" : "Activer la caméra"}
              </button>
            ) : recording ? (
              <button className="primary-button" onClick={() => void finishTake()}>
                <Square size={15} /> Terminer la prise
              </button>
            ) : (
              <button className="primary-button" disabled={camera !== "ready"} onClick={() => startTake()}>
                {camera === "recognizing" ? <Loader2 className="spin" size={16} /> : <Hand size={16} />}
                {camera === "recognizing" ? "Lecture des signes…" : "Signer"}
              </button>
            )}
            {camera !== "off" && (
              <button className="secondary-button" onClick={closeCamera} aria-label="Couper la caméra"><CameraOff size={15} /></button>
            )}
            <button className="secondary-button" disabled={micReady !== "ready"} onClick={() => (micOn ? stopMic.current() : void startMic())}
              title={micReady === "off" ? "Reconnaissance vocale indisponible" : ""}>
              {micOn ? <MicOff size={15} /> : micReady === "loading" ? <Loader2 className="spin" size={15} /> : <Mic size={15} />}
              {micOn ? "Couper le micro" : "Parler"}
            </button>
            {micOn && (
              <span className={`live-level${detector.current?.speaking ? " speaking" : ""}`} aria-label="Niveau du micro">
                <i style={{ width: `${Math.round(level * 100)}%` }} />
              </span>
            )}
          </div>
          <div className="film-options conversation-vocab" role="radiogroup" aria-label="Vocabulaire reconnu">
            <span>Signes reconnus :</span>
            <label>
              <input type="radio" name="conv-vocab" checked={vocabMode === "conversation"} onChange={() => setVocabMode("conversation")} />
              mots du quotidien{vocab ? ` (${vocab.conversation.length + (vocab.user.filter((u) => !vocab.conversation.some((c) => c.gloss === u.gloss)).length)})` : ""}
            </label>
            <label>
              <input type="radio" name="conv-vocab" checked={vocabMode === "all"} onChange={() => setVocabMode("all")} />
              lexique complet{vocab ? ` (${vocab.size})` : ""}
            </label>
          </div>
          {notice && <p className="conversation-notice" role="status">{notice}</p>}
          {recognition && (
            <div className="card conversation-confirm" aria-live="polite">
              {recognition.glosses.length ? (
                <>
                  <strong>Vous avez signé : « {draft || choiceText(recognition, choice)} » ?</strong>
                  <div className="conversation-segments">
                    {recognition.segments.map((s, k) => (
                      <div key={k} role="radiogroup" aria-label={`Signe ${k + 1}`}>
                        {s.candidates.map((c, j) => (
                          <button key={c.gloss} role="radio" aria-checked={choice[k] === j} className={choice[k] === j ? "active" : ""}
                            onClick={() => pick(k, j)}>
                            {c.label.toUpperCase()} <small>{Math.round(c.score * 100)} %</small>
                          </button>
                        ))}
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <strong>Aucun signe reconnu : levez les mains devant la poitrine et signez un mot à la fois.</strong>
              )}
              <label className="conversation-correct">
                <span>Corriger</span>
                <input value={draft} onChange={(e) => { setDraft(e.target.value); setEdited(true); }} placeholder="Écrivez ce que vous avez signé"
                  onKeyDown={(e) => e.key === "Enter" && confirmSigns()} />
              </label>
              <div className="film-actions">
                <button className="primary-button" disabled={!draft.trim()} onClick={confirmSigns}><Check size={15} /> Oui, envoyer</button>
                <button className="secondary-button" disabled={camera === "off"} onClick={() => startTake()}><RotateCcw size={15} /> Signer à nouveau</button>
              </div>
              <small className="film-phase">
                {recognition.segments.length} signe{recognition.segments.length > 1 ? "s" : ""} · {recognition.vocabulary} signes possibles ·
                lu en {((recognition.client_ms ?? recognition.ms) / 1000).toFixed(1)} s
              </small>
            </div>
          )}
          <form className="conversation-type" onSubmit={(e) => { e.preventDefault(); send(typed, "texte"); setTyped(""); }}>
            <Keyboard size={16} aria-hidden="true" />
            <input value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="Ou écrivez votre message…" aria-label="Message écrit" />
            <button className="primary-button" disabled={!typed.trim()} aria-label="Envoyer"><Send size={15} /></button>
          </form>
          {words.length > 0 && (
            <details className="card conversation-words">
              <summary><BookOpen size={15} /> Signes à connaître · l’avatar vous les montre, enregistrez votre version</summary>
              <div className="conversation-chips">
                {words.map((w) => (
                  <button key={w.gloss} className={w.gloss === word ? "active" : ""} onClick={() => void showSign(w.gloss, w.label)}>
                    {w.label.toUpperCase()}{mine.has(w.gloss) && <small> · {mine.get(w.gloss)}</small>}
                  </button>
                ))}
              </div>
              {selected && (
                <div className="film-actions">
                  <button className="secondary-button" onClick={() => void showSign(selected.gloss, selected.label)}><Play size={14} /> Revoir « {selected.label.toUpperCase()} »</button>
                  <button className="secondary-button" disabled={camera !== "ready"} onClick={() => startTake(selected.gloss)}
                    title={camera === "off" ? "Activez la caméra" : ""}>
                    <Video size={14} /> Enregistrer mon signe
                  </button>
                  {mine.has(selected.gloss) && (
                    <button className="secondary-button" onClick={() => void forget(selected.gloss)}><Trash2 size={14} /> Effacer mes prises</button>
                  )}
                </div>
              )}
            </details>
          )}
        </section>
        <section className="conversation-reply">
          <div className="film-avatar live">
            <AvatarCanvas avatarCredit={avatarCredit} motion={stage?.motion ?? null} currentTimeMs={motionTime} />
            {stage && <div className="film-subtitle">{stage.text}</div>}
            <small className="film-watermark">{WATERMARK}</small>
          </div>
          <ol className="conversation-history" aria-label="Échanges">
            {turns.length === 0 && <li className="conversation-empty">Dites bonjour en signes, à la voix ou par écrit.</li>}
            {turns.map((t) => (
              <li key={t.id} className={`${t.role}${stage?.turnId === t.id ? " current" : ""}${t.failed ? " failed" : ""}`}>
                <span className="conversation-who">{t.role === "user" ? `Vous, ${VIA_LABEL[t.via ?? "texte"]}` : "SignaCulture"}</span>
                <span className="conversation-text">{t.pending && !t.text ? <Loader2 className="spin" size={14} /> : t.text}</span>
                <span className="film-state">
                  {t.role === "assistant" && t.pending && t.text ? "signature…" : ""}
                  {t.ms?.recognition !== undefined ? `compris en ${(t.ms.recognition / 1000).toFixed(1)} s` : ""}
                  {t.ms?.brain !== undefined ? `réponse ${(t.ms.brain / 1000).toFixed(1)} s` : ""}
                  {t.ms?.signing !== undefined ? ` · signes ${(t.ms.signing / 1000).toFixed(1)} s` : ""}
                  {t.motion && (
                    <button className="link-button" onClick={() => setStage({ motion: t.motion!, text: t.text, started: performance.now(), turnId: t.id })}>Revoir</button>
                  )}
                </span>
              </li>
            ))}
          </ol>
        </section>
      </div>
    </div>
  );
}
