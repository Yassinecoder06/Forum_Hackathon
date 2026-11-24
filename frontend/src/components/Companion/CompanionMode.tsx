/// <reference types="vite/client" />
import { useEffect, useRef, useState } from "react";
import { Box, Camera, Check, Eye, EyeOff, Hand, Keyboard, Loader2, Mic, MicOff, Send, X } from "lucide-react";
import { request } from "../../services/api";
import type { MotionFile } from "../../types/contract";
import { generateBody } from "../../utils/lsfRequest";
import { encodeWav, openPhraseMic, type PhraseDetector } from "../../utils/phraseDetector";
import { HandPause, encodeTake } from "../../utils/signCapture";
import type { SignTracker } from "../../utils/signTracker";
import { choiceText, historyMessages, type Recognition, type Turn, type Via } from "../../utils/conversation";
import {
  SCENE_CHANGE, SCENE_CHECK_MS, SCENE_MIN_GAP_MS, SCENE_REMARK, SCENE_THUMB, lookAt, lookSteps, pinchPlacement, sceneDifference,
  toGray, type Focus, type Placement, type Touch2,
} from "../../utils/companion";
import type { RoleStep } from "../../utils/roleShift";
import { CompanionStage, grabVideo, pixelsToJpeg, thumbnail, thumbnailFromPixels, type StageStatus } from "./companionStage";
import "./companion.css";

type View = "intro" | "camera" | "xr";
type SignState = "off" | "loading" | "recording" | "recognizing" | "confirm";
type CTurn = Omit<Turn, "ms" | "via"> & {
  via?: Via | "scene";
  hidden?: boolean;
  focus?: Focus;
  ms?: { recognition?: number; frame?: number; brain?: number; signing?: number };
};
type Playback = { motion: MotionFile; started: number; look: RoleStep[] };
type BrainReply = { text: string; focus?: Focus; ms?: number };

const json = (body: unknown, timeout = 60000) => ({
  method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body), signal: AbortSignal.timeout(timeout),
});
const GREETING = "Je viens d’ouvrir la caméra. Salue-moi et dis en une phrase ce que tu vois autour de moi.";
const MAX_TAKE_MS = 15000;
/** After a reply has been signed, the avatar holds the last pose briefly, then returns to rest. */
const REST_AFTER_MS = 900;

/**
 * The AR companion (mobile first): the MetaHuman stands in the person's real room, sees it through the phone's camera
 * (one JPEG frame of about 512 px per turn, sent with the conversation to the brain: POST /brain/respond, mode
 * "companion"), and answers in LSF (POST /lsf/generate, C0) and in subtitles. Android Chrome: WebXR immersive-ar,
 * tap a surface to place it. iPhone Safari and browsers without WebXR AR: the rear camera full screen with the avatar
 * composited on top. Input: voice (phrases cut at pauses -> /live/transcribe), text, or signs with the front camera
 * (MediaPipe -> /sign/recognize, as in the conversation page).
 */
export function CompanionMode({ onExit, transcription, signRecognition, brainReady }: {
  onExit: () => void; transcription?: boolean; signRecognition?: boolean; brainReady?: boolean;
}) {
  const [view, setView] = useState<View>("intro"),
    [xrSupported, setXrSupported] = useState<boolean | null>(null),
    [stageStatus, setStageStatus] = useState<StageStatus>({ ready: false, placed: false, surface: false }),
    [turns, setTurns] = useState<CTurn[]>([]),
    [typed, setTyped] = useState(""),
    [typing, setTyping] = useState(false),
    [micOn, setMicOn] = useState(false),
    [micReady, setMicReady] = useState<"loading" | "ready" | "off">(transcription ? "loading" : "off"),
    [level, setLevel] = useState(0),
    [watching, setWatching] = useState(false),
    [sign, setSign] = useState<SignState>("off"),
    [signHint, setSignHint] = useState(""),
    [recognition, setRecognition] = useState<Recognition | null>(null),
    [draft, setDraft] = useState(""),
    [error, setError] = useState("");
  const root = useRef<HTMLDivElement>(null),
    ui = useRef<HTMLDivElement>(null),
    stageHost = useRef<HTMLDivElement>(null),
    video = useRef<HTMLVideoElement>(null),
    stage = useRef<CompanionStage | null>(null),
    stream = useRef<MediaStream | null>(null),
    stopTestCam = useRef<() => void>(() => {}),
    playback = useRef<Playback | null>(null),
    chain = useRef<Promise<void>>(Promise.resolve()),
    busy = useRef(0),
    turnsRef = useRef<CTurn[]>([]),
    viewRef = useRef<View>("intro"),
    lastThumb = useRef<Uint8Array | null>(null),
    lastRemark = useRef(0),
    greeted = useRef(false),
    mic = useRef<{ detector: PhraseDetector; stop: () => void } | null>(null),
    tracker = useRef<SignTracker | null>(null),
    signLoop = useRef(0),
    gesture = useRef<{ start: Touch2; base: Placement } | null>(null),
    drag = useRef<{ x: number; offset: number } | null>(null),
    timings = useRef<Record<string, unknown> | null>(null);
  turnsRef.current = turns;
  viewRef.current = view;

  // ------------------------------------------------------------------ set-up: stage, models, capabilities
  useEffect(() => {
    const host = stageHost.current;
    if (!host) return;
    let s: CompanionStage;
    try {
      s = new CompanionStage(host);
    } catch {
      setError("L’affichage 3D est indisponible sur cet appareil.");
      return;
    }
    stage.current = s;
    s.onStatus = setStageStatus;
    s.onSessionEnd = () => {
      setView("intro");
      greeted.current = false;
    };
    s.frameSource = () => {
      const p = playback.current;
      if (!p) return { motion: null, currentTimeMs: 0 };
      const t = performance.now() - p.started;
      if (t > p.motion.duration_ms + REST_AFTER_MS) {
        playback.current = null;
        return { motion: null, currentTimeMs: 0 };
      }
      const time = Math.min(t, p.motion.duration_ms);
      return { motion: p.motion, currentTimeMs: time, roleShift: lookAt(p.look, time) };
    };
    const xr = navigator.xr;
    if (xr && window.isSecureContext) xr.isSessionSupported("immersive-ar").then(setXrSupported).catch(() => setXrSupported(false));
    else setXrSupported(false);
    // Load the brain and the signing engine while the person gets ready (same warm-ups as the conversation page).
    request("/brain/warmup", { method: "POST", signal: AbortSignal.timeout(200000) }).catch(() => {});
    if (!transcription) request("/lsf/generate", json(generateBody("Bonjour.", "c0", true, true), 280000)).catch(() => {});
    return () => {
      s.dispose();
      stage.current = null;
      closeCamera();
      mic.current?.stop();
      cancelAnimationFrame(signLoop.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);
  useEffect(() => {
    if (!transcription) return;
    request("/live/warmup?engine=c0", { method: "POST", signal: AbortSignal.timeout(300000) })
      .then(() => setMicReady("ready")).catch(() => setMicReady("off"));
  }, [transcription]);

  // Controls on the AR overlay must not also place the avatar (Chrome sends taps on the DOM overlay to the session).
  useEffect(() => {
    const el = ui.current;
    if (!el) return;
    const stop = (e: Event) => {
      if ((e.target as HTMLElement).closest(".companion-controls")) e.preventDefault();
    };
    el.addEventListener("beforexrselect", stop);
    return () => el.removeEventListener("beforexrselect", stop);
  }, []);

  // Microphone level meter (no re-render of the whole page per audio frame: 10 Hz is enough).
  useEffect(() => {
    if (!micOn) return;
    const id = window.setInterval(() => setLevel(Math.min(1, (mic.current?.detector.level ?? 0) * 12)), 100);
    return () => window.clearInterval(id);
  }, [micOn]);

  // Scene watcher (optional): a 16x12 thumbnail every 1.5 s; a big change -> one spontaneous remark (never streamed).
  useEffect(() => {
    if (!watching || view === "intro") return;
    const id = window.setInterval(async () => {
      if (busy.current || sign !== "off" || performance.now() - lastRemark.current < SCENE_MIN_GAP_MS) return;
      const thumb = await sceneThumb();
      if (!thumb || !lastThumb.current) {
        if (thumb) lastThumb.current = thumb;
        return;
      }
      if (sceneDifference(thumb, lastThumb.current) > SCENE_CHANGE) {
        lastRemark.current = performance.now();
        send(SCENE_REMARK, "scene", true);
      }
    }, SCENE_CHECK_MS);
    return () => window.clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [watching, view, sign]);

  // Greeting: once the avatar stands in the scene, it says hello and what it sees.
  useEffect(() => {
    if (view === "intro" || greeted.current || !stageStatus.ready || !stageStatus.placed) return;
    greeted.current = true;
    const id = window.setTimeout(() => send(GREETING, "scene", true), view === "camera" ? 800 : 300);
    return () => window.clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [view, stageStatus.ready, stageStatus.placed]);

  // ------------------------------------------------------------------ camera (overlay mode)
  async function openRearCamera(facing: "environment" | "user" = "environment") {
    closeCamera();
    const v = video.current;
    if (!v) return false;
    const test = import.meta.env.DEV ? new URLSearchParams(location.search).get("testcam") : null;
    try {
      const s = test ? await testCamera(test) : await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } }, audio: false,
      });
      stream.current = s;
      v.srcObject = s;
      await v.play();
      v.classList.toggle("mirrored", facing === "user" && !test);
      return true;
    } catch (e) {
      closeCamera();
      setError(e instanceof Error && e.name === "NotAllowedError"
        ? "Caméra refusée : autorisez la caméra pour que le compagnon voie la pièce."
        : window.isSecureContext ? "Caméra indisponible." : "La caméra demande une adresse sécurisée (https).");
      return false;
    }
  }

  /** Development only: a still picture as the camera stream (slow pan), through the same <video> path as a camera. */
  function testCamera(url: string): Promise<MediaStream> {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => {
        const canvas = document.createElement("canvas");
        canvas.width = 1280;
        canvas.height = 720;
        const g = canvas.getContext("2d")!;
        let id = 0;
        const t0 = performance.now();
        const draw = () => {
          const dx = Math.sin((performance.now() - t0) / 4000) * 20;
          g.drawImage(img, -20 + dx, 0, 1320, 720);
          id = requestAnimationFrame(draw);
        };
        draw();
        stopTestCam.current = () => cancelAnimationFrame(id);
        resolve(canvas.captureStream(15));
      };
      img.onerror = () => reject(new Error("test camera"));
      img.src = url;
    });
  }

  function closeCamera() {
    stopTestCam.current();
    stopTestCam.current = () => {};
    stream.current?.getTracks().forEach((t) => t.stop());
    stream.current = null;
    if (video.current) video.current.srcObject = null;
  }

  async function startCamera() {
    setError("");
    if (!(await openRearCamera())) return;
    stage.current?.useOverlay();
    setView("camera");
  }

  async function startXR() {
    setError("");
    const s = stage.current, overlay = ui.current;
    if (!s || !overlay) return;
    try {
      closeCamera();
      await s.startXR(overlay);
      setView("xr");
    } catch {
      setError("La réalité augmentée n’a pas pu démarrer : la vue caméra prend le relais.");
      await startCamera();
    }
  }

  function leave() {
    if (viewRef.current === "xr") stage.current?.endXR();
    closeCamera();
    mic.current?.stop();
    mic.current = null;
    setMicOn(false);
    onExit();
  }

  // ------------------------------------------------------------------ what the companion sees
  /** The camera frame for the brain: JPEG, longest side 512 px. */
  async function grabScene(): Promise<string | null> {
    if (viewRef.current === "xr") {
      const f = await stage.current?.capture(512);
      if (!f) return null;
      lastThumb.current = thumbFromFrame(f);
      return pixelsToJpeg(f);
    }
    const v = video.current;
    if (!v || !stream.current || v.classList.contains("mirrored")) return null;
    const thumb = thumbnail(v, SCENE_THUMB.width, SCENE_THUMB.height);
    if (thumb) lastThumb.current = toGray(thumb);
    return grabVideo(v, 512);
  }

  function thumbFromFrame(f: { pixels: Uint8Array; width: number; height: number }) {
    const t = thumbnailFromPixels(f, SCENE_THUMB.width, SCENE_THUMB.height);
    return t ? toGray(t) : null;
  }

  async function sceneThumb(): Promise<Uint8Array | null> {
    if (viewRef.current === "xr") {
      const f = await stage.current?.capture(64);
      return f ? thumbFromFrame(f) : null;
    }
    const v = video.current;
    if (!v || !stream.current || v.classList.contains("mirrored")) return null;
    const t = thumbnail(v, SCENE_THUMB.width, SCENE_THUMB.height);
    return t ? toGray(t) : null;
  }

  // ------------------------------------------------------------------ one exchange
  const patch = (id: number, p: Partial<CTurn>) => setTurns((prev) => prev.map((t) => (t.id === id ? { ...t, ...p } : t)));

  /** The person's words (+ what the camera sees now) -> brain -> reply text -> LSF signing by the avatar. In order. */
  function send(text: string, via: CTurn["via"], hidden = false, recognitionMs?: number) {
    const clean = text.trim();
    if (!clean) return;
    setError("");
    busy.current++;
    const userId = Date.now() + Math.random(), botId = userId + 0.5;
    const user: CTurn = { id: userId, role: "user", text: clean, via, hidden, ms: recognitionMs !== undefined ? { recognition: recognitionMs } : undefined };
    setTurns((prev) => [...prev, user, { id: botId, role: "assistant", text: "", pending: true }]);
    chain.current = chain.current.then(async () => {
      try {
        const history = historyMessages([...turnsRef.current.filter((t) => t.id !== userId && t.id !== botId), user] as Turn[]);
        const t0 = performance.now();
        const image = await grabScene().catch(() => null);
        const t1 = performance.now();
        let reply: BrainReply;
        try {
          reply = await request<BrainReply>("/brain/respond", json({ messages: history, mode: "companion", ...(image ? { image_b64: image } : {}) }, 180000));
        } catch (e) {
          patch(botId, { pending: false, failed: true, text: e instanceof Error ? e.message : "Réponse impossible." });
          return;
        }
        const t2 = performance.now();
        const ms = { frame: image ? t1 - t0 : undefined, brain: t2 - t1 };
        patch(botId, { text: reply.text, focus: reply.focus, ms });
        try {
          const motion = await request<MotionFile>("/lsf/generate", json(generateBody(reply.text.slice(0, 400), "c0", true, true), 280000));
          const t3 = performance.now();
          patch(botId, { pending: false, motion, ms: { ...ms, signing: t3 - t2 } });
          play(motion, reply.focus);
          timings.current = { frame: ms.frame ?? 0, brain: ms.brain, signing: t3 - t2, total: t3 - t0, image: Boolean(image), focus: reply.focus ?? null, text: reply.text };
        } catch (e) {
          patch(botId, { pending: false });
          setError(e instanceof Error ? e.message : "Signature impossible.");
        }
      } finally {
        busy.current--;
      }
    });
  }

  function play(motion: MotionFile, focus?: Focus) {
    playback.current = { motion, started: performance.now(), look: lookSteps(focus, motion.duration_ms) };
  }

  // Development only: drive the page without a phone (scripted turns, timings).
  useEffect(() => {
    if (!import.meta.env.DEV) return;
    const w = window as unknown as Record<string, unknown>;
    w.signacultureCompanion = { send, play, timings: () => timings.current, turns: () => turnsRef.current, stage: () => stage.current };
    return () => { delete w.signacultureCompanion; };
  });

  // ------------------------------------------------------------------ voice
  async function toggleMic() {
    if (mic.current) {
      mic.current.stop();
      mic.current = null;
      setMicOn(false);
      return;
    }
    setError("");
    try {
      mic.current = await openPhraseMic((samples, rate) => {
        const form = new FormData();
        form.append("file", new Blob([encodeWav(samples, rate)], { type: "audio/wav" }), "phrase.wav");
        const t0 = performance.now();
        request<{ text: string }>("/live/transcribe", { method: "POST", body: form })
          .then((r) => r.text.trim() && send(r.text, "voix", false, performance.now() - t0))
          .catch((e) => setError(e instanceof Error ? e.message : "Transcription impossible."));
      });
      setMicOn(true);
    } catch {
      setError("Micro refusé ou indisponible.");
    }
  }

  // ------------------------------------------------------------------ signs (front camera, overlay mode)
  async function startSigning() {
    setError("");
    setRecognition(null);
    setSign("loading");
    if (!(await openRearCamera("user"))) {
      setSign("off");
      await openRearCamera();
      return;
    }
    try {
      if (!tracker.current) {
        const { SignTracker } = await import("../../utils/signTracker");
        const t = new SignTracker();
        await t.init();
        tracker.current = t;
      }
    } catch {
      setError("Vision indisponible : écrivez ou parlez.");
      await stopSigning();
      return;
    }
    const take = { points: [] as Float32Array[], t: [] as number[], pause: new HandPause(), started: -1 };
    setSign("recording");
    setSignHint("Tenez le téléphone face à vous, signez, puis baissez les mains.");
    let lastTime = -1;
    const step = () => {
      const v = video.current, t = tracker.current;
      if (v && t && v.currentTime !== lastTime) {
        lastTime = v.currentTime;
        const f = t.detect(v);
        if (f) {
          if (take.started < 0) take.started = f.t_ms;
          take.points.push(f.points);
          take.t.push(f.t_ms);
          const done = take.pause.push(f.points, f.t_ms);
          setSignHint(take.pause.raised ? "" : take.pause.signedMs > 0 ? "Mains baissées : lecture…" : "Levez les mains devant la poitrine pour signer.");
          if (done || f.t_ms - take.started > MAX_TAKE_MS) {
            void finishSigning(take);
            return;
          }
        }
      }
      signLoop.current = requestAnimationFrame(step);
    };
    signLoop.current = requestAnimationFrame(step);
  }

  async function finishSigning(take: { points: Float32Array[]; t: number[] }) {
    cancelAnimationFrame(signLoop.current);
    setSignHint("");
    if (take.points.length < 8) {
      setError("Prise trop courte : signez au moins une demi-seconde.");
      await stopSigning();
      return;
    }
    setSign("recognizing");
    const t0 = performance.now();
    try {
      const result = await request<Recognition>("/sign/recognize", json({
        data_b64: encodeTake(take.points), t_ms: take.t.map((t) => t - take.t[0]), fps: 25, top_k: 3, vocabulary: "conversation",
      }));
      const picks = result.segments.map(() => 0);
      setRecognition({ ...result, client_ms: performance.now() - t0 });
      setDraft(choiceText(result, picks));
      setSign("confirm");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Reconnaissance impossible.");
      await stopSigning();
    }
  }

  async function stopSigning() {
    cancelAnimationFrame(signLoop.current);
    setSign("off");
    setSignHint("");
    await openRearCamera();
  }

  async function confirmSigns() {
    const text = draft.trim(), ms = recognition?.client_ms;
    setRecognition(null);
    await stopSigning();
    if (text) send(text, "signes", false, ms);
  }

  // ------------------------------------------------------------------ gestures: pinch / twist (both modes), slide (camera view)
  const touches = (e: React.TouchEvent): Touch2 => ({
    x0: e.touches[0].clientX, y0: e.touches[0].clientY, x1: e.touches[1].clientX, y1: e.touches[1].clientY,
  });
  function onTouchStart(e: React.TouchEvent) {
    const s = stage.current;
    if (!s) return;
    if (e.touches.length === 2) {
      gesture.current = { start: touches(e), base: { ...s.placement } };
      drag.current = null;
      s.ignoreSelectUntil = performance.now() + 100000;
    } else if (e.touches.length === 1 && viewRef.current === "camera") {
      drag.current = { x: e.touches[0].clientX, offset: s.offsetX };
    }
  }
  function onTouchMove(e: React.TouchEvent) {
    const s = stage.current;
    if (!s) return;
    if (gesture.current && e.touches.length === 2) s.setPlacement(pinchPlacement(gesture.current.start, touches(e), gesture.current.base));
    else if (drag.current && e.touches.length === 1) {
      const width = root.current?.clientWidth || 400;
      s.offsetX = Math.max(-1.2, Math.min(1.2, drag.current.offset + ((e.touches[0].clientX - drag.current.x) / width) * 2));
    }
  }
  function onTouchEnd(e: React.TouchEvent) {
    if (e.touches.length < 2 && gesture.current) {
      gesture.current = null;
      if (stage.current) stage.current.ignoreSelectUntil = performance.now() + 400;
    }
    if (!e.touches.length) drag.current = null;
  }
  function onWheel(e: React.WheelEvent) {
    const s = stage.current;
    if (s) s.setPlacement({ ...s.placement, scale: s.placement.scale * Math.exp(-e.deltaY * 0.001) });
  }

  // ------------------------------------------------------------------ render
  const visible = turns.filter((t) => !t.hidden);
  const lastBot = [...turns].reverse().find((t) => t.role === "assistant");
  const lastUser = [...visible].reverse().find((t) => t.role === "user");
  const thinking = Boolean(lastBot?.pending);
  const signingCam = sign !== "off";
  const live = view !== "intro";
  const ms = lastBot?.ms;
  return (
    <div ref={root} className={`companion companion--${view}${signingCam ? " companion--signing" : ""}`}>
      <video ref={video} className="companion-video" playsInline muted aria-label="Ce que voit la caméra" />
      <div ref={stageHost} className="companion-stage" aria-label="Le compagnon SignePourMoi en 3D" />
      <div ref={ui} className="companion-ui">
        <div className="companion-gestures" onTouchStart={onTouchStart} onTouchMove={onTouchMove} onTouchEnd={onTouchEnd}
          onTouchCancel={onTouchEnd} onWheel={onWheel} aria-hidden="true" />
        <header className="companion-top companion-controls">
          <span className="companion-brand">SignePourMoi · Compagnon</span>
          <button className="companion-icon" onClick={leave} aria-label="Quitter le compagnon"><X size={18} /></button>
        </header>

        {view === "intro" && (
          <section className="companion-intro companion-controls">
            <span className="eyebrow">COMPAGNON EN LSF</span>
            <h1>Il vous voit, il vous <em>répond.</em></h1>
            <p>Votre compagnon se tient dans votre pièce, regarde ce qui vous entoure et vous parle en langue des signes et en texte.</p>
            {brainReady === false && <p className="companion-note">Le modèle de langage local n’est pas démarré : lancez Ollama sur l’ordinateur.</p>}
            {xrSupported && (
              <button className="primary-button" disabled={!stageStatus.ready} onClick={() => void startXR()}>
                <Box size={17} /> Placer le compagnon dans la pièce
              </button>
            )}
            <button className={xrSupported ? "secondary-button" : "primary-button"} disabled={!stageStatus.ready} onClick={() => void startCamera()}>
              <Camera size={17} /> {xrSupported ? "Vue caméra" : "Ouvrir la caméra"}
            </button>
            {!stageStatus.ready && !stageStatus.error && <small className="companion-loading"><Loader2 className="spin" size={14} /> Arrivée du compagnon…</small>}
            {stageStatus.error && <p className="companion-note">{stageStatus.error}</p>}
          </section>
        )}

        {view === "xr" && !stageStatus.placed && (
          <div className="companion-hint" role="status">
            {stageStatus.surface ? "Touchez l’écran pour placer le compagnon sur la surface." : "Balayez lentement le sol ou une table avec le téléphone…"}
          </div>
        )}
        {signHint && <div className="companion-hint" role="status">{signHint}</div>}
        {sign === "loading" && <div className="companion-hint" role="status"><Loader2 className="spin" size={14} /> Préparation de la vision…</div>}
        {sign === "recognizing" && <div className="companion-hint" role="status"><Loader2 className="spin" size={14} /> Lecture des signes…</div>}

        {live && (
          <div className="companion-bottom">
            {(lastBot || lastUser) && !signingCam && (
              <div className="companion-subtitles" aria-live="polite">
                {lastUser && <p className="companion-said">Vous : {lastUser.text}</p>}
                {lastBot && (
                  <p className={`companion-reply${lastBot.failed ? " failed" : ""}`}>
                    {thinking && !lastBot.text ? <><Loader2 className="spin" size={15} /> Il regarde et réfléchit…</> : lastBot.text}
                  </p>
                )}
                {ms?.brain !== undefined && (
                  <small className="companion-timing">
                    {ms.frame !== undefined ? `image ${(ms.frame / 1000).toFixed(2)} s · ` : ""}réponse {(ms.brain / 1000).toFixed(1)} s
                    {ms.signing !== undefined ? ` · signes ${(ms.signing / 1000).toFixed(1)} s` : thinking ? " · signature…" : ""}
                  </small>
                )}
              </div>
            )}
            {sign === "confirm" && recognition && (
              <div className="companion-card companion-controls" aria-live="polite">
                <strong>{recognition.glosses.length ? `Vous avez signé : « ${draft} » ?` : "Aucun signe reconnu : signez un mot à la fois."}</strong>
                <input value={draft} onChange={(e) => setDraft(e.target.value)} placeholder="Corrigez ou écrivez ici" aria-label="Ce que vous avez signé"
                  onKeyDown={(e) => e.key === "Enter" && void confirmSigns()} />
                <div className="companion-row">
                  <button className="primary-button" disabled={!draft.trim()} onClick={() => void confirmSigns()}><Check size={15} /> Envoyer</button>
                  <button className="secondary-button" onClick={() => { setRecognition(null); void stopSigning(); }}>Annuler</button>
                </div>
              </div>
            )}
            {error && <p role="alert" className="companion-error companion-controls" onClick={() => setError("")}>{error}</p>}
            <div className="companion-dock companion-controls">
              {typing ? (
                <form className="companion-type" onSubmit={(e) => { e.preventDefault(); send(typed, "texte"); setTyped(""); setTyping(false); }}>
                  <input autoFocus value={typed} onChange={(e) => setTyped(e.target.value)} placeholder="Votre message…" aria-label="Message écrit" />
                  <button className="companion-round" disabled={!typed.trim()} aria-label="Envoyer"><Send size={18} /></button>
                  <button type="button" className="companion-round ghost" onClick={() => setTyping(false)} aria-label="Fermer le clavier"><X size={18} /></button>
                </form>
              ) : (
                <>
                  <button className={`companion-round ghost${watching ? " on" : ""}`} onClick={() => setWatching((w) => !w)}
                    aria-pressed={watching} aria-label={watching ? "Ne plus commenter la scène" : "Commenter quand la scène change"}
                    title="Commenter quand la scène change">
                    {watching ? <Eye size={19} /> : <EyeOff size={19} />}
                  </button>
                  <button className="companion-round ghost" onClick={() => setTyping(true)} aria-label="Écrire"><Keyboard size={19} /></button>
                  <button className={`companion-mic${micOn ? " on" : ""}`} disabled={micReady !== "ready" || signingCam} onClick={() => void toggleMic()}
                    aria-label={micOn ? "Couper le micro" : "Parler"} style={{ ["--level" as string]: level }}>
                    {micOn ? <MicOff size={24} /> : micReady === "loading" ? <Loader2 className="spin" size={24} /> : <Mic size={24} />}
                  </button>
                  {view === "camera" && signRecognition !== false && (
                    <button className={`companion-round ghost${signingCam ? " on" : ""}`} disabled={sign === "loading" || sign === "recognizing"}
                      onClick={() => void (signingCam ? stopSigning() : startSigning())} aria-label={signingCam ? "Revenir à la scène" : "Signer face au téléphone"}>
                      <Hand size={19} />
                    </button>
                  )}
                  {view === "xr" && <button className="companion-round ghost" onClick={() => stage.current?.endXR()} aria-label="Quitter la réalité augmentée"><X size={19} /></button>}
                </>
              )}
            </div>
            {micOn && <small className="companion-listening">{mic.current?.detector.speaking ? "Je vous écoute…" : "Parlez quand vous voulez : je réponds à chaque pause."}</small>}
          </div>
        )}
      </div>
    </div>
  );
}
