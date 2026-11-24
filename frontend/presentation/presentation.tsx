/* eslint-disable react-refresh/only-export-components */
// Presentation of SignaCulture: full-screen slides + a live MetaHuman interpreter in the bottom-right corner.
// Same services as the Studio: live microphone stream -> /live/transcribe (Whisper, every ~0.7 s) -> sentences ->
// /lsf/generate (C0) -> avatar. The microphone never stops while a sentence is signed.
import { createRoot } from "react-dom/client";
import { useCallback, useEffect, useRef, useState } from "react";
import { AvatarCanvas } from "../src/components/AvatarPlayer/AvatarCanvas";
import { request } from "../src/services/api";
import { generateBody } from "../src/utils/lsfRequest";
import { encodeWav, openPhraseMic, type PhraseDetector } from "../src/utils/phraseDetector";
import { isNoise, openLiveStream, type Word } from "./liveStream";
import { analyseProsody, sliceProsody, type Frame, type Prosody } from "./prosody";
import type { MotionFile } from "../src/types/contract";
import { SLIDES } from "./slides";
import "../src/fonts.css";
import "./presentation.css";

// One heard phrase = one line; long phrases are signed as several parts, one after the other.
type Part = { text: string; motion?: MotionFile };
type Line = { text: string; parts: Part[]; done?: boolean; dropped?: boolean; shown?: boolean };
type Ready = "loading" | "ready" | "off";
// Between sentences the avatar returns to its rest pose: fade in from rest holding the first frame (LEAD_MS),
// sign, fade back to rest and stay there (REST_MS) before the next sentence. BLEND_MS = length of each fade.
const BLEND_MS = 600, LEAD_MS = 600, REST_MS = 900;
// Motions are derived for the avatar a few ms per frame (never a stall), signing or not.
const PREP_BUDGET_SIGNING_MS = 5, PREP_BUDGET_IDLE_MS = 12;
// C0 writes at most 96 tokens (about 15 s of signing): a longer text loses its end. Parts stay well below that.
const C0_MAX_TOKENS = 96, MAX_WORDS = 16;

const words = (t: string) => t.split(/\s+/).filter(Boolean);
/** Sentences, then comma groups, then balanced word chunks: every part at most MAX_WORDS words. */
function splitText(text: string): string[] {
  const out: string[] = [];
  for (const sentence of text.split(/(?<=[.!?…;:])\s+/)) {
    if (!words(sentence).length) continue;
    // comma groups, merged back together while they fit
    const groups: string[] = [];
    for (const g of sentence.split(/(?<=,)\s+/)) {
      const last = groups.length - 1;
      if (last >= 0 && words(groups[last]).length + words(g).length <= MAX_WORDS) groups[last] += " " + g;
      else groups.push(g);
    }
    for (const g of groups) {
      const w = words(g), n = Math.ceil(w.length / MAX_WORDS), size = Math.ceil(w.length / n);
      for (let i = 0; i < w.length; i += size) out.push(w.slice(i, i + size).join(" "));
    }
  }
  return out;
}

// C0 + refiners (server), LSF order, grammatical face, pointing spans, and the stressed words (voice and written cues).
// No voice emotion here on purpose.
// Who / what was named in the previous sentences (the server's referent table), so a pronoun in a later sentence
// ("Marie présente le projet." ... "Elle est fière") still points to its referent. Same as the Film mode's scene.
// Forgotten after a long silence. The signing queue is serial, so the order is the speaking order.
const SCENE_GAP_MS = 90000;
const scene: { referents: unknown[]; last: number } = { referents: [], last: 0 };
async function generate(text: string, prosody?: Prosody | null) {
  if (performance.now() - scene.last > SCENE_GAP_MS) scene.referents = [];
  const motion = await request<MotionFile>("/lsf/generate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...generateBody(text, "c0", true, true, null, null, prosody ? { prosody } : null),
      ...(scene.referents.length ? { referents: scene.referents.slice(-64) } : {}) }),
  });
  scene.referents = (motion.generation as { gloss_analysis?: { referents?: unknown[] } } | undefined)?.gloss_analysis?.referents ?? scene.referents;
  scene.last = performance.now();
  return motion;
}
/** Generate one part; if C0 used all its tokens (the end may be missing), sign it as two halves instead. */
async function generateWhole(text: string, add: (part: Part) => void, prosody?: Prosody | null): Promise<void> {
  const motion = await generate(text, prosody);
  const w = words(text);
  if (((motion as { generation?: { tokens?: number } }).generation?.tokens ?? 0) >= C0_MAX_TOKENS && w.length > 3) {
    const mid = Math.ceil(w.length / 2);
    await generateWhole(w.slice(0, mid).join(" "), add);
    await generateWhole(w.slice(mid).join(" "), add);
  } else add({ text, motion });
}

/** Stressed words of a sentence: acoustic frames from /analyze-audio (fast, no transcription) + the stream's word times. */
async function voiceStress(voice: { wav: ArrayBuffer; words: Word[] }): Promise<Prosody | null> {
  try {
    const form = new FormData();
    form.append("file", new Blob([voice.wav], { type: "audio/wav" }), "sentence.wav");
    const { frames } = await request<{ frames: Frame[] }>("/analyze-audio", { method: "POST", body: form });
    return analyseProsody(frames, voice.words);
  } catch {
    return null; // the written cues ("!", "très", capitals) still apply
  }
}

/** Earlier listening mode: the browser cuts at pauses, each phrase is transcribed once (serial), then signed. */
async function openPausePhrases(sign: (text: string) => void) {
  let chain = Promise.resolve();
  const mic = await openPhraseMic((samples, rate) => {
    const wav = encodeWav(samples, rate);
    chain = chain.then(async () => {
      const form = new FormData();
      form.append("file", new Blob([wav], { type: "audio/wav" }), "phrase.wav");
      const text = (await request<{ text: string }>("/live/transcribe", { method: "POST", body: form }).catch(() => ({ text: "" }))).text.trim();
      if (!isNoise(text)) sign(text);
    });
  });
  return { detector: mic.detector, stop: async () => mic.stop() };
}

const slideFromHash = () => {
  const n = parseInt(location.hash.slice(1), 10);
  return Number.isFinite(n) ? Math.max(0, Math.min(SLIDES.length - 1, n - 1)) : 0;
};

/** Live interpreter box: listens, transcribes, signs each phrase in speaking order. */
function Interpreter({ signRequest }: { signRequest: { text: string; n: number } | null }) {
  const [ready, setReady] = useState<Ready>("loading");
  const [on, setOn] = useState(false);
  const [busy, setBusy] = useState(0);       // phrases in the sign generator
  const [lines, setLines] = useState<Line[]>([]);
  const [playing, setPlaying] = useState<{ index: number; part: number; started: number } | null>(null);
  const [now, setNow] = useState(() => performance.now());
  // Motions the avatar has derived (sliced prewarm): a part starts only once ready, so nothing is skipped.
  const prepared = useRef(new WeakSet<MotionFile>());
  const [, setPreparedTick] = useState(0);
  const onPrepared = useCallback((m: MotionFile) => { prepared.current.add(m); setPreparedTick((n) => n + 1); }, []);
  const [error, setError] = useState("");
  const [size, setSize] = useState<"s" | "m" | "l">("m");
  const [hidden, setHidden] = useState(false);
  // Two independent queues: Whisper never waits for the signing, the signing never waits for Whisper.
  const signChain = useRef(Promise.resolve());
  const count = useRef(0);
  const detector = useRef<PhraseDetector | null>(null);
  const stop = useRef<() => void>(() => {});

  // Load Whisper + C0 in advance, so the first phrase is fast.
  useEffect(() => {
    request("/live/warmup?engine=c0", { method: "POST", signal: AbortSignal.timeout(280000) })
      .then(() => setReady("ready"))
      .catch(() => setReady("off"));
    return () => stop.current();
  }, []);

  useEffect(() => {
    let raf = 0;
    const tick = () => { setNow(performance.now()); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Next part as soon as the current one is finished (signed + back to rest). Phrases play strictly in speaking
  // order: a phrase still in Whisper or in the generator is waited for, so nothing is skipped.
  useEffect(() => {
    const line = playing ? lines[playing.index] : undefined;
    const part = line?.parts[playing!.part];
    const finished = !part?.motion || now - playing!.started > LEAD_MS + part.motion.duration_ms + REST_MS;
    if (!finished) return;
    if (playing && line) {
      const following = line.parts[playing.part + 1]?.motion;
      if (following) { if (prepared.current.has(following)) setPlaying({ ...playing, part: playing.part + 1, started: now }); return; }
      if (!line.done && !line.dropped) return;
    }
    let next = playing ? playing.index + 1 : 0;
    while (next < lines.length && lines[next].dropped && !lines[next].parts.length) next++; // noise, errors
    const first = lines[next]?.parts[0]?.motion;
    if (!first || !prepared.current.has(first)) return; // not transcribed / generated / derived yet: wait at rest

    setPlaying({ index: next, part: 0, started: now });
    // played phrases give their motions back (a long talk keeps memory flat)
    setLines((prev) => prev.map((l, i) => (i === next ? { ...l, shown: true } : i < next && l.parts.length ? { ...l, parts: l.parts.map((x) => ({ text: x.text })) } : l)));
  }, [now, lines, playing]);

  /** Signing queue: one heard (or slide) sentence -> parts -> C0, each part playable as soon as it is generated.
   * With the sentence's audio, its stressed words (loudness, pitch, lengthening) make those signs bigger, held, nodded. */
  const enqueue = useCallback((text: string, voice?: { wav: ArrayBuffer; words: Word[] }) => {
    const index = count.current++;
    setLines((prev) => [...prev, { text, parts: [] }]);
    const update = (patch: (l: Line) => Partial<Line>) => setLines((prev) => prev.map((l, i) => (i === index ? { ...l, ...patch(l) } : l)));
    setBusy((b) => b + 1);
    signChain.current = signChain.current.then(async () => {
      try {
        const prosody = voice ? await voiceStress(voice) : null;
        const pieces = splitText(text);
        // pieces are consecutive words of the sentence: each gets its own words of the voice analysis
        const aligned = prosody && pieces.reduce((n, x) => n + words(x).length, 0) === prosody.words.length;
        let at = 0;
        for (const piece of pieces) {
          const n = words(piece).length;
          await generateWhole(piece, (part) => update((l) => ({ parts: [...l.parts, part] })), aligned ? sliceProsody(prosody, at, at + n) : null);
          at += n;
        }
        update(() => ({ done: true }));
        setError("");
      } catch (e) {
        update(() => ({ dropped: true }));
        setError(e instanceof Error ? e.message : "Traitement impossible.");
      } finally {
        setBusy((b) => b - 1);
      }
    });
  }, []);

  useEffect(() => {
    if (signRequest) enqueue(signRequest.text);
  }, [signRequest, enqueue]);

  const start = useCallback(async () => {
    setError("");
    try {
      // Fallback (address ?ecoute=pauses): the earlier mode, one phrase per pause, transcribed once.
      const mic = new URLSearchParams(location.search).get("ecoute") === "pauses" ? await openPausePhrases(enqueue) : await openLiveStream({
        transcribe: async (wav) => {
          const form = new FormData();
          form.append("file", new Blob([wav], { type: "audio/wav" }), "live.wav");
          const r = await request<{ segments?: { words?: Word[] }[] }>("/live/transcribe", { method: "POST", body: form });
          return (r.segments ?? []).flatMap((x) => x.words ?? []);
        },
        onSentence: enqueue,
        onLive: () => {}, // heard text stays in the background: nothing is drawn over the avatar
        onError: (e) => setError(e instanceof Error ? e.message : "Transcription impossible."),
      });
      detector.current = mic.detector;
      stop.current = () => {
        void mic.stop();
        detector.current = null;
        stop.current = () => {};
        setOn(false);
      };
      setOn(true);
    } catch {
      setError("Micro refusé ou indisponible.");
    }
  }, [enqueue]);

  const toggle = useCallback(() => (on ? stop.current() : ready === "ready" ? void start() : undefined), [on, ready, start]);

  // Keyboard: M = micro, H = hide/show the box, Z = box size.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "m") toggle();
      else if (k === "h") setHidden((h) => !h);
      else if (k === "z") setSize((s) => (s === "s" ? "m" : s === "m" ? "l" : "s"));
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [toggle]);

  const current = playing ? lines[playing.index]?.parts[playing.part] : undefined;
  const elapsed = playing ? now - playing.started : 0;
  const signing = !!current?.motion && elapsed < LEAD_MS + current.motion.duration_ms;
  const t = signing ? Math.max(0, elapsed - LEAD_MS) : 0;
  const speaking = !!detector.current?.speaking;
  const level = detector.current ? Math.min(1, detector.current.level * 12) : 0;
  // What the avatar must derive next, in playing order (the current part first while it is not ready).
  const upcoming: MotionFile[] = [];
  lines.forEach((l, i) => {
    if (playing && i < playing.index) return;
    l.parts.forEach((x, j) => { if (x.motion && (!playing || i > playing.index || j >= playing.part)) upcoming.push(x.motion); });
  });
  // Both sides at once: what the avatar does, and what the ear does.
  const hand = signing ? "Je signe" : busy ? "Traduction…" : "";
  const ear = speaking ? "j’écoute" : "";
  const both = [hand, ear].filter(Boolean).join(" · ");
  const status = ready === "loading" ? "Préparation…"
    : ready === "off" ? "Reconnaissance vocale indisponible"
    : both ? both.charAt(0).toUpperCase() + both.slice(1)
    : on ? "À l’écoute" : "Micro coupé";

  return (
    <aside className={`interp size-${size}${hidden ? " hidden" : ""}`} aria-label="Interprète LSF">
      <header>
        <span className={`dot${on ? " on" : ""}${speaking ? " speaking" : ""}`} />
        <strong>Interprète LSF</strong>
        <span className="status">{status}</span>
      </header>
      <div className="interp-avatar">
        <AvatarCanvas avatarCredit="" motion={signing ? current!.motion! : null} currentTimeMs={t} blendMs={BLEND_MS} expressiveModulation
          prewarm={upcoming} prewarmBudgetMs={signing ? PREP_BUDGET_SIGNING_MS : PREP_BUDGET_IDLE_MS} onPrepared={onPrepared} />
        {/* Transcribed phrases wait in the background: a phrase appears only when the avatar signs it. */}
      </div>
      {/* Subtitle under the avatar, never over it; only while that sentence is signed (empty strip at rest). */}
      <div className="interp-caption" aria-live="polite">{signing && current ? current.text : ""}</div>
      <footer>
        <button className={`mic${on ? " on" : ""}`} disabled={ready !== "ready"} onClick={toggle} title="Micro (M)">
          {ready === "loading" ? "Chargement…" : on ? "Couper le micro" : "Activer le micro"}
        </button>
        {on && <span className="level" aria-hidden><i style={{ width: `${Math.round(level * 100)}%` }} /></span>}
      </footer>
      {error && <p role="alert" className="interp-error">{error}</p>}
    </aside>
  );
}

function Deck() {
  const [index, setIndex] = useState(slideFromHash);
  const [signRequest, setSignRequest] = useState<{ text: string; n: number } | null>(null);
  const go = useCallback((d: number) => setIndex((i) => Math.max(0, Math.min(SLIDES.length - 1, i + d))), []);
  const signSlide = useCallback(() => setSignRequest((r) => ({ text: SLIDES[index].sign, n: (r?.n ?? 0) + 1 })), [index]);

  useEffect(() => { history.replaceState(null, "", `#${index + 1}`); }, [index]);
  useEffect(() => {
    const onHash = () => setIndex(slideFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.ctrlKey || e.metaKey || e.altKey) return;
      if (["ArrowRight", "PageDown", " ", "Enter"].includes(e.key)) { e.preventDefault(); go(1); }
      else if (["ArrowLeft", "PageUp", "Backspace"].includes(e.key)) { e.preventDefault(); go(-1); }
      else if (e.key === "Home") setIndex(0);
      else if (e.key === "End") setIndex(SLIDES.length - 1);
      else if (e.key.toLowerCase() === "s") signSlide();
      else if (e.key.toLowerCase() === "f") void (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen());
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [go, signSlide]);

  const slide = SLIDES[index];
  return (
    <div className={`deck tone-${slide.tone ?? "light"}`}>
      <div className="brand">Signa<em>Culture</em></div>
      {SLIDES.map((s, i) => (
        <section key={s.id} className={`slide${i === index ? " active" : i < index ? " past" : ""}`} aria-hidden={i !== index}>
          <div className="slide-inner">
            {s.kicker && <p className="kicker">{s.kicker}</p>}
            {s.body}
          </div>
        </section>
      ))}
      <nav className="deck-nav">
        <button onClick={() => go(-1)} disabled={index === 0} aria-label="Diapositive précédente">‹</button>
        <span>{index + 1} / {SLIDES.length}</span>
        <button onClick={() => go(1)} disabled={index === SLIDES.length - 1} aria-label="Diapositive suivante">›</button>
        <button className="sign-slide" onClick={signSlide} title="Faire signer la phrase de la diapo (S)">Signer la diapo</button>
      </nav>
      <div className="progress"><i style={{ width: `${((index + 1) / SLIDES.length) * 100}%` }} /></div>
      <Interpreter signRequest={signRequest} />
    </div>
  );
}

createRoot(document.getElementById("root")!).render(<Deck />);
