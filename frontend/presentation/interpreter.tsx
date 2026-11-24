// Live LSF interpreter box: listens, transcribes, signs. Used by the presentation (bottom-right corner) and by the
// interpreter-only page (/interprete/, the whole window). Same services as the Studio: live microphone stream ->
// /live/transcribe (Whisper, every ~0.7 s) -> sentences -> /lsf/generate (C0) -> avatar.
import { useCallback, useEffect, useRef, useState } from "react";
import { AvatarCanvas } from "../src/components/AvatarPlayer/AvatarCanvas";
import { request } from "../src/services/api";
import { generateBody } from "../src/utils/lsfRequest";
import { encodeWav, openPhraseMic, type PhraseDetector } from "../src/utils/phraseDetector";
import { openLiveStream, type Word } from "./liveStream";
import { checkSentence } from "./frenchFilter";
import { analyseProsody, sliceProsody, type Frame, type Prosody } from "./prosody";
import type { EmotionWindow, MotionFile } from "../src/types/contract";
import "./interpreter.css";

// One heard phrase = one line; long phrases are signed as several parts, one after the other.
// emotion: the voice's emotion2vec scores for this sentence; timeline: its windows, stretched to the part's signing time.
type Part = { text: string; motion?: MotionFile; emotion?: Scores | null; timeline?: EmotionWindow[] | null };
type Scores = Record<string, number>;
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

// C0 + refiners (server), LSF order, grammatical face, pointing spans, the stressed words (voice and written cues) and the
// voice emotion (emotion2vec: speed and size of the signs on the server, the face and posture on the avatar).
// Who / what was named in the previous sentences (the server's referent table), so a pronoun in a later sentence
// ("Marie présente le projet." ... "Elle est fière") still points to its referent. Same as the Film mode's scene.
// Forgotten after a long silence. The signing queue is serial, so the order is the speaking order.
const SCENE_GAP_MS = 90000;
const scene: { referents: unknown[]; last: number } = { referents: [], last: 0 };
type Feel = { emotion?: Scores | null; timeline?: EmotionWindow[] | null };
async function generate(text: string, prosody?: Prosody | null, feel: Feel = {}) {
  if (performance.now() - scene.last > SCENE_GAP_MS) scene.referents = [];
  const motion = await request<MotionFile>("/lsf/generate", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...generateBody(text, "c0", true, true, null, feel.emotion ?? null,
      prosody || feel.timeline?.length ? { prosody: prosody ?? null, emotion_timeline: feel.timeline ?? null } : null),
      ...(scene.referents.length ? { referents: scene.referents.slice(-64) } : {}) }),
  });
  scene.referents = (motion.generation as { gloss_analysis?: { referents?: unknown[] } } | undefined)?.gloss_analysis?.referents ?? scene.referents;
  scene.last = performance.now();
  return motion;
}
/** Generate one part; if C0 used all its tokens (the end may be missing), sign it as two halves instead. */
async function generateWhole(text: string, add: (part: Part) => void, prosody?: Prosody | null, feel: Feel & { speechMs?: number } = {}): Promise<void> {
  const motion = await generate(text, prosody, feel);
  const w = words(text);
  if (((motion as { generation?: { tokens?: number } }).generation?.tokens ?? 0) >= C0_MAX_TOKENS && w.length > 3) {
    const mid = Math.ceil(w.length / 2);
    await generateWhole(w.slice(0, mid).join(" "), add, null, { emotion: feel.emotion });
    await generateWhole(w.slice(mid).join(" "), add, null, { emotion: feel.emotion });
  } else {
    // the face follows the voice's emotion over the sentence: speech time -> signing time
    const k = feel.speechMs ? motion.duration_ms / feel.speechMs : 0;
    const timeline = k && feel.timeline?.length
      ? feel.timeline.map((x) => ({ ...x, start_ms: Math.round(x.start_ms * k), end_ms: Math.round(x.end_ms * k) })) : null;
    add({ text, motion, emotion: feel.emotion ?? null, timeline });
  }
}

/** Emotion windows inside [from, to] ms of the sentence audio, with times from `from`. */
function sliceTimeline(timeline: EmotionWindow[] | null, from: number, to: number) {
  if (!timeline?.length) return null;
  const out = timeline.filter((x) => x.end_ms > from && x.start_ms < to)
    .map((x) => ({ ...x, start_ms: Math.max(0, x.start_ms - from), end_ms: Math.min(to, x.end_ms) - from }));
  return out.length ? out.slice(0, 64) : null;
}

/** The voice of a sentence, one /analyze-audio call (no transcription): stressed words (acoustic frames + the stream's
 * word times) and emotion2vec (scores for the sentence + ~2 s windows). Model hypotheses about the voice, not a measure. */
const MIN_EMOTION_SPEECH_MS = 1500;
type VoiceInfo = { prosody: Prosody | null; emotion: Scores | null; timeline: EmotionWindow[] | null };
async function analyseVoice(voice: { wav: ArrayBuffer; words: Word[] }): Promise<VoiceInfo> {
  try {
    const form = new FormData();
    form.append("file", new Blob([voice.wav], { type: "audio/wav" }), "sentence.wav");
    const r = await request<{ frames: Frame[]; emotion_hypothesis?: Record<string, number | string> | null; emotion_timeline?: EmotionWindow[] | null }>(
      "/analyze-audio?include_emotion=true", { method: "POST", body: form });
    const scores = Object.fromEntries(Object.entries(r.emotion_hypothesis ?? {}).filter((e): e is [string, number] => typeof e[1] === "number"));
    const prosody = analyseProsody(r.frames, voice.words);
    // under ~1.5 s of speech emotion2vec guesses (a 1 s fragment came out "angry 97 %"): no emotion then
    const speech = voice.words.length ? voice.words[voice.words.length - 1].end_ms - voice.words[0].start_ms : 0;
    if (speech < MIN_EMOTION_SPEECH_MS || !Object.keys(scores).length) return { prosody, emotion: null, timeline: null };
    return { prosody, emotion: scores, timeline: r.emotion_timeline ?? null };
  } catch {
    return { prosody: null, emotion: null, timeline: null }; // the written cues ("!", "très", capitals) still apply
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
      if (checkSentence(text).keep) sign(text);
    });
  });
  return { detector: mic.detector, stop: async () => mic.stop() };
}

/** Live interpreter box: listens, transcribes, signs each phrase in speaking order. */
export function Interpreter({ signRequest = null, standalone = false }: { signRequest?: { text: string; n: number } | null; standalone?: boolean }) {
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
        const heard = voice ? await analyseVoice(voice) : { prosody: null, emotion: null, timeline: null };
        const { prosody } = heard;
        const pieces = splitText(text);
        // pieces are consecutive words of the sentence: each gets its own words (and emotion windows) of the voice analysis
        const aligned = prosody && pieces.reduce((n, x) => n + words(x).length, 0) === prosody.words.length;
        let at = 0;
        for (const piece of pieces) {
          const n = words(piece).length;
          const span = aligned ? [prosody.words[at].start_ms, prosody.words[at + n - 1].end_ms] : voice?.words.length
            ? [voice.words[0].start_ms, voice.words[voice.words.length - 1].end_ms] : null;
          const feel = { emotion: heard.emotion, timeline: span && pieces.length > 1 ? sliceTimeline(heard.timeline, span[0], span[1]) : heard.timeline,
            speechMs: span ? Math.max(1, span[1] - span[0]) : undefined };
          await generateWhole(piece, (part) => update((l) => ({ parts: [...l.parts, part] })), aligned ? sliceProsody(prosody, at, at + n) : null, feel);
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
      else if (standalone) { if (k === "f") void (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen()); }
      else if (k === "h") setHidden((h) => !h);
      else if (k === "z") setSize((s) => (s === "s" ? "m" : s === "m" ? "l" : "s"));
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [toggle, standalone]);

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
    <aside className={`interp ${standalone ? "standalone" : `size-${size}`}${hidden ? " hidden" : ""}`} aria-label="Interprète LSF">
      <header>
        <span className={`dot${on ? " on" : ""}${speaking ? " speaking" : ""}`} />
        <strong>Interprète LSF</strong>
        <span className="status">{status}</span>
      </header>
      <div className="interp-avatar">
        <AvatarCanvas avatarCredit="" background={standalone ? "#ffffff" : undefined} motion={signing ? current!.motion! : null} currentTimeMs={t} blendMs={BLEND_MS} expressiveModulation
          emotion={signing ? current!.emotion ?? null : null} emotionTimeline={signing ? current!.timeline ?? null : null}
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
