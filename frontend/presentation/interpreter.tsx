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
// Between sentences / chunks: about 0.1 s (REST_MS) with a short fade (BLEND_MS) toward the rest pose, then the next one
// starts at once (LEAD_MS). User choice: short gaps, faster signing.
const BLEND_MS = 300, LEAD_MS = 0, REST_MS = 100;
// Signing speed: x1.5 (user choice), or the address ?vitesse=1.2 (0.5-3).
// Fast mode (address ?rapide=1, the :5207 endpoint): the signing keeps up with the speaker.
//  1. Grouping: sentences heard while the avatar is busy wait, then go to C0 together in ONE call (C0 writes about 5 s
//     for any sentence, even one word: one call for three short sentences is 35-65 % shorter than three calls).
//  2. Held parts faster: frames where the hands barely move (C0 marks about a third of its steps as holds) play up to
//     x3; real signing keeps the base speed.
const RAPIDE = new URLSearchParams(location.search).get("rapide") === "1";
const WARP_REF = 0.35, WARP_MAX = 3; // wrist speed (m/s) under which a part is played faster; at most x3
const FLUSH_MS = 1500;               // grouping: generate the waiting sentences when the avatar has this much left
const GROUP_WORDS = 16;              // grouped text is cut in pieces of at most this many words (C0 token limit)
type Warp = { at: Float32Array; dt: number; total: number };
const warps = new WeakMap<MotionFile, Warp>();
/** Display time of each frame: base speed, and with ?rapide the slow (held) frames up to x3 faster. */
function warpOf(m: MotionFile): Warp {
  const cached = warps.get(m);
  if (cached) return cached;
  const n = m.frames.length, dt = 1000 / (m.fps || 25), at = new Float32Array(n + 1);
  const speed = new Float32Array(n);
  for (let i = 1; i < n; i++) {
    const a = m.frames[i - 1].joints, b = m.frames[i].joints;
    let v = 0;
    for (const k of ["left_wrist", "right_wrist"]) {
      const p = a[k], q = b[k];
      if (p && q) v = Math.max(v, Math.hypot(q[0] - p[0], q[1] - p[1], q[2] - p[2]) / (dt / 1000));
    }
    speed[i] = v;
  }
  for (let i = 0; i < n; i++) {
    let sum = 0, k = 0; // 5-frame average: no jitter in the rate
    for (let j = Math.max(1, i - 2); j <= Math.min(n - 1, i + 2); j++) { sum += speed[j]; k++; }
    const v = k ? sum / k : WARP_REF;
    const rate = RAPIDE ? Math.min(WARP_MAX, Math.max(1, WARP_REF / Math.max(v, 1e-3))) : 1;
    at[i + 1] = at[i] + dt / (SIGN_SPEED * rate);
  }
  const w = { at, dt, total: at[n] };
  warps.set(m, w);
  return w;
}
/** Motion time (ms) to show after `elapsed` ms of display. */
function motionTime(m: MotionFile, elapsed: number) {
  const { at, dt } = warpOf(m);
  if (elapsed <= 0) return 0;
  let lo = 0, hi = at.length - 2;
  while (lo < hi) { const mid = (lo + hi + 1) >> 1; if (at[mid] <= elapsed) lo = mid; else hi = mid - 1; }
  const f = Math.min(1, (elapsed - at[lo]) / Math.max(1e-6, at[lo + 1] - at[lo]));
  return (lo + f) * dt;
}
/** Cut a grouped text in balanced pieces of at most GROUP_WORDS words (sentence ends are not cut points here). */
function chunkWords(text: string) {
  const w = words(text), n = Math.ceil(w.length / GROUP_WORDS), size = Math.ceil(w.length / Math.max(1, n));
  const out: string[] = [];
  for (let i = 0; i < w.length; i += size) out.push(w.slice(i, i + size).join(" "));
  return out;
}

const SIGN_SPEED = (() => {
  const v = parseFloat(new URLSearchParams(location.search).get("vitesse") ?? "");
  return Number.isFinite(v) ? Math.min(3, Math.max(0.5, v)) : 1.5;
})();
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
// Under this much speech emotion2vec guesses (a 1 s fragment once came out "angry 97 %"): no emotion. Address
// ?emotion_min=1000 lowers it (e.g. to show a short, clearly emotional line).
const MIN_EMOTION_SPEECH_MS = (() => { const v = parseInt(new URLSearchParams(location.search).get("emotion_min") ?? "", 10); return Number.isFinite(v) ? Math.max(500, v) : 1500; })();
// Manual emotion (address ?emotion=happy|sad|angry|surprised|fearful|neutral, for demos): replaces the voice's emotion
// for every sentence (signing speed / size and face). Without it, the emotion comes from the voice (emotion2vec).
const FORCED_EMOTION = (() => { const e = new URLSearchParams(location.search).get("emotion"); return e ? { [e]: 1 } as Scores : null; })();
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

/** Why the microphone did not open, in words the presenter can act on. */
function micProblem(e: unknown) {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia)
    return `Micro bloqué : le navigateur n’autorise le micro qu’en https. Ouvrez https://${location.hostname}:5206/interprete/ (ou localhost sur ce PC).`;
  const name = (e as { name?: string })?.name;
  if (name === "NotAllowedError" || name === "SecurityError")
    return "Micro refusé : autorisez le micro pour ce site (icône du cadenas à gauche de l’adresse), puis réessayez.";
  if (name === "NotFoundError" || name === "OverconstrainedError") return "Aucun micro détecté sur cet appareil.";
  if (name === "NotReadableError") return "Le micro est déjà utilisé par une autre application.";
  return "Micro indisponible.";
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
  // Microphone choice (remembered): the browser's default may be a microphone that is not plugged in.
  const [mics, setMics] = useState<MediaDeviceInfo[]>([]);
  const [micId, setMicId] = useState(() => localStorage.getItem("spm-mic") ?? "");
  const listMics = useCallback(async () => {
    try { setMics((await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput" && d.deviceId && d.deviceId !== "default" && d.deviceId !== "communications")); } catch { /* no device list */ }
  }, []);
  useEffect(() => {
    void listMics();
    navigator.mediaDevices?.addEventListener?.("devicechange", listMics);
    return () => navigator.mediaDevices?.removeEventListener?.("devicechange", listMics);
  }, [listMics]);
  const [size, setSize] = useState<"s" | "m" | "l">("m");
  const [hidden, setHidden] = useState(false);
  // Two independent queues: Whisper never waits for the signing, the signing never waits for Whisper.
  const signChain = useRef(Promise.resolve());
  const epoch = useRef(0);
  const muted = useRef(false); // after S: the mic's last flush is thrown away
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
    const finished = !part?.motion || now - playing!.started > LEAD_MS + warpOf(part.motion).total + REST_MS;
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
  type Voice = { wav: ArrayBuffer; words: Word[] };
  /** One heard sentence (voice) or, in fast mode, a group of sentences (voices) -> pieces -> C0. */
  const enqueue = useCallback((text: string, voice?: Voice, group?: Voice[]) => {
    const index = count.current++;
    // after a stop (key S) the epoch changes: work queued before it is skipped and its results are thrown away
    const e = epoch.current, live = () => epoch.current === e;
    setLines((prev) => [...prev, { text, parts: [] }]);
    const update = (patch: (l: Line) => Partial<Line>) => { if (live()) setLines((prev) => prev.map((l, i) => (i === index ? { ...l, ...patch(l) } : l))); };
    setBusy((b) => b + 1);
    signChain.current = signChain.current.then(async () => {
      if (!live()) return;
      try {
        let heard: VoiceInfo = voice ? await analyseVoice(voice) : { prosody: null, emotion: null, timeline: null };
        if (group) { // grouped sentences: their voices' emotions are averaged; word stress is not aligned across them
          const all = await Promise.all(group.map(analyseVoice));
          const scores = all.flatMap((x) => (x.emotion ? [x.emotion] : []));
          const mean: Record<string, number> = {};
          for (const sc of scores) for (const [k, v] of Object.entries(sc)) mean[k] = (mean[k] ?? 0) + v / scores.length;
          heard = { prosody: null, emotion: scores.length ? mean : null, timeline: null };
        }
        if (FORCED_EMOTION) heard = { ...heard, emotion: FORCED_EMOTION, timeline: null };
        const { prosody } = heard;
        const pieces = group ? chunkWords(text) : splitText(text);
        // pieces are consecutive words of the sentence: each gets its own words (and emotion windows) of the voice analysis
        const aligned = prosody && pieces.reduce((n, x) => n + words(x).length, 0) === prosody.words.length;
        let at = 0;
        for (const piece of pieces) {
          if (!live()) return; // stopped (key S) between two pieces
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
      } catch (err) {
        update(() => ({ dropped: true }));
        if (live()) setError(err instanceof Error ? err.message : "Traitement impossible.");
      } finally {
        if (live()) setBusy((b) => b - 1);
      }
    });
  }, []);

  // Fast mode: sentences heard while the avatar is busy wait here and are generated together.
  const waiting = useRef<{ text: string; voice: Voice }[]>([]);
  const [waitingCount, setWaitingCount] = useState(0);
  const heardSentence = useCallback((text: string, voice: Voice) => {
    if (muted.current) return;
    if (!RAPIDE) { enqueue(text, voice); return; }
    waiting.current.push({ text, voice });
    setWaitingCount(waiting.current.length);
  }, [enqueue]);

  // Fast mode: when the avatar has FLUSH_MS or less left to sign (or is idle) and the generator is free, everything
  // heard meanwhile goes to C0 as one text (a single sentence keeps its own word stress).
  useEffect(() => {
    if (!RAPIDE || !waiting.current.length || busy > 0) return;
    let left = 0;
    lines.forEach((l, i) => l.parts.forEach((x, j) => {
      if (!x.motion) return;
      if (playing && i === playing.index && j === playing.part) left += Math.max(0, LEAD_MS + warpOf(x.motion).total - (now - playing.started));
      else if (!playing || i > playing.index || (i === playing.index && j > playing.part)) left += warpOf(x.motion).total + REST_MS;
    }));
    if (left > FLUSH_MS) return;
    const items = waiting.current.splice(0);
    setWaitingCount(0);
    if (items.length === 1) enqueue(items[0].text, items[0].voice);
    else enqueue(items.map((x) => x.text).join(" "), undefined, items.map((x) => x.voice));
  }, [now, busy, playing, lines, enqueue]);

  /** Key S (interpreter page): stop everything at once: the mic, the signing in progress and every phrase waiting;
   * the avatar goes back to rest. M starts listening again. */
  const stopAll = useCallback(() => {
    muted.current = true;
    waiting.current = [];
    setWaitingCount(0);
    epoch.current++;
    stop.current();
    count.current = 0;
    setLines([]);
    setPlaying(null);
    setBusy(0);
  }, []);

  useEffect(() => {
    if (signRequest) enqueue(signRequest.text);
  }, [signRequest, enqueue]);

  const start = useCallback(async () => {
    muted.current = false;
    setError("");
    try {
      // Fallback (address ?ecoute=pauses): the earlier mode, one phrase per pause, transcribed once.
      const mic = new URLSearchParams(location.search).get("ecoute") === "pauses" ? await openPausePhrases((text) => { if (!muted.current) enqueue(text); }) : await openLiveStream({
        transcribe: async (wav) => {
          const form = new FormData();
          form.append("file", new Blob([wav], { type: "audio/wav" }), "live.wav");
          const r = await request<{ segments?: { words?: Word[] }[] }>("/live/transcribe", { method: "POST", body: form });
          return (r.segments ?? []).flatMap((x) => x.words ?? []);
        },
        deviceId: micId || undefined,
        onSentence: heardSentence,
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
      void listMics(); // names are known once the mic is allowed
    } catch (e) {
      setError(micProblem(e));
      void listMics();
    }
  }, [enqueue, heardSentence, micId, listMics]);

  const toggle = useCallback(() => (on ? stop.current() : ready === "ready" ? void start() : undefined), [on, ready, start]);

  // Keyboard: M = micro, H = hide/show the box, Z = box size; interpreter page: S = stop mic + signing, F = full screen.
  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === "m") toggle();
      else if (standalone) {
        if (k === "s") stopAll();
        else if (k === "f") void (document.fullscreenElement ? document.exitFullscreen() : document.documentElement.requestFullscreen());
      }
      else if (k === "h") setHidden((h) => !h);
      else if (k === "z") setSize((s) => (s === "s" ? "m" : s === "m" ? "l" : "s"));
    };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [toggle, standalone, stopAll]);

  const current = playing ? lines[playing.index]?.parts[playing.part] : undefined;
  const elapsed = playing ? now - playing.started : 0;
  const signing = !!current?.motion && elapsed < LEAD_MS + warpOf(current.motion).total;
  const t = signing ? motionTime(current!.motion!, elapsed - LEAD_MS) : 0;
  const speaking = !!detector.current?.speaking;
  const level = detector.current ? Math.min(1, detector.current.level * 12) : 0;
  // What the avatar must derive next, in playing order (the current part first while it is not ready).
  const upcoming: MotionFile[] = [];
  lines.forEach((l, i) => {
    if (playing && i < playing.index) return;
    l.parts.forEach((x, j) => { if (x.motion && (!playing || i > playing.index || j >= playing.part)) upcoming.push(x.motion); });
  });
  // Both sides at once: what the avatar does, and what the ear does.
  const hand = (signing ? "Je signe" : busy ? "Traduction…" : "") + (waitingCount ? ` (${waitingCount} en attente)` : "");
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
        {!on && mics.length > 1 && (
          <select className="mic-pick" value={micId} title="Micro" aria-label="Micro"
            onChange={(e) => { setMicId(e.target.value); localStorage.setItem("spm-mic", e.target.value); }}>
            <option value="">Micro par défaut</option>
            {mics.map((m, i) => <option key={m.deviceId} value={m.deviceId}>{m.label || `Micro ${i + 1}`}</option>)}
          </select>
        )}
      </footer>
      {error && <p role="alert" className="interp-error">{error}</p>}
    </aside>
  );
}
