/* eslint-disable react-refresh/only-export-components */
// A French film, signed in sync: the soundtrack is transcribed (Whisper, word times), the voice is read (stress, emotion2vec),
// each line is generated with C0 + refiners and signed when it is spoken, fast enough to end ~1.2 s after the speech (x1.5 to
// x2.5), so she is back at rest when nobody speaks. The avatar never lags behind the picture:
// a line that starts while the previous one is still signed is skipped. When nobody speaks she returns to rest.
import { createRoot } from "react-dom/client";
import { useCallback, useEffect, useRef, useState } from "react";
import { AvatarCanvas } from "../src/components/AvatarPlayer/AvatarCanvas";
import { request } from "../src/services/api";
import { generateBody } from "../src/utils/lsfRequest";
import type { EmotionWindow, MotionFile } from "../src/types/contract";
import "../src/fonts.css";
import "./filmDirect.css";

type Word = { text: string; start_ms: number; end_ms: number };
type PWord = Word & { prominence: number; emphasized: boolean; syl_per_s: number; liveliness: number };
type Analysis = { emotion_timeline?: EmotionWindow[] | null; prosody?: { pace_syl_per_s: number; words: PWord[] } | null;
  transcription?: { segments?: { start_ms: number; end_ms: number; text: string; words?: Word[] }[] } | null };
type Cue = { start: number; end: number; text: string; motion?: MotionFile; emotion?: Record<string, number> | null };
const SPEED = (() => { const v = parseFloat(new URLSearchParams(location.search).get("vitesse") ?? ""); return Number.isFinite(v) ? Math.min(3, Math.max(0.5, v)) : 1.5; })();
const BLEND_MS = 300, TAIL_MS = 1200, MAX_SPEED = 2.5;
/** Signing speed of a line: finish about TAIL_MS after it is spoken, never slower than SPEED nor faster than MAX_SPEED. */
const rateOf = (c: Cue) => Math.min(Math.max(SPEED, MAX_SPEED), Math.max(SPEED, (c.motion?.duration_ms ?? 0) / (c.end - c.start + TAIL_MS)));

/** Subtitle file (SRT or VTT) -> timed lines. */
function parseSubtitles(text: string) {
  const ms = (t: string) => { const m = t.trim().replace(",", ".").match(/(?:(\d+):)?(\d+):(\d+(?:\.\d+)?)/); return m ? ((+(m[1] ?? 0)) * 3600 + +m[2] * 60 + +m[3]) * 1000 : NaN; };
  return text.replace(/\r/g, "").split(/\n\s*\n/).flatMap((block) => {
    const lines = block.split("\n").filter((l) => l.trim() && l.trim() !== "WEBVTT");
    const at = lines.findIndex((l) => l.includes("-->"));
    if (at < 0) return [];
    const [a, b] = lines[at].split("-->").map(ms);
    const words = lines.slice(at + 1).join(" ").replace(/<[^>]+>/g, "").trim();
    return Number.isFinite(a) && Number.isFinite(b) && words && !/^\[.*\]$/.test(words) ? [{ start: Math.round(a), end: Math.round(b), text: words }] : [];
  });
}

/** Mean emotion2vec scores of the windows that overlap [a, b]. */
function emotionOf(tl: EmotionWindow[] | null | undefined, a: number, b: number) {
  const ws = (tl ?? []).filter((w) => w.end_ms > a && w.start_ms < b);
  if (!ws.length) return null;
  const out: Record<string, number> = {};
  for (const w of ws) for (const [k, v] of Object.entries(w.scores)) out[k] = (out[k] ?? 0) + v / ws.length;
  return out;
}

function FilmDirect() {
  const [videoUrl, setVideoUrl] = useState("");
  const [cues, setCues] = useState<Cue[]>([]);
  const [stage, setStage] = useState<"idle" | "listening" | "signing" | "ready">("idle");
  const [error, setError] = useState("");
  const [prepared, setPrepared] = useState(0);
  const preparedSet = useRef(new WeakSet<MotionFile>());
  const onPrepared = useCallback((m: MotionFile) => { preparedSet.current.add(m); setPrepared((n) => n + 1); }, []);
  const video = useRef<HTMLVideoElement>(null);
  const [now, setNow] = useState(0); // film time (ms)
  const play = useRef<{ cue: number; start: number; rate: number } | null>(null); // cue being signed, film time it started
  const skipped = useRef(new Set<number>());

  // With a subtitle file (?sous-titres): its lines and times are used (any film language; the voice is still read for
  // emotion). Without: the French soundtrack is transcribed.
  const [subs, setSubs] = useState<{ start: number; end: number; text: string }[] | null>(null);
  const [muted, setMuted] = useState(false);
  const load = useCallback(async (file: File, lines = subs) => {
    setError(""); setCues([]); setPrepared(0); play.current = null; skipped.current = new Set();
    setVideoUrl(URL.createObjectURL(file));
    setStage("listening");
    try {
      const form = new FormData();
      form.append("file", file, file.name);
      // with subtitles the voice analysis only adds the emotion: signing goes on without it if it fails (e.g. > 60 s)
      const ask = request<Analysis>(`/analyze-audio?include_emotion=true&include_transcription=${lines ? "false" : "true"}`, { method: "POST", body: form });
      const a = lines ? await ask.catch(() => ({} as Analysis)) : await ask;
      const segs = lines ?? (a.transcription?.segments ?? []).filter((s) => s.text.trim()).map((s) => ({ start: s.start_ms, end: s.end_ms, text: s.text }));
      const list: Cue[] = segs.map((s) => ({ start: s.start, end: s.end, text: s.text.trim(), emotion: emotionOf(a.emotion_timeline, s.start, s.end) }));
      setCues(list);
      setStage("signing");
      let referents: unknown[] = [];
      for (let i = 0; i < list.length; i++) {
        const c = list[i];
        const words = (a.prosody?.words ?? []).filter((w) => w.start_ms >= c.start - 50 && w.end_ms <= c.end + 50)
          .map((w) => ({ ...w, start_ms: w.start_ms - c.start, end_ms: w.end_ms - c.start }));
        const prosody = words.length ? { pace_syl_per_s: a.prosody!.pace_syl_per_s, words } : null;
        const motion = await request<MotionFile>("/lsf/generate", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ...generateBody(c.text, "c0", true, true, null, c.emotion ?? null, prosody ? { prosody } : null),
            ...(referents.length ? { referents: referents.slice(-64) } : {}) }) });
        referents = (motion.generation as { gloss_analysis?: { referents?: unknown[] } } | undefined)?.gloss_analysis?.referents ?? referents;
        list[i] = { ...c, motion };
        setCues([...list]);
      }
      setStage("ready");
    } catch (e) {
      setError(e instanceof Error ? e.message : "Traitement impossible.");
      setStage("idle");
    }
  }, [subs]);

  // Clock = the film: the avatar follows play, pause and seeking.
  useEffect(() => {
    let raf = 0;
    const tick = () => { if (video.current) setNow(video.current.currentTime * 1000); raf = requestAnimationFrame(tick); };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, []);

  // Which line is signed now: a line starts when it is spoken if the avatar is free, else it is skipped.
  const cur = play.current;
  const curMotion = cur ? cues[cur.cue]?.motion : undefined;
  const signedFor = curMotion && cur ? curMotion.duration_ms / cur.rate : 0;
  if (cur && (now < cur.start - 200 || now > cur.start + signedFor)) play.current = null; // finished, or the film jumped back
  if (!play.current) {
    const i = cues.findIndex((c, k) => c.motion && now >= c.start && now < c.end && !skipped.current.has(k) && preparedSet.current.has(c.motion));
    if (i >= 0) {
      play.current = { cue: i, start: cues[i].start, rate: rateOf(cues[i]) };
      cues.forEach((c, k) => { if (k < i && now >= c.start) skipped.current.add(k); });
      skipped.current.add(i);
    }
  }
  const p = play.current, motion = p ? cues[p.cue]?.motion ?? null : null;
  const t = p ? Math.max(0, now - p.start) * p.rate : 0;
  const spoken = cues.find((c) => now >= c.start && now < c.end + 300);
  const onSeek = () => { skipped.current = new Set(cues.map((c, k) => (c.start < (video.current?.currentTime ?? 0) * 1000 - 300 ? k : -1)).filter((k) => k >= 0)); play.current = null; };
  const motions = cues.flatMap((c) => (c.motion ? [c.motion] : []));
  const ready = stage === "ready" && prepared >= motions.length;
  const status = stage === "listening" ? "Écoute de la bande-son…" : stage === "signing" ? `Traduction ${motions.length} / ${cues.length}…`
    : stage === "ready" ? (ready ? (motion ? "Je signe" : "Au repos") : "Préparation de l’avatar…") : "";

  return (
    <main className="fd">
      <header>
        <img src="/favicon.svg" alt="" width={38} height={38} />
        <h1>SignePourMoi <span>· Film en direct</span></h1>
        <label className="pick ghost">
          <input type="file" accept=".srt,.vtt,text/vtt" hidden onChange={(e) => { const f = e.target.files?.[0]; if (f) void f.text().then((t) => setSubs(parseSubtitles(t))); }} />
          {subs ? `Sous-titres : ${subs.length} répliques` : "Sous-titres (facultatif)"}
        </label>
        <label className="mute"><input type="checkbox" checked={muted} onChange={(e) => setMuted(e.target.checked)} /> Son coupé</label>
        <label className="pick">
          <input type="file" accept="video/*" hidden onChange={(e) => e.target.files?.[0] && void load(e.target.files[0])} />
          {subs ? "Choisir le film" : "Choisir un film (français)"}
        </label>
        <span className="status">{status}</span>
      </header>
      <section className="stage">
        <div className="col">
          <div className="film">
            {videoUrl ? <video ref={video} src={videoUrl} muted={muted} controls={!ready ? false : true} onSeeked={onSeek} playsInline /> : <div className="empty">Choisissez un extrait de film en français</div>}
          </div>
          {/* subtitles under the pictures, never over the signing hands */}
          <p className="sub">{spoken?.text ?? ""}</p>
        </div>
        <div className="col">
          <div className="avatar">
            <AvatarCanvas avatarCredit="" motion={motion} currentTimeMs={t} blendMs={BLEND_MS} expressiveModulation
              emotion={p ? cues[p.cue]?.emotion ?? null : null} prewarm={motions} prewarmBudgetMs={12} onPrepared={onPrepared} />
          </div>
          <p className="sub signed">{p ? cues[p.cue]?.text : ""}</p>
        </div>
      </section>
      {error && <p role="alert" className="error">{error}</p>}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<FilmDirect />);
