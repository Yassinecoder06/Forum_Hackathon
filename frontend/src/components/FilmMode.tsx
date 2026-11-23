import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowDownToLine, Clapperboard, Loader2, Mic, MicOff, Square, Upload, Wand2 } from "lucide-react";
import { request } from "../services/api";
import type { AudioFeatures, MotionFile } from "../types/contract";
import { AvatarCanvas } from "./AvatarPlayer/AvatarCanvas";
import { activeAt, parseSubtitles, placeMotions, type Cue } from "../utils/subtitles";
import { EMOTION_LABEL, MARKER_LABEL, generateBody, voiceScores, type LsfEngine } from "../utils/lsfRequest";
import { CAPTURE_WORKLET, PhraseDetector, encodeWav } from "../utils/phraseDetector";
import { castOf, defaultCast, type CastEntry } from "../utils/speakers";
import { FRONTAL, filmSegments, locusSide, roleShiftAt, roleShiftSteps } from "../utils/roleShift";
import { CastSheet, SpeakerChip, StageDots } from "./FilmSpeakers";
import type { Referent } from "../types/contract";

type Item = { cue: Cue; motion?: MotionFile; error?: string; emotion?: Record<string, number> | null };
const clock = (ms: number) => `${Math.floor(ms / 60000).toString().padStart(2, "0")}:${((ms % 60000) / 1000).toFixed(1).padStart(4, "0")}`;
const WATERMARK = "SignaCulture · LSF";
/** A silence this long between two subtitles starts a new scene (fresh referent table). */
const SCENE_GAP_MS = 8000;
const castKey = (name: string) => `signaculture.cast.${name}`;
function savedCast(name: string): CastEntry[] {
  try {
    return JSON.parse(localStorage.getItem(castKey(name)) || "[]");
  } catch {
    return [];
  }
}

/**
 * Film mode (the pre-processed product of the project notes): a film + its French subtitles -> one LSF motion
 * per subtitle line (engines B2 / C0), played by the avatar beside the film, in sync, with the current line
 * highlighted; export of film + avatar as WebM. Live mode: microphone -> Whisper in 5 s windows -> engine -> avatar,
 * with the measured delay shown. Everything is an automatic proxy, never a validated translation.
 */
export function FilmMode({
  avatarCredit,
  engines,
  transcription,
  emotion,
}: {
  avatarCredit?: string;
  engines?: Partial<Record<LsfEngine, boolean>>;
  transcription?: boolean;
  emotion?: boolean;
}) {
  const [videoUrl, setVideoUrl] = useState(""),
    [videoName, setVideoName] = useState(""),
    [subsName, setSubsName] = useState(""),
    [items, setItems] = useState<Item[]>([]),
    [engine, setEngine] = useState<LsfEngine>("c0"),
    [lsfOrder, setLsfOrder] = useState(true),
    [nonmanual, setNonmanual] = useState(true),
    [busy, setBusy] = useState(false),
    [done, setDone] = useState(0),
    [prepared, setPrepared] = useState(0),
    [videoMs, setVideoMs] = useState(0),
    [error, setError] = useState(""),
    [exporting, setExporting] = useState(false),
    [mode, setMode] = useState<"film" | "live">("film"),
    [videoFile, setVideoFile] = useState<File | null>(null),
    [useEmotion, setUseEmotion] = useState(false),
    [phase, setPhase] = useState(""),
    [cast, setCast] = useState<CastEntry[]>([]);
  const video = useRef<HTMLVideoElement>(null),
    avatarCanvas = useRef<HTMLCanvasElement | null>(null),
    cancel = useRef(false);

  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);
  // Follow the film's clock (also while paused / seeking).
  useEffect(() => {
    let id = 0;
    const tick = () => {
      if (video.current) setVideoMs(video.current.currentTime * 1000);
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, []);

  const motions = useMemo(() => items.map((i) => i.motion).filter((m): m is MotionFile => Boolean(m)), [items]);
  const placed = useMemo(() => placeMotions(items.map((i) => i.cue), items.map((i) => i.motion?.duration_ms)), [items]);
  const ready = items.map((i) => Boolean(i.motion));
  const active = activeAt(placed, ready, videoMs);
  const activeMotion = active ? items[active.index].motion ?? null : null;
  const cueIndex = items.findIndex((i) => i.cue.start_ms <= videoMs && videoMs < i.cue.end_ms);
  const maxDelay = Math.max(0, ...placed.filter((_, i) => ready[i]).map((p) => p.delay_ms));
  // Role shift: while a speaker's line is signed, the avatar turns toward that speaker's place (cast sheet);
  // quoted / reported speech inside a line turns toward the quoted person.
  const shiftSteps = useMemo(() => {
    const sides = items.map((i) => (i.motion ? locusSide(castOf(cast, i.cue.speaker)?.locus) : null));
    const inner = items.map((i) => i.motion?.role_shift?.map((r) => ({
      start_ms: r.start_ms, end_ms: r.end_ms, side: locusSide(castOf(cast, r.label)?.locus ?? r.locus),
    })));
    return roleShiftSteps(filmSegments(placed, sides, inner));
  }, [items, placed, cast]);
  const roleShift = shiftSteps.length ? roleShiftAt(shiftSteps, videoMs) : FRONTAL;
  const signing = active && items[active.index]?.motion && videoMs >= placed[active.index].start_ms - 200 ? items[active.index] : undefined;
  const signer = castOf(cast, signing?.cue.speaker);
  const firstLine = useMemo(() => new Set(cast.map((c) => items.findIndex((i) => i.cue.speaker === c.id))), [cast, items]);
  const lines = useMemo(() => items.reduce<Record<string, number>>((n, i) => (i.cue.speaker ? { ...n, [i.cue.speaker]: (n[i.cue.speaker] ?? 0) + 1 } : n), {}), [items]);
  function editCast(next: CastEntry[]) {
    setCast(next);
    try {
      if (subsName) localStorage.setItem(castKey(subsName), JSON.stringify(next));
    } catch {
      /* the cast sheet still works for this session */
    }
  }

  async function loadSubtitles(file: File) {
    setError("");
    const cues = parseSubtitles(await file.text());
    if (!cues.length) {
      setError("Aucun sous-titre lisible (formats SRT ou WebVTT).");
      return;
    }
    setSubsName(file.name);
    setItems(cues.map((cue) => ({ cue })));
    setCast(defaultCast(cues, savedCast(file.name)));
    setDone(0);
    setPrepared(0);
  }

  async function transcribeFilm(file: File) {
    setBusy(true);
    setError("");
    try {
      const form = new FormData();
      form.append("file", file);
      const audio = await request<AudioFeatures>("/analyze-audio?include_emotion=false&include_transcription=true", { method: "POST", body: form });
      const cues = (audio.transcription?.segments ?? []).filter((s) => s.text.trim()).map((s) => ({ start_ms: s.start_ms, end_ms: s.end_ms, text: s.text.trim() }));
      if (!cues.length) throw new Error("La transcription n’a trouvé aucune phrase.");
      setSubsName("Transcription automatique");
      setItems(cues.map((cue) => ({ cue })));
      setCast([]);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Transcription impossible.");
    } finally {
      setBusy(false);
    }
  }

  async function generateAll() {
    cancel.current = false;
    setBusy(true);
    setError("");
    setDone(0);
    setPrepared(0);
    setItems((prev) => prev.map((i) => ({ cue: i.cue })));
    // Optional: the voice's emotion per subtitle line (emotion2vec on each line's audio, one model load).
    let feelings: (Record<string, number> | null)[] = items.map(() => null);
    if (useEmotion && videoFile) {
      setPhase("Analyse de l’émotion de la voix, réplique par réplique…");
      try {
        const form = new FormData();
        form.append("file", videoFile);
        form.append("cues", JSON.stringify(items.map((i) => ({ start_ms: i.cue.start_ms, end_ms: i.cue.end_ms }))));
        const result = await request<{ items: (Record<string, number> | null)[] }>("/analyze-audio/emotion-segments", { method: "POST", body: form });
        feelings = result.items.map((s) => voiceScores(s));
      } catch (e) {
        setError((e instanceof Error ? e.message : "Analyse émotionnelle impossible") + " — génération sans émotion.");
      }
      setPhase("");
    }
    // One scene = subtitles without a long silence: they share the referent table (who "il" / "elle" is, loci).
    let scene: Referent[] = [];
    let recent: CastEntry[] = [];                     // speakers of the scene, most recent first
    for (let k = 0; k < items.length; k++) {
      if (cancel.current) break;
      const cue = items[k].cue;
      if (k && cue.start_ms - items[k - 1].cue.end_ms > SCENE_GAP_MS) [scene, recent] = [[], []];
      const who = castOf(cast, cue.speaker);
      const other = who && recent.find((c) => c.id !== who.id);   // "tu" = the last other speaker
      const voice = who ? { speaker: { label: who.label, ...(who.locus !== "N" ? { locus: who.locus } : {}) } } : {};
      const to = other ? { addressee: { label: other.label, ...(other.locus !== "N" ? { locus: other.locus } : {}) } } : {};
      try {
        const motion = await request<MotionFile>("/lsf/generate", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            ...generateBody(cue.text.slice(0, 400), engine, lsfOrder, nonmanual, null, feelings[k]),
            ...voice, ...to, ...(scene.length ? { referents: scene.slice(-64) } : {}),
          }),
        });
        scene = motion.generation?.gloss_analysis?.referents ?? scene;
        if (who) recent = [who, ...recent.filter((c) => c.id !== who.id)];
        setItems((prev) => prev.map((i, j) => (j === k ? { cue: i.cue, motion, emotion: feelings[k] } : i)));
      } catch (e) {
        const message = e instanceof Error ? e.message : "échec";
        setItems((prev) => prev.map((i, j) => (j === k ? { cue: i.cue, error: message, emotion: feelings[k] } : i)));
        if (/indisponible|503/.test(message)) {
          setError(message);
          break;
        }
      }
      setDone(k + 1);
    }
    setBusy(false);
  }

  async function exportFilm() {
    const v = video.current, avatar = avatarCanvas.current;
    if (!v || !avatar || exporting) return;
    setExporting(true);
    setError("");
    try {
      const out = document.createElement("canvas");
      out.width = 1280;
      out.height = 720;
      const ctx = out.getContext("2d")!;
      const stream = out.captureStream(30);
      const capture = (v as HTMLVideoElement & { captureStream?: () => MediaStream }).captureStream?.();
      capture?.getAudioTracks().forEach((t) => stream.addTrack(t));
      const type = ["video/webm;codecs=vp9,opus", "video/webm;codecs=vp8,opus", "video/webm"].find((t) => MediaRecorder.isTypeSupported(t));
      const recorder = new MediaRecorder(stream, type ? { mimeType: type } : undefined);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
      const finished = new Promise<void>((resolve) => (recorder.onstop = () => resolve()));
      let raf = 0;
      const draw = () => {
        ctx.fillStyle = "#14110f";
        ctx.fillRect(0, 0, out.width, out.height);
        const filmW = 880, filmH = Math.round(filmW * (v.videoHeight || 9) / (v.videoWidth || 16));
        ctx.drawImage(v, 0, Math.round((out.height - filmH) / 2), filmW, filmH);
        const aw = out.width - filmW - 24, ah = Math.round(aw * avatar.height / avatar.width);
        ctx.drawImage(avatar, filmW + 12, Math.round((out.height - ah) / 2), aw, ah);
        const t = v.currentTime * 1000, cue = items.find((i) => i.cue.start_ms <= t && t < i.cue.end_ms);
        ctx.font = "600 22px system-ui, sans-serif";
        ctx.textAlign = "center";
        if (cue) {
          ctx.fillStyle = "rgba(0,0,0,.65)";
          ctx.fillRect(40, out.height - 86, filmW - 80, 44);
          ctx.fillStyle = "#fff";
          ctx.fillText(cue.cue.text.slice(0, 90), filmW / 2, out.height - 56);
        }
        ctx.font = "14px system-ui, sans-serif";
        ctx.fillStyle = "rgba(255,255,255,.85)";
        ctx.fillText(WATERMARK, filmW + 12 + aw / 2, out.height - 20);
        raf = requestAnimationFrame(draw);
      };
      v.pause();
      v.currentTime = 0;
      await new Promise((r) => setTimeout(r, 300));
      recorder.start(1000);
      draw();
      await v.play();
      await new Promise<void>((resolve) => v.addEventListener("ended", () => resolve(), { once: true }));
      cancelAnimationFrame(raf);
      recorder.stop();
      await finished;
      const url = URL.createObjectURL(new Blob(chunks, { type: "video/webm" }));
      const a = document.createElement("a");
      a.href = url;
      a.download = (videoName.replace(/\.[^.]+$/, "") || "film") + "-lsf-avatar.webm";
      a.click();
      setTimeout(() => URL.revokeObjectURL(url), 60_000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Export impossible dans ce navigateur.");
    } finally {
      setExporting(false);
    }
  }

  const engineReady = engines?.[engine] !== false;
  return (
    <div className="film-mode">
      <div className="film-tabs" role="tablist">
        <button role="tab" aria-selected={mode === "film"} className={mode === "film" ? "active" : ""} onClick={() => setMode("film")}>
          <Clapperboard size={16} /> Film sous-titré
        </button>
        <button role="tab" aria-selected={mode === "live"} className={mode === "live" ? "active" : ""} onClick={() => setMode("live")}>
          <Mic size={16} /> Direct (micro)
        </button>
      </div>
      <p className="film-disclaimer">
        Chaque sous-titre est traduit en LSF et signé par l’avatar, en synchronie avec l’image.
      </p>
      {error && <p role="alert" className="research-error">{error}</p>}
      {mode === "film" ? (
        <>
          <div className="film-setup card">
            <div className="film-inputs">
              <label className="audio-drop">
                <Upload size={18} />
                <span>{videoName || "Choisir le film (MP4, WebM)"}</span>
                <input type="file" accept="video/*" hidden onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (!f) return;
                  setVideoName(f.name);
                  setVideoFile(f);
                  setVideoUrl(URL.createObjectURL(f));
                }} />
              </label>
              <label className="audio-drop">
                <Upload size={18} />
                <span>{subsName || "Sous-titres français (SRT, VTT)"}</span>
                <input type="file" accept=".srt,.vtt,text/vtt" hidden onChange={(e) => e.target.files?.[0] && void loadSubtitles(e.target.files[0])} />
              </label>
              <label className={`audio-drop${transcription ? "" : " disabled"}`} title={transcription ? "" : "Modèle Whisper non installé sur ce serveur"}>
                <Upload size={18} />
                <span>{transcription ? "Ou transcrire l’audio (60 s max)" : "Transcription indisponible (Whisper non installé)"}</span>
                <input type="file" accept="audio/*,video/*" hidden disabled={!transcription || busy} onChange={(e) => e.target.files?.[0] && void transcribeFilm(e.target.files[0])} />
              </label>
            </div>
            <div className="film-options">
              <div role="radiogroup" aria-label="Moteur">
                {(["c0", "b2"] as const).map((id) => (
                  <label key={id}>
                    <input type="radio" name="film-engine" checked={engine === id} disabled={busy} onChange={() => setEngine(id)} />
                    {id === "b2" ? "B2 · signes du lexique" : "C0 · générateur"}
                    {engines?.[id] === false && <small> (non installé)</small>}
                  </label>
                ))}
              </div>
              <label>
                <input type="checkbox" checked={lsfOrder} disabled={busy || engine !== "b2"} onChange={(e) => setLsfOrder(e.target.checked)} />
                Ordre LSF
              </label>
              <label>
                <input type="checkbox" checked={nonmanual} disabled={busy} onChange={(e) => setNonmanual(e.target.checked)} />
                Visage grammatical
              </label>
              <label title={emotion ? "" : "Modèle emotion2vec non installé sur ce serveur"}>
                <input type="checkbox" checked={useEmotion} disabled={busy || !emotion || !videoFile} onChange={(e) => setUseEmotion(e.target.checked)} />
                Émotion de la voix par réplique (vitesse et amplitude des mains)
                {!emotion && <small> (indisponible)</small>}
              </label>
            </div>
            {phase && <small className="film-phase">{phase}</small>}
            <CastSheet cast={cast} onChange={editCast} counts={lines} />
            <div className="film-actions">
              <button className="primary-button" disabled={busy || !items.length || !engineReady} onClick={() => void generateAll()}>
                {busy ? <Loader2 className="spin" size={16} /> : <Wand2 size={16} />}
                {busy ? `Génération ${done} / ${items.length}…` : `Générer la piste LSF (${items.length} sous-titres)`}
              </button>
              {busy && (
                <button className="secondary-button" onClick={() => (cancel.current = true)}>
                  <Square size={14} /> Arrêter
                </button>
              )}
              <button className="secondary-button" disabled={!videoUrl || !motions.length || exporting || busy || prepared < motions.length}
                onClick={() => void exportFilm()}>
                {exporting ? <Loader2 className="spin" size={15} /> : <ArrowDownToLine size={15} />}
                {exporting ? "Enregistrement en temps réel…" : "Exporter film + avatar (WebM)"}
              </button>
            </div>
            {items.length > 0 && (
              <div className="film-progress" aria-live="polite">
                <progress max={items.length} value={done} />
                <small>
                  {motions.length} / {items.length} sous-titres signés · {prepared} / {motions.length} prêts pour la lecture
                  {maxDelay > 0 && ` · retard maximal de l’avatar ${(maxDelay / 1000).toFixed(1)} s`}
                </small>
              </div>
            )}
          </div>
          <div className="film-stage">
            <div className="film-video">
              {videoUrl ? <video ref={video} src={videoUrl} controls playsInline /> : <div className="film-placeholder"><Clapperboard size={36} /><span>Ajoutez un film et ses sous-titres</span></div>}
              {cueIndex >= 0 && firstLine.has(cueIndex) && (
                <div className="film-intro" key={cueIndex}><SpeakerChip entry={castOf(cast, items[cueIndex].cue.speaker)} side big /></div>
              )}
              {cueIndex >= 0 && <div className="film-subtitle">{items[cueIndex].cue.text}</div>}
            </div>
            <div className="film-avatar">
              {cast.length > 0 && (
                <div className="film-speaker-badge" data-role-shift={roleShift.torsoYaw.toFixed(3)}>
                  <StageDots cast={cast} active={signer} />
                  <SpeakerChip entry={signer} />
                </div>
              )}
              <AvatarCanvas
                avatarCredit={avatarCredit}
                motion={activeMotion}
                currentTimeMs={active?.local_ms ?? 0}
                roleShift={roleShift}
                prewarm={motions}
                onPrewarmed={setPrepared}
                onCanvasReady={(c) => (avatarCanvas.current = c)}
              />
              <small className="film-watermark">{WATERMARK}</small>
            </div>
          </div>
          {items.length > 0 && (
            <ol className="film-cues">
              {items.map((item, i) => (
                <li key={i} className={i === cueIndex ? "current" : ""}>
                  <button onClick={() => video.current && (video.current.currentTime = item.cue.start_ms / 1000)}>
                    <span className="film-time">{clock(item.cue.start_ms)}</span>
                    <span className="film-text">
                      {cast.length > 0 && <SpeakerChip entry={castOf(cast, item.cue.speaker)} />}
                      {item.cue.text}
                    </span>
                    <span className="film-state">
                      {item.motion
                        ? [
                            item.motion.generation?.lemmas ? item.motion.generation.lemmas.join(" · ") || "repos" : `${(item.motion.duration_ms / 1000).toFixed(1)} s`,
                            ...[...new Set((item.motion.nonmanual ?? []).map((m) => MARKER_LABEL[m.type]))],
                            item.motion.generation?.emotion && item.motion.generation.emotion.label !== "neutral"
                              ? `émotion : ${EMOTION_LABEL[item.motion.generation.emotion.label] ?? item.motion.generation.emotion.label}` : "",
                            placed[i]?.delay_ms > 300 ? `retard ${(placed[i].delay_ms / 1000).toFixed(1)} s` : "",
                          ].filter(Boolean).join(" — ")
                        : item.error ? `erreur : ${item.error}` : busy ? "en attente" : ""}
                    </span>
                  </button>
                </li>
              ))}
            </ol>
          )}
        </>
      ) : (
        <LiveMode avatarCredit={avatarCredit} engine={engine} setEngine={setEngine} lsfOrder={lsfOrder} nonmanual={nonmanual}
          transcription={transcription} engines={engines} emotion={emotion} />
      )}
    </div>
  );
}

type LiveLine = { text: string; heard_ms: number; shown_ms?: number; motion?: MotionFile; dropped?: boolean };

/** Microphone -> phrases cut at pauses -> persistent local Whisper -> engine -> avatar, in speaking order. */
function LiveMode({
  avatarCredit, engine, setEngine, lsfOrder, nonmanual, transcription, engines,
}: {
  avatarCredit?: string; engine: LsfEngine; setEngine: (e: LsfEngine) => void; lsfOrder: boolean; nonmanual: boolean;
  transcription?: boolean; engines?: Partial<Record<LsfEngine, boolean>>; emotion?: boolean;
}) {
  const [on, setOn] = useState(false),
    [lines, setLines] = useState<LiveLine[]>([]),
    [playing, setPlaying] = useState<{ index: number; started: number } | null>(null),
    [now, setNow] = useState(0),
    [ready, setReady] = useState<"loading" | "ready" | "off">(transcription ? "loading" : "off"),
    [error, setError] = useState("");
  const stop = useRef<() => void>(() => {});
  const detector = useRef<PhraseDetector | null>(null);
  const chain = useRef<Promise<void>>(Promise.resolve());
  const count = useRef(0);
  const options = useRef({ engine, lsfOrder, nonmanual });
  options.current = { engine, lsfOrder, nonmanual };
  useEffect(() => {
    let id = 0;
    const tick = () => { setNow(performance.now()); id = requestAnimationFrame(tick); };
    id = requestAnimationFrame(tick);
    return () => { cancelAnimationFrame(id); stop.current(); };
  }, []);
  // Load the speech model and the signing engine while the user gets ready, so the first phrase is as fast as the next.
  useEffect(() => {
    if (!transcription) return;
    setReady("loading");
    request(`/live/warmup?engine=${engine}`, { method: "POST" }).then(() => setReady("ready")).catch(() => setReady("off"));
  }, [transcription, engine]);
  // Play the queue: next line as soon as the current one is finished.
  useEffect(() => {
    const current = playing ? lines[playing.index] : undefined;
    const finished = !current?.motion || now - playing!.started > current.motion.duration_ms + 200;
    if (!finished) return;
    const next = lines.findIndex((l, i) => l.motion && (!playing || i > playing.index));
    if (next >= 0) {
      setPlaying({ index: next, started: now });
      setLines((prev) => prev.map((l, i) => (i === next ? { ...l, shown_ms: now } : l)));
    }
  }, [now, lines, playing]);

  function onPhrase(samples: Float32Array, rate: number) {
    const heard = performance.now();
    const index = count.current++;
    setLines((prev) => [...prev, { text: "…", heard_ms: heard }]);
    const wav = encodeWav(samples, rate);
    const update = (patch: Partial<LiveLine>) => setLines((prev) => prev.map((l, i) => (i === index ? { ...l, ...patch } : l)));
    // One phrase at a time, in speaking order: each step waits for the previous one.
    chain.current = chain.current.then(async () => {
      try {
        const form = new FormData();
        form.append("file", new Blob([wav], { type: "audio/wav" }), "phrase.wav");
        const heardText = (await request<{ text: string }>("/live/transcribe", { method: "POST", body: form })).text.trim();
        if (!heardText) {
          update({ dropped: true });
          return;
        }
        update({ text: heardText });
        const { engine: eng, lsfOrder: order, nonmanual: face } = options.current;
        const motion = await request<MotionFile>("/lsf/generate", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify(generateBody(heardText.slice(0, 400), eng, order, face)),
        });
        update({ motion });
      } catch (e) {
        update({ dropped: true });
        setError(e instanceof Error ? e.message : "Traitement impossible.");
      }
    });
  }

  async function start() {
    setError("");
    let stream: MediaStream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    } catch {
      setError("Micro refusé ou indisponible.");
      return;
    }
    const context = new AudioContext();
    const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" }));
    await context.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
    const source = context.createMediaStreamSource(stream);
    const node = new AudioWorkletNode(context, "signaculture-capture");
    const rate = context.sampleRate;
    const vad = new PhraseDetector({ sampleRate: rate }, (samples) => onPhrase(samples, rate));
    detector.current = vad;
    node.port.onmessage = (e: MessageEvent<Float32Array>) => vad.push(e.data);
    source.connect(node);
    stop.current = () => {
      vad.flush();
      node.port.onmessage = null;
      source.disconnect();
      stream.getTracks().forEach((t) => t.stop());
      void context.close();
      detector.current = null;
      setOn(false);
    };
    setOn(true);
  }

  const current = playing ? lines[playing.index] : undefined;
  const shown = lines.filter((l) => l.shown_ms);
  const lastDelay = shown.length ? shown[shown.length - 1].shown_ms! - shown[shown.length - 1].heard_ms : 0;
  const level = detector.current ? Math.min(1, detector.current.level * 12) : 0;
  const label = on ? "Arrêter le direct" : ready === "loading" ? "Préparation de la reconnaissance vocale…"
    : ready === "ready" ? "Démarrer le direct (micro)" : "Direct indisponible : Whisper non installé";
  return (
    <div className="live-mode">
      <div className="film-setup card">
        <div className="film-options">
          <div role="radiogroup" aria-label="Moteur">
            {(["c0", "b2"] as const).map((id) => (
              <label key={id}>
                <input type="radio" name="live-engine" checked={engine === id} disabled={on} onChange={() => setEngine(id)} />
                {id === "b2" ? "B2 · signes du lexique" : "C0 · générateur"}
                {engines?.[id] === false && <small> (non installé)</small>}
              </label>
            ))}
          </div>
        </div>
        <div className="film-actions">
          <button className="primary-button" disabled={ready !== "ready"} onClick={() => (on ? stop.current() : void start())}>
            {on ? <MicOff size={16} /> : ready === "loading" ? <Loader2 className="spin" size={16} /> : <Mic size={16} />}
            {label}
          </button>
          {on && (
            <span className={`live-level${detector.current?.speaking ? " speaking" : ""}`} aria-label="Niveau du micro">
              <i style={{ width: `${Math.round(level * 100)}%` }} />
            </span>
          )}
        </div>
        <small>
          Parlez naturellement : chaque phrase est transcrite dès que vous marquez une pause, puis signée par l’avatar.
          Tout est traité sur cet ordinateur{lastDelay ? ` · délai fin de phrase → signes : ${(lastDelay / 1000).toFixed(1)} s` : ""}.
        </small>
        {error && <p role="alert" className="research-error">{error}</p>}
      </div>
      <div className="film-stage">
        <div className="film-avatar live">
          <AvatarCanvas
            avatarCredit={avatarCredit}
            motion={current?.motion ?? null}
            currentTimeMs={current?.motion && playing ? Math.min(now - playing.started, current.motion.duration_ms) : 0}
          />
          {current && <div className="film-subtitle">{current.text}</div>}
          <small className="film-watermark">{WATERMARK}</small>
        </div>
        <ol className="film-cues live">
          {lines.map((l, i) => l.dropped ? null : (
            <li key={i} className={playing?.index === i ? "current" : ""}>
              <span className="film-text">{l.text}</span>
              <span className="film-state">
                {l.motion ? (l.shown_ms ? `signé ${((l.shown_ms - l.heard_ms) / 1000).toFixed(1)} s après la phrase` : "en file")
                  : l.text === "…" ? "transcription…" : "génération…"}
              </span>
            </li>
          ))}
        </ol>
      </div>
    </div>
  );
}
