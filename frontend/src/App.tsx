import { useEffect, useRef, useState, useCallback } from "react";
import {
  AudioLines,
  ArrowDownToLine,
  ArrowRight,
  Check,
  ChevronRight,
  CircleHelp,
  Film,
  FlaskConical,
  Hand,
  Library,
  Loader2,
  Pause,
  Play,
  Plus,
  RotateCcw,
  Sparkles,
  Upload,
  Waves,
  X,
} from "lucide-react";
import { request } from "./services/api";
import { LexicalPipelinePanel } from "./components/LexicalPipelinePanel";
import { ConceptComposer } from "./components/ConceptComposer";
import { ResearchPanel } from "./components/ResearchPanel";
import { AvatarCanvas } from "./components/AvatarPlayer/AvatarCanvas";
import { stitchAnimationPlan } from "./utils/planStitcher";
import type {
  AnimationPlan,
  AudioFeatures,
  GlossEntry,
  MotionFile,
} from "./types/contract";
import "./studio.css";

type Page = "studio" | "library" | "capture" | "research";
type Capabilities = {
  cv_corrected_count?: number;
  dictasign?: {
    status: string;
    indexed_concepts: number;
    inventory_processed: number;
    processed: number;
    total: number;
    imported: number;
    quality_review_needed: number;
    failed: number;
  } | null;
  corpus?: {
    status: string;
    total: number;
    processed: number;
    imported: number;
    failed: number;
  } | null;
  capture: boolean;
  emotion: boolean;
  nlp: string;
  library_count: number;
};
type Quality = {
  pose_coverage: number;
  left_hand_coverage: number;
  right_hand_coverage: number;
  face_coverage: number;
  sampled_frames: number;
};
type CaptureJob = {
  job_id: string;
  status: "queued" | "processing" | "ready" | "failed";
  progress: number;
  filename: string;
  error?: string;
  quality?: Quality;
  duration_ms?: number;
};
type LibraryEntry = {
  capture_id?: string;
  source_video_url?: string;
  gloss: GlossEntry;
  motion: { motion_id: string; duration_ms: number; path: string };
  source: string;
  quality: Quality;
};

const json = (body: unknown) => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});
const seconds = (ms: number) => `${(ms / 1000).toFixed(1)} s`;
const clock = (ms: number) =>
  `${Math.floor(ms / 60000)
    .toString()
    .padStart(2, "0")}:${((ms % 60000) / 1000).toFixed(1).padStart(4, "0")}`;

export default function App() {
  const [page, setPage] = useState<Page>("studio");
  const [connected, setConnected] = useState(false);
  const [caps, setCaps] = useState<Capabilities | null>(null);
  const [glossary, setGlossary] = useState<GlossEntry[]>([]);
  const [library, setLibrary] = useState<LibraryEntry[]>([]);
  // A showcase link can open the Studio with ?text=... and play it immediately.
  const linkedText = useRef(new URLSearchParams(window.location.search).get("text"));
  const [text, setText] = useState(linkedText.current || "Bonjour.");
  const [plan, setPlan] = useState<AnimationPlan | null>(null);
  const [motion, setMotion] = useState<MotionFile | null>(null);
  const [label, setLabel] = useState("Une scène prête pour votre histoire");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [speed, setSpeed] = useState(1);
  const [audio, setAudio] = useState<AudioFeatures | null>(null);
  const [audioUrl, setAudioUrl] = useState("");
  const [audioName, setAudioName] = useState("");
  const [audioBusy, setAudioBusy] = useState(false);
  const [expressive, setExpressive] = useState(false);
  const [canvas, setCanvas] = useState<HTMLCanvasElement | null>(null);
  const [recording, setRecording] = useState(false);
  const [sourceUrl, setSourceUrl] = useState("");
  const [referenceUrl, setReferenceUrl] = useState("");
  const [captureFile, setCaptureFile] = useState<File | null>(null);
  const [rights, setRights] = useState(false);
  const [job, setJob] = useState<CaptureJob | null>(null);
  const [concept, setConcept] = useState("");
  const [aliases, setAliases] = useState("");
  const [source, setSource] = useState("");
  const [licence, setLicence] = useState("");
  const [saved, setSaved] = useState(false);
  const [search, setSearch] = useState("");
  const audioInput = useRef<HTMLInputElement>(null);
  const videoInput = useRef<HTMLInputElement>(null);
  const audioPlayer = useRef<HTMLAudioElement>(null);
  const generation = useRef(0);
  const duration = motion?.duration_ms || 0;
  const active = plan?.segments.find(
    (s) => time >= s.start_ms && time < s.start_ms + s.duration_ms,
  );
  const sample = audio?.frames.find(
    (f) =>
      f.start_ms <= (duration ? (time / duration) * audio.duration_ms : 0) &&
      f.end_ms > (duration ? (time / duration) * audio.duration_ms : 0),
  );
  const emotion = audio?.emotion_hypothesis || null;

  const refresh = useCallback(async () => {
    try {
      const [cap, words, entries] = await Promise.all([
        request<Capabilities>("/capabilities"),
        request<{ entries: GlossEntry[] }>("/glossary?include_synthetic=true"),
        request<{ entries: LibraryEntry[] }>("/library"),
      ]);
      setCaps(cap);
      setGlossary(words.entries);
      setLibrary(entries.entries);
      setConnected(true);
    } catch {
      setConnected(false);
    }
  }, []);
  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), 15000);
    return () => clearInterval(timer);
  }, [refresh]);
  useEffect(
    () => () => {
      if (audioUrl) URL.revokeObjectURL(audioUrl);
    },
    [audioUrl],
  );
  useEffect(
    () => () => {
      if (sourceUrl) URL.revokeObjectURL(sourceUrl);
    },
    [sourceUrl],
  );

  // Compose the linked sentence once, after the library has loaded.
  useEffect(() => {
    if (!linkedText.current || !connected || !library.length) return;
    linkedText.current = null;
    void compose();
  }, [connected, library.length]); // eslint-disable-line react-hooks/exhaustive-deps

  useEffect(() => {
    if (!playing || !duration) return;
    let id = 0,
      last: number | undefined;
    const tick = (now: number) => {
      const delta = last === undefined ? 0 : Math.min(100, now - last);
      last = now;
      setTime((prev) => {
        const next = prev + delta * speed;
        if (next >= duration) {
          setPlaying(false);
          return duration;
        }
        return next;
      });
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(id);
  }, [playing, duration, speed]);

  useEffect(() => {
    if (!job || !["queued", "processing"].includes(job.status)) return;
    let cancelled = false;
    const timer = window.setInterval(async () => {
      try {
        const result = await request<CaptureJob>(`/captures/${job.job_id}`);
        if (cancelled) return;
        if (result.status === "ready") {
          const captured = await request<MotionFile>(
            `/captures/${result.job_id}/motion`,
          );
          if (cancelled) return;
          setMotion(captured);
          setTime(0);
          setPlaying(true);
          setPlan(null);
          setLabel("Votre capture · mouvement extrait");
        }
        if (!cancelled) setJob(result);
      } catch (err) {
        if (!cancelled) {
          setError(String(err instanceof Error ? err.message : err));
          setJob({ ...job, status: "failed" });
        }
      }
    }, 1000);
    return () => {
      cancelled = true;
      clearInterval(timer);
    };
  }, [job?.job_id, job?.status]); // eslint-disable-line react-hooks/exhaustive-deps

  async function compose() {
    if (busy || !text.trim()) return;
    const token = ++generation.current;
    setBusy(true);
    setError("");
    setPlan(null);
    setMotion(null);
    setTime(0);
    setPlaying(false);
    try {
      const result = await request<AnimationPlan>(
        "/compose-animation",
        json({ text, ...(audio ? { audio_features: audio } : {}) }),
      );
      const files = await Promise.all(
        result.segments.map(
          async (seg) =>
            [
              seg.motion_id,
              await request<MotionFile>(
                seg.asset_url || `/assets/${seg.motion_id}.json`,
              ),
            ] as const,
        ),
      );
      if (token !== generation.current) return;
      setPlan(result);
      const entry =
        result.segments.length === 1
          ? library.find(
              (e) => e.motion.motion_id === result.segments[0].motion_id,
            )
          : undefined;
      setReferenceUrl(
        entry?.capture_id
          ? `/api/captures/${entry.capture_id}/source`
          : entry?.source_video_url
            ? `/api${entry.source_video_url}`
            : "",
      );
      if (files.length) {
        setMotion(stitchAnimationPlan(result, new Map(files)));
        setLabel(
          result.segments.length === 1
            ? glossary.find((g) => g.gloss_id === result.segments[0].gloss_id)
                ?.concept_fr || "Votre séquence"
            : "Votre séquence",
        );
        setPlaying(true);
      } else setLabel("Aucun mouvement trouvé");
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Connexion au service impossible.",
      );
    } finally {
      if (token === generation.current) setBusy(false);
    }
  }

  async function loadConceptPlan(result: AnimationPlan) {
    const files = await Promise.all(
      result.segments.map(
        async (seg) =>
          [
            seg.motion_id,
            await request<MotionFile>(
              seg.asset_url || `/assets/${seg.motion_id}.json`,
            ),
          ] as const,
      ),
    );
    generation.current += 1;
    setPlan(result);
    setMotion(stitchAnimationPlan(result, new Map(files)));
    setTime(0);
    setPlaying(true);
    setLabel("Sélection de concepts");
    setReferenceUrl("");
    setError("");
  }

  async function analyseAudio(file: File) {
    setAudioBusy(true);
    setError("");
    setAudio(null);
    setAudioName(file.name);
    setAudioUrl(URL.createObjectURL(file));
    try {
      const form = new FormData();
      form.append("file", file);
      setAudio(
        await request<AudioFeatures>(
          `/analyze-audio?include_emotion=${caps?.emotion ? "true" : "false"}`,
          { method: "POST", body: form },
        ),
      );
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Analyse audio impossible.",
      );
    } finally {
      setAudioBusy(false);
    }
  }

  async function startCapture() {
    if (!captureFile || !rights || busy) return;
    setBusy(true);
    setError("");
    setSaved(false);
    try {
      const form = new FormData();
      form.append("file", captureFile);
      form.append("rights_confirmed", "true");
      const result = await request<CaptureJob>("/captures", {
        method: "POST",
        body: form,
      });
      setJob(result);
      if (result.status === "ready") {
        setMotion(
          await request<MotionFile>(`/captures/${result.job_id}/motion`),
        );
        setPlan(null);
        setTime(0);
        setPlaying(true);
        setLabel("Capture importée");
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Import impossible.");
    } finally {
      setBusy(false);
    }
  }

  async function saveCapture() {
    if (!job || !concept.trim() || !source.trim() || !licence.trim()) return;
    setBusy(true);
    setError("");
    try {
      await request(
        `/captures/${job.job_id}/register`,
        json({
          concept,
          aliases: aliases
            .split(",")
            .map((x) => x.trim())
            .filter(Boolean),
          source,
          license: licence,
        }),
      );
      await refresh();
      setSaved(true);
    } catch (err) {
      setError(
        err instanceof Error ? err.message : "Enregistrement impossible.",
      );
    } finally {
      setBusy(false);
    }
  }

  async function loadEntry(id: string, name: string) {
    setError("");
    setPlaying(false);
    setPlan(null);
    setTime(0);
    try {
      const meta = await request<{ path: string; synthetic?: boolean }>(
        `/motions/${id}`,
      );
      const filename = meta.path.split("/").pop();
      const clip = await request<MotionFile>(`/assets/${filename}`);
      setMotion({ ...clip, synthetic: meta.synthetic ?? clip.synthetic });
      const entry = library.find((e) => e.motion.motion_id === id);
      setReferenceUrl(
        entry?.capture_id
          ? `/api/captures/${entry.capture_id}/source`
          : entry?.source_video_url
            ? `/api${entry.source_video_url}`
            : "",
      );
      setLabel(name);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Mouvement introuvable.");
    }
  }

  function togglePlay() {
    if (!duration) return;
    if (time >= duration) setTime(0);
    setPlaying((value) => !value);
  }

  async function exportPreview() {
    if (!canvas || !duration || recording) return;
    if (!window.MediaRecorder) {
      setError(
        "Cet appareil ne permet pas l’enregistrement vidéo. Utilisez Chrome ou Firefox.",
      );
      return;
    }
    setRecording(true);
    setError("");
    setTime(0);
    setSpeed(1);
    setPlaying(false);
    // Let React commit the reset and the avatar render frame zero before capture.
    // Otherwise the first encoded frame can contain the previous paused pose.
    await new Promise<void>((resolve) =>
      requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
    );
    const output = document.createElement("canvas");
    output.width = 1280;
    output.height = 960;
    const ctx = output.getContext("2d")!;
    const stream = output.captureStream(30);
    const type = [
      "video/webm;codecs=vp9",
      "video/webm;codecs=vp8",
      "video/webm",
    ].find((t) => MediaRecorder.isTypeSupported(t));
    const recorder = new MediaRecorder(
      stream,
      type ? { mimeType: type } : undefined,
    );
    const chunks: Blob[] = [];
    let frame = 0;
    const draw = () => {
      ctx.fillStyle = "#f5efe7";
      ctx.fillRect(0, 0, 1280, 960);
      const ratio = Math.min(1280 / canvas.width, 900 / canvas.height);
      const width = canvas.width * ratio,
        height = canvas.height * ratio;
      ctx.drawImage(
        canvas,
        (1280 - width) / 2,
        (900 - height) / 2,
        width,
        height,
      );
      ctx.fillStyle = "rgba(48,28,19,.8)";
      ctx.fillRect(0, 884, 1280, 76);
      ctx.fillStyle = "#fff";
      ctx.font = "18px sans-serif";
      ctx.fillText(
        plan?.unknowns.length
          ? `Aperçu incomplet · mots non représentés : ${plan.unknowns.join(", ")}`
          : "SignaCulture · Aperçu expérimental · Non validé en LSF",
        28,
        913,
        1224,
      );
      ctx.font = "12px sans-serif";
      ctx.fillText(
        (
          motion?.attributions ||
          (motion?.attribution ? [motion.attribution] : [])
        )
          .map((a) => `${a.credit} · ${a.license} · ${a.source_url}`)
          .join(" / ") +
          " · Avatar Eva: UPF-GTI / Ready Player Me · revue locale",
        28,
        943,
        1224,
      );
      frame = requestAnimationFrame(draw);
    };
    recorder.ondataavailable = (e) => {
      if (e.data.size) chunks.push(e.data);
    };
    recorder.onstop = () => {
      cancelAnimationFrame(frame);
      stream.getTracks().forEach((t) => t.stop());
      const url = URL.createObjectURL(
        new Blob(chunks, { type: recorder.mimeType }),
      );
      const link = document.createElement("a");
      link.href = url;
      link.download = "signaculture-apercu.webm";
      link.click();
      setTimeout(() => URL.revokeObjectURL(url), 10000);
      setRecording(false);
      setPlaying(false);
    };
    draw();
    recorder.start();
    setPlaying(true);
    window.setTimeout(() => {
      if (recorder.state === "recording") recorder.stop();
    }, duration + 200);
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a
          className="brand-mark"
          href="#studio"
          aria-label="SignaCulture accueil"
        >
          <Hand size={26} />
        </a>
        <nav aria-label="Navigation principale">
          <button
            className={page === "studio" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("studio")}
          >
            <Waves />
            <span>Studio</span>
          </button>
          <button
            className={page === "library" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("library")}
          >
            <Library />
            <span>Bibliothèque</span>
          </button>
          <button
            className={page === "capture" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("capture")}
          >
            <Plus />
            <span>Capturer</span>
          </button>
          <button
            className={page === "research" ? "nav-item active" : "nav-item"}
            onClick={() => setPage("research")}
          >
            <FlaskConical />
            <span>Recherche</span>
          </button>
        </nav>
        <a
          className="nav-help"
          href="https://github.com/upf-gti/animics"
          target="_blank"
          rel="noreferrer"
          aria-label="Documentation Animics"
        >
          <CircleHelp size={20} />
        </a>
        <span className="workspace-avatar">SC</span>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <a className="wordmark" href="/" title="Présentation du projet">
            signaculture<span>AI</span>
          </a>
          <span className="topbar-divider" />
          <span className="project-name">
            Mon espace de création <ChevronRight size={14} />
            <strong>
              {page === "studio"
                ? "Nouveau récit"
                : page === "capture"
                  ? "Nouvelle capture"
                  : page === "research"
                    ? "Données & modèles"
                    : "Mes mouvements"}
            </strong>
          </span>
          <button className="connection" onClick={() => void refresh()}>
            <i className={connected ? "online" : ""} />
            {connected ? "Espace local connecté" : "Service déconnecté"}
          </button>
        </header>
        <main>
          <section className="page-heading">
            <div>
              <span className="eyebrow">LE MOUVEMENT DONNE DU SENS</span>
              <h1>
                {page === "studio" ? (
                  <>
                    Vos mots prennent <em>vie.</em>
                  </>
                ) : page === "capture" ? (
                  <>
                    Du geste à la <em>matière.</em>
                  </>
                ) : page === "research" ? (
                  <>
                    Mesurer, puis <em>progresser.</em>
                  </>
                ) : (
                  <>
                    Une bibliothèque <em>vivante.</em>
                  </>
                )}
              </h1>
              <p>
                {page === "studio"
                  ? "Composez un récit visuel, enrichissez-le de votre voix et explorez le mouvement."
                  : page === "capture"
                    ? "Transformez vos vidéos en mouvements réutilisables, du corps jusqu’aux doigts."
                    : page === "research"
                      ? "Des sources identifiées, des expériences reproductibles et des résultats mesurés."
                      : "Retrouvez vos captures et les concepts qui leur donnent un sens."}
              </p>
            </div>
            <span className="project-tag">
              <span />
              PROJET EXPÉRIMENTAL
            </span>
          </section>
          {error && (
            <div className="error-banner" role="alert">
              <span>{error}</span>
              <button aria-label="Fermer l’erreur" onClick={() => setError("")}>
                <X size={16} />
              </button>
            </div>
          )}
          <div className="studio-layout">
            <section className="editor-column">
              {page === "research" && <ResearchPanel />}
              {page === "studio" && (
                <>
                  <div className="card text-card">
                    <div className="card-heading">
                      <span className="step">01</span>
                      <div>
                        <h2>Tout commence par vos mots</h2>
                        <p>Votre texte en français</p>
                      </div>
                      <span className="small-chip">FR → Concepts</span>
                    </div>
                    <label className="sr-only" htmlFor="story">
                      Votre texte en français
                    </label>
                    <textarea
                      id="story"
                      disabled={busy}
                      value={text}
                      maxLength={4000}
                      onChange={(e) => setText(e.target.value)}
                      placeholder="Racontez une histoire, accueillez votre public…"
                    />
                    <div className="text-footer">
                      <span>{text.length} / 4 000 caractères</span>
                      <span className="tiny-status">
                        <i />
                        Brouillon à relire
                      </span>
                    </div>
                    <div className="suggestions">
                      <span>Essayez</span>
                      {/* Each phrase resolves every content word to a real captured sign. */}
                      {[
                        "Bonjour, comment ça va ?",
                        "Je veux boire de l'eau.",
                        "Je visite Paris avec ma famille.",
                        "Le musée est fermé demain.",
                        "Je veux apprendre la langue des signes.",
                      ].map((phrase) => (
                        <button
                          key={phrase}
                          disabled={busy}
                          onClick={() => setText(phrase)}
                        >
                          {phrase}
                          <ArrowRight size={12} />
                        </button>
                      ))}
                    </div>
                    <button
                      className="primary-button compose-button"
                      onClick={() => void compose()}
                      disabled={busy || !text.trim() || recording}
                    >
                      {busy ? (
                        <Loader2 className="spin" size={17} />
                      ) : (
                        <Sparkles size={17} />
                      )}{" "}
                      {busy ? "Composition en cours…" : "Créer l’animation"}
                      <ArrowRight size={17} />
                    </button>
                  </div>
                  <div className="card">
                    <ConceptComposer
                      text={text}
                      audio={audio}
                      onPlan={loadConceptPlan}
                    />
                  </div>
                  <div className="card">
                    <LexicalPipelinePanel text={text} />
                  </div>
                  <div className="card audio-card">
                    <div className="card-heading">
                      <span className="step">02</span>
                      <div>
                        <h2>Ajoutez votre voix</h2>
                        <p>Le rythme, les pauses et l’expressivité</p>
                      </div>
                      <span className="optional-tag">Facultatif</span>
                    </div>
                    <input
                      ref={audioInput}
                      type="file"
                      accept="audio/*,video/mp4,video/webm"
                      hidden
                      onChange={(e) => {
                        if (e.target.files?.[0])
                          void analyseAudio(e.target.files[0]);
                        e.target.value = "";
                      }}
                    />
                    <button
                      className="audio-drop"
                      onClick={() => audioInput.current?.click()}
                      disabled={audioBusy}
                    >
                      <span className="file-icon">
                        {audioBusy ? (
                          <Loader2 className="spin" />
                        ) : (
                          <AudioLines />
                        )}
                      </span>
                      <span>
                        <strong>
                          {audioBusy
                            ? "Écoute et analyse en cours…"
                            : audioName || "Importer un audio ou une vidéo"}
                        </strong>
                        <small>
                          {audioBusy && caps?.emotion
                            ? "Analyse locale avec emotion2vec+"
                            : "WAV, MP3, MP4 · 60 secondes maximum"}
                        </small>
                      </span>
                      <Upload size={17} />
                    </button>
                    {audioUrl && (
                      <audio
                        ref={audioPlayer}
                        controls
                        src={audioUrl}
                        className="audio-player"
                      />
                    )}
                    {audio && (
                      <>
                        <div
                          className="waveform"
                          aria-label="Énergie mesurée de l’audio"
                        >
                          {audio.frames
                            .filter(
                              (_, i) =>
                                i %
                                  Math.max(
                                    1,
                                    Math.floor(audio.frames.length / 80),
                                  ) ===
                                0,
                            )
                            .slice(0, 80)
                            .map((f, i) => (
                              <span
                                key={i}
                                className={f.is_pause ? "pause-bar" : ""}
                                style={{
                                  height: `${Math.max(5, Math.min(100, f.energy * 600))}%`,
                                }}
                              />
                            ))}
                        </div>
                        <div className="audio-metrics">
                          <span>
                            <strong>{seconds(audio.duration_ms)}</strong>Durée
                          </span>
                          <span>
                            <strong>
                              {Math.round(
                                audio.frames
                                  .filter((f) => f.pitch_hz)
                                  .reduce((a, f) => a + (f.pitch_hz || 0), 0) /
                                  Math.max(
                                    1,
                                    audio.frames.filter((f) => f.pitch_hz)
                                      .length,
                                  ),
                              )}{" "}
                              Hz
                            </strong>
                            Hauteur moyenne
                          </span>
                          <span>
                            <strong>
                              {Math.round(
                                (audio.frames.filter((f) => f.is_pause).length /
                                  audio.frames.length) *
                                  100,
                              )}{" "}
                              %
                            </strong>
                            Pauses
                          </span>
                        </div>
                        {emotion && (
                          <div className="emotion-results">
                            <span className="section-label">
                              EXPRESSIVITÉ ESTIMÉE · EMOTION2VEC+
                            </span>
                            {Object.entries(emotion)
                              .sort((a, b) => Number(b[1]) - Number(a[1]))
                              .slice(0, 3)
                              .map(([name, score]) => (
                                <div key={name}>
                                  <span>
                                    {(
                                      {
                                        happy: "Joie",
                                        neutral: "Neutre",
                                        sad: "Tristesse",
                                        angry: "Colère",
                                        surprised: "Surprise",
                                        fearful: "Peur",
                                        disgusted: "Dégoût",
                                        other: "Autre",
                                        unknown: "Inconnu",
                                      } as Record<string, string>
                                    )[name] || name}
                                  </span>
                                  <meter
                                    min={0}
                                    max={1}
                                    value={Number(score)}
                                  />
                                  <b>{Math.round(Number(score) * 100)} %</b>
                                </div>
                              ))}
                          </div>
                        )}
                        <label className="switch-row">
                          <input
                            type="checkbox"
                            checked={expressive}
                            onChange={(e) => setExpressive(e.target.checked)}
                          />
                          <span>
                            Explorer l’expressivité de la voix
                            <small>
                              Animation faciale expérimentale, sans validation
                              linguistique.
                            </small>
                          </span>
                        </label>
                      </>
                    )}
                  </div>
                  {plan && (
                    <div className="card sequence-card">
                      <div className="card-heading">
                        <span className="step">03</span>
                        <div>
                          <h2>Votre séquence</h2>
                          <p>
                            {plan.segments.length} mouvement
                            {plan.segments.length > 1 ? "s" : ""} ·{" "}
                            {seconds(plan.duration_ms)}
                          </p>
                        </div>
                      </div>
                      <div className="gloss-sequence">
                        {plan.segments.map((seg, i) => (
                          <button
                            key={`${seg.motion_id}-${i}`}
                            className={
                              active === seg
                                ? "gloss-token selected"
                                : "gloss-token"
                            }
                            onClick={() => {
                              setTime(seg.start_ms);
                              setPlaying(false);
                            }}
                          >
                            <span>{String(i + 1).padStart(2, "0")}</span>
                            <strong>
                              {glossary.find((g) => g.gloss_id === seg.gloss_id)
                                ?.gloss || seg.gloss_id}
                            </strong>
                            <small>{seconds(seg.duration_ms)}</small>
                          </button>
                        ))}
                      </div>
                      {plan.unknowns.length > 0 && (
                        <p className="missing-words">
                          <CircleHelp size={15} />
                          <span>
                            À compléter dans la bibliothèque :{" "}
                            <strong>{plan.unknowns.join(", ")}</strong>
                          </span>
                        </p>
                      )}
                      {!plan.segments.length && (
                        <p className="empty-copy">
                          Aucun mouvement ne correspond encore à ce texte.
                          Importez une vidéo pour enrichir votre bibliothèque.
                        </p>
                      )}
                    </div>
                  )}
                </>
              )}
              {page === "capture" && (
                <>
                  <div className="card capture-card">
                    <div className="card-heading">
                      <span className="step">01</span>
                      <div>
                        <h2>Importez un geste</h2>
                        <p>Une vidéo, un signe ou une expression</p>
                      </div>
                      <Film size={20} />
                    </div>
                    <input
                      ref={videoInput}
                      type="file"
                      accept="video/*,.json"
                      hidden
                      onChange={(e) => {
                        const file = e.target.files?.[0];
                        if (file) {
                          setCaptureFile(file);
                          setSourceUrl(
                            file.type.startsWith("video/")
                              ? URL.createObjectURL(file)
                              : "",
                          );
                          setJob(null);
                          setSaved(false);
                        }
                        e.target.value = "";
                      }}
                    />
                    <button
                      className="video-drop"
                      onClick={() => videoInput.current?.click()}
                    >
                      <span className="upload-orbit">
                        <Upload size={25} />
                      </span>
                      <strong>
                        {captureFile?.name || "Choisir une vidéo"}
                      </strong>
                      <span>MP4, MOV, WebM ou landmarks Animics JSON</span>
                      <small>
                        De face · buste et mains visibles · 50 Mo maximum
                      </small>
                    </button>
                    {sourceUrl && (
                      <video
                        src={sourceUrl}
                        controls
                        className="source-video"
                      />
                    )}
                    <label className="permission">
                      <input
                        type="checkbox"
                        checked={rights}
                        onChange={(e) => setRights(e.target.checked)}
                      />
                      <span>
                        Je suis autorisé à traiter ce fichier et les mouvements
                        qu’il contient.
                      </span>
                    </label>
                    <button
                      className="primary-button"
                      disabled={
                        !captureFile ||
                        !rights ||
                        busy ||
                        ["queued", "processing"].includes(job?.status || "")
                      }
                      onClick={() => void startCapture()}
                    >
                      {busy ||
                      ["queued", "processing"].includes(job?.status || "") ? (
                        <Loader2 className="spin" size={17} />
                      ) : (
                        <Waves size={17} />
                      )}
                      Extraire les mouvements
                      <ArrowRight size={17} />
                    </button>
                    {job && (
                      <div className="capture-progress">
                        <div>
                          <strong>
                            {job.status === "ready"
                              ? "Extraction terminée"
                              : job.status === "failed"
                                ? "Extraction interrompue"
                                : job.status === "queued"
                                  ? "En attente de traitement"
                                  : "Analyse du corps, des mains et du visage…"}
                          </strong>
                          <span>{job.progress}%</span>
                        </div>
                        <progress value={job.progress} max={100} />
                        {job.error && <p role="alert">{job.error}</p>}
                      </div>
                    )}
                  </div>
                  {job?.status === "ready" && (
                    <div className="card">
                      <div className="card-heading">
                        <span className="step">02</span>
                        <div>
                          <h2>Donnez un nom au mouvement</h2>
                          <p>
                            Enregistrez-le pour le retrouver depuis votre texte
                          </p>
                        </div>
                      </div>
                      {job.quality && (
                        <div className="tracking-quality">
                          {[
                            ["Corps", job.quality.pose_coverage],
                            ["Main gauche", job.quality.left_hand_coverage],
                            ["Main droite", job.quality.right_hand_coverage],
                            ["Visage", job.quality.face_coverage],
                          ].map(([name, value]) => (
                            <div key={name}>
                              <span>{name}</span>
                              <strong>
                                {Math.round(Number(value) * 100)} %
                              </strong>
                              <small>Images suivies</small>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="capture-form">
                        <label>
                          Concept français
                          <input
                            value={concept}
                            onChange={(e) => setConcept(e.target.value)}
                            placeholder="Ex. bonjour"
                          />
                        </label>
                        <label>
                          Autres expressions correspondantes
                          <input
                            value={aliases}
                            onChange={(e) => setAliases(e.target.value)}
                            placeholder="Séparées par des virgules"
                          />
                        </label>
                        <label>
                          Source de l’enregistrement
                          <input
                            value={source}
                            onChange={(e) => setSource(e.target.value)}
                            placeholder="Auteur, corpus ou référence"
                          />
                        </label>
                        <label>
                          Droits et autorisation
                          <input
                            value={licence}
                            onChange={(e) => setLicence(e.target.value)}
                            placeholder="Licence ou autorisation obtenue"
                          />
                        </label>
                      </div>
                      <button
                        className="primary-button"
                        disabled={
                          !concept.trim() ||
                          !source.trim() ||
                          !licence.trim() ||
                          busy ||
                          saved
                        }
                        onClick={() => void saveCapture()}
                      >
                        {saved ? <Check size={17} /> : <Plus size={17} />}
                        {saved
                          ? "Ajouté à votre bibliothèque"
                          : "Ajouter à la bibliothèque"}
                      </button>
                      <p className="quiet-note">
                        Le mouvement reste à relire pour vérifier le sens, les
                        mains et le visage.
                      </p>
                    </div>
                  )}
                </>
              )}
              {page === "library" && (
                <div className="card library-card">
                  <div className="card-heading">
                    <span className="file-icon">
                      <Library size={20} />
                    </span>
                    <div>
                      <h2>Vos mouvements capturés</h2>
                      <p>
                        {library.length} entrée{library.length > 1 ? "s" : ""}{" "}
                        dans votre bibliothèque locale
                      </p>
                    </div>
                    <button
                      className="icon-button"
                      aria-label="Importer un mouvement"
                      onClick={() => setPage("capture")}
                    >
                      <Plus />
                    </button>
                  </div>
                  {caps?.dictasign && (
                    <div className="dictasign-status">
                      <strong>Lexique Dicta-Sign · France-LSF</strong>
                      <p>
                        {caps.dictasign.imported} captures disponibles ·{" "}
                        {caps.dictasign.quality_review_needed} à vérifier ·{" "}
                        {caps.dictasign.failed} erreurs
                      </p>
                      <progress
                        aria-label="Progression Dicta-Sign"
                        max={caps.dictasign.indexed_concepts || 1}
                        value={caps.dictasign.processed}
                      />
                      <p>
                        {caps.dictasign.status === "complete"
                          ? "Extraction terminée"
                          : `Extraction en cours · ${caps.dictasign.inventory_processed}/${caps.dictasign.indexed_concepts} pages indexées`}
                      </p>
                      <a href="/api/datasets/dictasign/lexicon" download>
                        Exporter le lexique et les liens vers les squelettes
                      </a>
                    </div>
                  )}
                  {!!caps?.cv_corrected_count && (
                    <p className="dictasign-status">
                      {caps.cv_corrected_count} captures CV actualisées et actives · aperçu de recherche, revue en attente
                    </p>
                  )}
                  <input
                    className="library-search"
                    aria-label="Rechercher un concept"
                    placeholder="Rechercher un concept…"
                    value={search}
                    onChange={(e) => setSearch(e.target.value)}
                  />
                  {library.length === 0 ? (
                    <div className="empty-library">
                      <Hand size={38} />
                      <h3>Votre premier geste vous attend.</h3>
                      <p>
                        Importez une vidéo autorisée pour constituer votre
                        propre bibliothèque de mouvements.
                      </p>
                      <button
                        className="secondary-button"
                        onClick={() => setPage("capture")}
                      >
                        Importer une vidéo
                        <ArrowRight size={15} />
                      </button>
                    </div>
                  ) : (
                    <div className="library-list">
                      {library
                        .filter((x) =>
                          x.gloss.concept_fr
                            .toLowerCase()
                            .includes(search.toLowerCase()),
                        )
                        .map((entry) => (
                          <button
                            key={entry.motion.motion_id}
                            onClick={() =>
                              void loadEntry(
                                entry.motion.motion_id,
                                entry.gloss.concept_fr,
                              )
                            }
                          >
                            <span className="library-thumb">
                              <Hand />
                            </span>
                            <span>
                              <strong>{entry.gloss.concept_fr}</strong>
                              <small>
                                {entry.gloss.gloss} · {entry.source} ·{" "}
                                {seconds(entry.motion.duration_ms)}
                              </small>
                            </span>
                            <span className="review-pill">À relire</span>
                            <Play size={16} />
                          </button>
                        ))}
                    </div>
                  )}
                  <details className="technical-library">
                    <summary>
                      Mouvements techniques de démonstration{" "}
                      <span>
                        {
                          glossary.filter((g) =>
                            g.motion_id?.startsWith("SYNTHETIC"),
                          ).length
                        }
                      </span>
                    </summary>
                    <p>
                      Ces trajectoires synthétiques servent à tester le lecteur.
                      Elles ne représentent pas les signes LSF associés aux
                      étiquettes.
                    </p>
                    <div>
                      {glossary
                        .filter((g) => g.motion_id?.startsWith("SYNTHETIC"))
                        .map((entry) => (
                          <button
                            key={entry.gloss_id}
                            onClick={() =>
                              void loadEntry(
                                entry.motion_id!,
                                `${entry.concept_fr} · test technique`,
                              )
                            }
                          >
                            {entry.concept_fr}
                            <Play size={13} />
                          </button>
                        ))}
                    </div>
                  </details>
                </div>
              )}
            </section>
            <aside className="preview-column">
              <div className="preview-card">
                <div className="preview-heading">
                  <span>
                    <i />
                    APERÇU EN DIRECT
                  </span>
                  <button
                    className="icon-button"
                    aria-label="Revenir au début"
                    onClick={() => {
                      setTime(0);
                      setPlaying(false);
                    }}
                  >
                    <RotateCcw size={17} />
                  </button>
                </div>
                <AvatarCanvas
                  motion={motion}
                  currentTimeMs={time}
                  expressiveModulation={expressive}
                  energyFactor={
                    sample?.energy ? Math.min(1, sample.energy * 5) : 0
                  }
                  emotion={expressive ? emotion : null}
                  onCanvasReady={setCanvas}
                />
                <div className="preview-caption">
                  <span className="preview-avatar-icon">
                    <Hand size={17} />
                  </span>
                  <div>
                    <strong>{label}</strong>
                    <small>
                      {motion
                        ? `${motion.frames.length} images · ${motion.fps.toFixed(0)} fps`
                        : "Avatar procédural · prêt à animer"}
                    </small>
                  </div>
                  <span className="view-tag">3D</span>
                </div>
                <div className="player">
                  <div className="player-time">
                    <span>{clock(time)}</span>
                    <span>{clock(duration)}</span>
                  </div>
                  <input
                    type="range"
                    aria-label="Position de l’animation"
                    min={0}
                    max={duration || 1}
                    step={10}
                    value={time}
                    disabled={!canvas || !duration || recording}
                    onChange={(e) => {
                      setTime(Number(e.target.value));
                      setPlaying(false);
                    }}
                  />
                  <div className="transport">
                    <button
                      className="play-button"
                      onClick={togglePlay}
                      disabled={!canvas || !duration || recording}
                      aria-label={
                        playing ? "Mettre en pause" : "Lire l’animation"
                      }
                    >
                      {playing ? <Pause size={17} /> : <Play size={17} />}
                    </button>
                    <span>
                      {playing
                        ? "Lecture en cours"
                        : duration
                          ? "Prêt pour la lecture"
                          : "Votre animation apparaîtra ici"}
                    </span>
                    <select
                      aria-label="Vitesse de lecture"
                      value={speed}
                      onChange={(e) => setSpeed(Number(e.target.value))}
                      disabled={recording}
                    >
                      {[0.5, 0.75, 1, 1.25, 1.5].map((x) => (
                        <option value={x} key={x}>
                          {x}×
                        </option>
                      ))}
                    </select>
                  </div>
                </div>
              </div>
              <button
                className="export-button"
                disabled={!canvas || !duration || recording}
                onClick={() => void exportPreview()}
              >
                {recording ? (
                  <Loader2 className="spin" size={18} />
                ) : (
                  <ArrowDownToLine size={18} />
                )}
                <span>
                  {recording
                    ? "Enregistrement de l’aperçu…"
                    : "Télécharger l’aperçu vidéo"}
                  <small>WebM · animation 3D avec filigrane</small>
                </span>
                <ArrowRight size={17} />
              </button>
              <div className="preview-note">
                <CircleHelp size={15} />
                <p>
                  {page === "capture"
                    ? "Les points absents lors d’une occultation ne sont pas recréés. Vérifiez les mains avant de conserver le mouvement."
                    : motion?.synthetic
                      ? "Cette séquence contient des mouvements de test synthétiques. Ils ne représentent pas des signes LSF validés."
                      : "Reconstruction à partir de captures. Une revue LSF reste nécessaire avant une diffusion comme traduction."}
                </p>
              </div>
              {(
                motion?.attributions ||
                (motion?.attribution ? [motion.attribution] : [])
              ).map((a) => (
                <p className="source-credit" key={a.source_url}>
                  Source :{" "}
                  <a href={a.source_url} target="_blank" rel="noreferrer">
                    {a.credit}
                  </a>{" "}
                  ·{" "}
                  <a
                    href={a.license_url || a.source_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {a.license}
                  </a>
                </p>
              ))}
              {referenceUrl && (
                <details className="reference-video">
                  <summary>Comparer avec la vidéo d’origine</summary>
                  <video
                    key={referenceUrl}
                    src={referenceUrl}
                    controls
                    preload="metadata"
                    playsInline
                  />
                  <p>
                    Vérifiez la configuration des mains et les occultations dans
                    la source.
                  </p>
                </details>
              )}
              <div className="pipeline-strip">
                <span>Texte</span>
                <ChevronRight size={12} />
                <span>Gloss</span>
                <ChevronRight size={12} />
                <span>Mouvement</span>
                <ChevronRight size={12} />
                <strong>Avatar</strong>
              </div>
            </aside>
          </div>
          <footer className="studio-footer">
            <span>La culture se partage. Dans toutes ses langues.</span>
            <span>
              SignaCulture AI <i />
              Recherche & création
            </span>
          </footer>
        </main>
      </div>
    </div>
  );
}
