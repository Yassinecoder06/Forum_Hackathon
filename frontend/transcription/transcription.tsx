/* eslint-disable react-refresh/only-export-components */
// Whisper alone: what the live interpreter hears, before any signing. Same stream as the interpreter
// (presentation/liveStream.ts: a pass every ~0.7 s, two passes must agree, sentence cut, French filter).
// Black = confirmed words, grey = this pass's guess (can still change), underlined = low confidence.
import { createRoot } from "react-dom/client";
import { useCallback, useEffect, useRef, useState } from "react";
import { request } from "../src/services/api";
import type { PhraseDetector } from "../src/utils/phraseDetector";
import { openLiveStream, type LiveText, type Word } from "../presentation/liveStream";
import "../src/fonts.css";
import "./transcription.css";

type Done = { at: number; text: string; words: Word[]; kept: boolean; why?: string };
const WHY: Record<string, string> = { noise: "bruit ou musique", unsure: "Whisper peu sûr", "not-french": "pas en français" };
const LOW = 0.5;
const mean = (ws: Word[]) => { const p = ws.flatMap((w) => (typeof w.probability === "number" ? [w.probability] : [])); return p.length ? p.reduce((a, b) => a + b, 0) / p.length : null; };

function Words({ words, className }: { words: Word[]; className: string }) {
  return <>{words.map((w, i) => (
    <span key={i} className={`${className}${typeof w.probability === "number" && w.probability < LOW ? " low" : ""}`}
      title={typeof w.probability === "number" ? `confiance ${Math.round(w.probability * 100)} %` : undefined}>{w.text} </span>))}</>;
}

function Transcription() {
  const [ready, setReady] = useState<"loading" | "ready" | "off">("loading");
  const [on, setOn] = useState(false);
  const [live, setLive] = useState<LiveText>({ sure: "", tentative: "" });
  const [done, setDone] = useState<Done[]>([]);
  const [passes, setPasses] = useState<number[]>([]); // ms per Whisper pass (last 20)
  const [error, setError] = useState("");
  const [, tick] = useState(0);
  const detector = useRef<PhraseDetector | null>(null);
  const stop = useRef<() => void>(() => {});
  const t0 = useRef(0);

  useEffect(() => {
    request("/live/warmup?engine=c0", { method: "POST", signal: AbortSignal.timeout(280000) }).then(() => setReady("ready")).catch(() => setReady("off"));
    return () => stop.current();
  }, []);
  useEffect(() => { // level meter
    let raf = 0;
    const f = () => { tick((n) => n + 1); raf = requestAnimationFrame(f); };
    raf = requestAnimationFrame(f);
    return () => cancelAnimationFrame(raf);
  }, []);

  const start = useCallback(async () => {
    setError("");
    try {
      t0.current = performance.now();
      const mic = await openLiveStream({
        transcribe: async (wav) => {
          const started = performance.now();
          const form = new FormData();
          form.append("file", new Blob([wav], { type: "audio/wav" }), "live.wav");
          const r = await request<{ segments?: { words?: Word[] }[] }>("/live/transcribe", { method: "POST", body: form });
          setPasses((p) => [...p.slice(-19), performance.now() - started]);
          return (r.segments ?? []).flatMap((x) => x.words ?? []);
        },
        onSentence: (text, voice) => setDone((d) => [{ at: performance.now() - t0.current, text, words: voice.words, kept: true }, ...d].slice(0, 200)),
        onRejected: (r) => setDone((d) => [{ at: performance.now() - t0.current, text: r.text, words: r.words, kept: false, why: WHY[r.why] ?? r.why }, ...d].slice(0, 200)),
        onLive: setLive,
        onError: (e) => setError(e instanceof Error ? e.message : "Transcription impossible."),
      });
      detector.current = mic.detector;
      stop.current = () => { void mic.stop(); detector.current = null; stop.current = () => {}; setOn(false); };
      setOn(true);
    } catch {
      setError("Micro refusé ou indisponible.");
    }
  }, []);
  const toggle = useCallback(() => (on ? stop.current() : ready === "ready" ? void start() : undefined), [on, ready, start]);
  useEffect(() => {
    const key = (e: KeyboardEvent) => { if (e.key.toLowerCase() === "m" && !e.ctrlKey && !e.metaKey && !e.altKey) toggle(); };
    window.addEventListener("keydown", key);
    return () => window.removeEventListener("keydown", key);
  }, [toggle]);

  const level = detector.current ? Math.min(1, detector.current.level * 12) : 0;
  const speaking = !!detector.current?.speaking;
  const avg = passes.length ? passes.reduce((a, b) => a + b, 0) / passes.length : 0;
  const status = ready === "loading" ? "Chargement de Whisper…" : ready === "off" ? "Reconnaissance vocale indisponible"
    : !on ? "Micro coupé" : speaking ? "J’écoute" : "À l’écoute";

  return (
    <main className="tr">
      <header>
        <span className={`dot${on ? " on" : ""}${speaking ? " speaking" : ""}`} />
        <h1>Transcription en direct <small>Whisper large-v3-turbo</small></h1>
        <span className="status">{status}</span>
        <button className={`mic${on ? " on" : ""}`} disabled={ready !== "ready"} onClick={toggle} title="Micro (M)">
          {on ? "Couper le micro" : "Activer le micro"}
        </button>
      </header>
      <div className="meter"><i style={{ width: `${Math.round(level * 100)}%` }} /></div>
      <section className="now" aria-live="polite">
        <p className="label">Phrase en cours</p>
        <p className="text">
          {live.sureWords ? <Words words={live.sureWords} className="sure" /> : <span className="sure">{live.sure} </span>}
          {live.tentativeWords ? <Words words={live.tentativeWords} className="guess" /> : <span className="guess">{live.tentative}</span>}
          {!live.sure && !live.tentative && <span className="empty">{on ? "Parlez…" : "Activez le micro (M)"}</span>}
        </p>
        <p className="legend"><span className="sure">noir</span> confirmé (deux passes d’accord) · <span className="guess">gris</span> hypothèse en cours · <span className="sure low">souligné</span> confiance &lt; 50 %
          {passes.length > 0 && <> · passe Whisper : {Math.round(avg)} ms en moyenne</>}</p>
      </section>
      <section className="done">
        <p className="label">Phrases terminées ({done.filter((d) => d.kept).length} gardées, {done.filter((d) => !d.kept).length} écartées)</p>
        <ol>
          {done.map((d, i) => {
            const m = mean(d.words);
            return (
              <li key={done.length - i} className={d.kept ? "kept" : "dropped"}>
                <span className="time">{(d.at / 1000).toFixed(1)} s</span>
                <span className="sentence"><Words words={d.words} className="sure" /></span>
                <span className="tag">{d.kept ? "gardée" : `écartée : ${d.why}`}{m !== null && ` · ${Math.round(m * 100)} %`}</span>
              </li>
            );
          })}
        </ol>
      </section>
      {error && <p role="alert" className="error">{error}</p>}
    </main>
  );
}

createRoot(document.getElementById("root")!).render(<Transcription />);
