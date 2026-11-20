import { useEffect, useRef, useState } from "react";
import { ListOrdered, Loader2, Wand2 } from "lucide-react";
import { request } from "../services/api";
import type { GlossAnalysis, MotionFile } from "../types/contract";
import { EMOTION_LABEL, MARKER_LABEL, dominantEmotion, generateBody, voiceScores, type LsfEngine } from "../utils/lsfRequest";

const ENGINES: { id: LsfEngine; label: string; hint: string }[] = [
  {
    id: "c0",
    label: "C0 · générateur texte → poses",
    hint: "Modèle génératif texte → mouvement, avec pauses naturelles : couvre toute phrase.",
  },
  {
    id: "b2",
    label: "B2 · signes du lexique enchaînés",
    hint: "Signes du lexique (942 entrées) enchaînés par des transitions apprises sur des interprètes.",
  },
];

/** Generate a motion with the LSF engines (POST /lsf/generate) and play it. */
export function LsfEnginePanel({
  text,
  available,
  disabled,
  onMotion,
  voiceEmotion,
}: {
  text: string;
  available?: Partial<Record<LsfEngine, boolean>>;
  disabled?: boolean;
  onMotion: (motion: MotionFile, label: string) => void;
  /** emotion2vec hypothesis of the voice imported in step 02, if any. */
  voiceEmotion?: Record<string, number | string> | null;
}) {
  const [engine, setEngine] = useState<LsfEngine>("c0"),
    [lsfOrder, setLsfOrder] = useState(true),
    [nonmanual, setNonmanual] = useState(true),
    [analysis, setAnalysis] = useState<GlossAnalysis | null>(null),
    [edited, setEdited] = useState<string[] | null>(null),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [summary, setSummary] = useState(""),
    [feeling, setFeeling] = useState<string>("none");
  const voice = voiceScores(voiceEmotion);
  const voiceTop = dominantEmotion(voice);
  const emotionScores = feeling === "voice" ? voice : feeling === "none" ? null : { [feeling]: 1 };
  const version = useRef(0);
  useEffect(() => {
    version.current++;
    setError("");
    setSummary("");
    setAnalysis(null);
    setEdited(null);
  }, [text]);
  useEffect(() => setSummary(""), [engine, lsfOrder, nonmanual, feeling]);
  useEffect(() => { if (feeling === "voice" && !voice) setFeeling("none"); }, [feeling, voice]);
  const ready = available?.[engine] !== false;

  async function proposeOrder() {
    const current = version.current;
    setError("");
    try {
      const result = await request<GlossAnalysis>("/lsf/order", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (current !== version.current) return null;
      setAnalysis(result);
      setEdited(result.clauses.map((c) => c.glosses.join(" ")));
      return result;
    } catch (e) {
      if (current === version.current) setError(e instanceof Error ? e.message : "Ordre LSF indisponible.");
      return null;
    }
  }

  async function run() {
    const current = version.current;
    setBusy(true);
    setError("");
    setSummary("");
    try {
      // Send the gloss lines only if the user changed the proposed order.
      const changed =
        analysis && edited && edited.some((line, i) => line.trim() !== analysis.clauses[i]?.glosses.join(" "));
      const glosses = changed ? edited!.map((line) => line.split(/\s+/).filter(Boolean)) : null;
      const motion = await request<MotionFile>("/lsf/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(generateBody(text, engine, lsfOrder, nonmanual, glosses, emotionScores)),
      });
      if (current !== version.current) return;
      const g = motion.generation;
      if (g?.gloss_analysis && !analysis) {
        setAnalysis(g.gloss_analysis);
        setEdited(g.gloss_analysis.clauses.map((c) => c.glosses.join(" ")));
      }
      const order = g?.order === "edited" ? " (ordre modifié)" : g?.order === "lsf" ? " (ordre LSF)" : "";
      const markers = (motion.nonmanual ?? []).map((m) => MARKER_LABEL[m.type]);
      const e = g?.emotion;
      const emotionText = e && e.label !== "neutral" ? ` — émotion : ${EMOTION_LABEL[e.label] ?? e.label} (vitesse ×${e.speed.toFixed(2)}, amplitude ×${e.amplitude.toFixed(2)})` : "";
      setSummary(
        (engine === "b2"
          ? g?.lemmas?.length
            ? `Signes du lexique${order} : ${g.lemmas.join(" · ")}` + (g.unknown?.length ? ` — hors lexique : ${g.unknown.join(", ")}` : "")
            : "Aucun mot de la phrase n’est dans le lexique : position de repos."
          : `${(motion.duration_ms / 1000).toFixed(1)} s générées`) + (markers.length ? ` — visage : ${[...new Set(markers)].join(", ")}` : "") + emotionText,
      );
      onMotion(motion, engine === "b2" ? "Moteur B2" : "Moteur C0");
    } catch (e) {
      if (current === version.current) setError(e instanceof Error ? e.message : "Moteur indisponible.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="concept-composer lsf-engine-panel">
      <summary>Générer avec les moteurs LSF</summary>
      <p>
        Mouvement produit à partir du texte par des modèles entraînés sur des
        interprètes LSF.
      </p>
      <div role="radiogroup" aria-label="Moteur" className="lsf-engine-choice">
        {ENGINES.map((e) => (
          <label key={e.id}>
            <input
              type="radio"
              name="lsf-engine"
              value={e.id}
              checked={engine === e.id}
              disabled={busy}
              onChange={() => setEngine(e.id)}
            />
            <span>
              <strong>{e.label}</strong>
              <small>
                {e.hint}
                {available?.[e.id] === false ? " Modèle non installé sur ce serveur." : ""}
              </small>
            </span>
          </label>
        ))}
      </div>
      <div className="lsf-options">
        <label>
          <input type="checkbox" checked={lsfOrder} disabled={busy || engine !== "b2"} onChange={(e) => setLsfOrder(e.target.checked)} />
          <span>
            Ordre des signes LSF (temps, lieu, sujet, objet, verbe, négation, mot interrogatif)
            {engine !== "b2" && <small> — C0 lit le texte français tel quel.</small>}
          </span>
        </label>
        <label>
          <input type="checkbox" checked={nonmanual} disabled={busy} onChange={(e) => setNonmanual(e.target.checked)} />
          <span>Visage grammatical (questions : sourcils et tête ; négation : hochement de tête)</span>
        </label>
        <label className="lsf-emotion">
          <span>Émotion de la signature</span>
          <select value={feeling} disabled={busy} onChange={(e) => setFeeling(e.target.value)}>
            <option value="none">Neutre</option>
            {voice && (
              <option value="voice">
                Voix importée : {EMOTION_LABEL[voiceTop?.label ?? "neutral"] ?? voiceTop?.label} ({Math.round((voiceTop?.confidence ?? 0) * 100)} %)
              </option>
            )}
            {["happy", "angry", "sad", "surprised", "fearful"].map((k) => (
              <option key={k} value={k}>{EMOTION_LABEL[k]}</option>
            ))}
          </select>
          <small>Module la vitesse et l’amplitude des mains ; le visage grammatical reste prioritaire.</small>
        </label>
      </div>
      {engine === "b2" && lsfOrder && (
        <div className="gloss-order">
          <button className="secondary-button" disabled={busy || !text.trim() || text.length > 400} onClick={() => void proposeOrder()}>
            <ListOrdered size={15} /> {analysis ? "Recalculer l’ordre LSF" : "Proposer l’ordre LSF"}
          </button>
          {analysis?.clauses.map((clause, i) => (
            <div className="gloss-clause" key={i}>
              <small className="gloss-source">« {clause.text} »</small>
              <input
                aria-label={`Gloses de la proposition ${i + 1}`}
                value={edited?.[i] ?? clause.glosses.join(" ")}
                disabled={busy}
                onChange={(e) => setEdited((prev) => (prev ?? analysis.clauses.map((c) => c.glosses.join(" "))).map((v, j) => (j === i ? e.target.value : v)))}
              />
              <div className="gloss-meta">
                {clause.markers.map((m) => (
                  <span key={m} className="small-chip">{MARKER_LABEL[m]}</span>
                ))}
                {clause.rules.length > 0 && <small>Règles : {clause.rules.join(" · ")}</small>}
              </div>
            </div>
          ))}
        </div>
      )}
      <button
        className="primary-button"
        onClick={() => void run()}
        disabled={busy || disabled || !ready || !text.trim() || text.length > 400}
      >
        {busy ? <Loader2 className="spin" size={16} /> : <Wand2 size={16} />}
        {busy ? "Génération en cours…" : "Générer et animer l’avatar"}
      </button>
      {text.length > 400 && <p className="research-error">400 caractères au maximum pour les moteurs.</p>}
      {error && (
        <p role="alert" className="research-error">
          {error}
        </p>
      )}
      {summary && <p className="lsf-engine-summary">{summary}</p>}
    </details>
  );
}
