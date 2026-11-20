import { useEffect, useRef, useState } from "react";
import { Loader2, Wand2 } from "lucide-react";
import { request } from "../services/api";
import type { MotionFile } from "../types/contract";

export type LsfEngine = "b2" | "c0";

const ENGINES: { id: LsfEngine; label: string; hint: string }[] = [
  {
    id: "b2",
    label: "B2 · signes du lexique enchaînés",
    hint: "Clips de mots (942 entrées) reliés par des transitions apprises. Les mots absents du lexique sont ignorés.",
  },
  {
    id: "c0",
    label: "C0 · générateur texte → poses",
    hint: "Modèle génératif : couvre toute phrase, mais les signes produits ne sont pas garantis.",
  },
];

/** Generate a motion with the experimental LSF engines (POST /lsf/generate) and play it. */
export function LsfEnginePanel({
  text,
  available,
  disabled,
  onMotion,
}: {
  text: string;
  available?: Partial<Record<LsfEngine, boolean>>;
  disabled?: boolean;
  onMotion: (motion: MotionFile, label: string) => void;
}) {
  const [engine, setEngine] = useState<LsfEngine>("b2"),
    [busy, setBusy] = useState(false),
    [error, setError] = useState(""),
    [summary, setSummary] = useState("");
  const version = useRef(0);
  useEffect(() => {
    version.current++;
    setError("");
    setSummary("");
  }, [text, engine]);
  const ready = available?.[engine] !== false;
  async function run() {
    const current = version.current;
    setBusy(true);
    setError("");
    setSummary("");
    try {
      const motion = await request<MotionFile>("/lsf/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text, engine }),
      });
      if (current !== version.current) return;
      const g = motion.generation;
      setSummary(
        engine === "b2"
          ? g?.lemmas?.length
            ? `Signes du lexique : ${g.lemmas.join(" · ")}` + (g.unknown?.length ? ` — hors lexique : ${g.unknown.join(", ")}` : "")
            : "Aucun mot de la phrase n’est dans le lexique : position de repos."
          : `${(motion.duration_ms / 1000).toFixed(1)} s générées`,
      );
      onMotion(motion, engine === "b2" ? "Moteur B2" : "Moteur C0");
    } catch (e) {
      if (current === version.current)
        setError(e instanceof Error ? e.message : "Moteur indisponible.");
    } finally {
      setBusy(false);
    }
  }
  return (
    <details className="concept-composer lsf-engine-panel">
      <summary>Générer avec les moteurs LSF expérimentaux</summary>
      <p>
        Mouvement produit automatiquement à partir du texte par des modèles
        entraînés sur des interprètes LSF. Ce n’est pas une traduction validée :
        aucune personne sourde ou interprète n’a encore relu ces sorties.
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
      <button
        className="primary-button"
        onClick={() => void run()}
        disabled={busy || disabled || !ready || !text.trim() || text.length > 400}
      >
        {busy ? <Loader2 className="spin" size={16} /> : <Wand2 size={16} />}
        {busy ? "Génération en cours (le premier appel charge les modèles)…" : "Générer et animer l’avatar"}
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
