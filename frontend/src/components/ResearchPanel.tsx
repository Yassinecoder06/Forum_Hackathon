import { useEffect, useState } from "react";
import {
  ArrowRight,
  BarChart3,
  Check,
  FlaskConical,
  Loader2,
} from "lucide-react";
import { request } from "../services/api";

type Metrics = {
  exact_match: number;
  token_wer: number;
  corpus_bleu: number;
  examples: number;
};
type Run = {
  run_id: string;
  model_base: string;
  epochs: number;
  train_seconds: number;
  cuda_device: string | null;
  peak_allocated_mib: number | null;
  inference_ready: boolean;
  counts: Record<string, number>;
  metrics: { val?: Record<string, Metrics>; test: Record<string, Metrics> };
};
type Dataset = {
  version: string;
  raw_segments?: number;
  derived_pairs?: number;
  strict_approved_pairs?: number;
  counts: Record<string, number>;
  limitations?: string[];
};
type Research = {
  lexical_video_references: number;
  datasets: Dataset[];
  runs: Run[];
  corpus: {
    total: number;
    processed: number;
    imported: number;
    status: string;
    failed: number;
  } | null;
};
type Prediction = {
  gloss_sequence: string[];
  translation_validated: boolean;
  model_base: string;
  run_id: string;
};
export function ResearchPanel() {
  const [status, setStatus] = useState<Research | null>(null);
  const [selected, setSelected] = useState("");
  const [text, setText] = useState("Bonjour à tous");
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState("");
  useEffect(() => {
    let active = true;
    const refresh = async () => {
      try {
        const value = await request<Research>("/research/status");
        if (active) {
          setStatus(value);
          setSelected(
            (current) =>
              current ||
              value.runs.filter((r) => r.inference_ready).slice(-1)[0]
                ?.run_id ||
              "",
          );
        }
      } catch {
        if (active) setError("Le bilan des expériences n’est pas disponible.");
      }
    };
    void refresh();
    const timer = setInterval(() => void refresh(), 15000);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, []);
  async function predict() {
    setBusy(true);
    setPrediction(null);
    setError("");
    try {
      setPrediction(
        await request<Prediction>("/research/predict", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text, run_id: selected }),
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Prédiction indisponible.");
    } finally {
      setBusy(false);
    }
  }
  const dataset = status?.datasets[status.datasets.length - 1];
  return (
    <>
      <div className="card">
        <div className="card-heading">
          <span className="file-icon">
            <BarChart3 size={20} />
          </span>
          <div>
            <h2>Les données, avant les promesses</h2>
            <p>Sources publiques et état réel des expériences</p>
          </div>
        </div>
        <div className="research-summary">
          <div>
            <strong>{status?.lexical_video_references ?? "—"}</strong>
            <span>Références de signes</span>
          </div>
          <div>
            <strong>
              {dataset
                ? (dataset.derived_pairs ??
                  Object.values(dataset.counts).reduce((a, b) => a + b, 0))
                : "—"}
            </strong>
            <span>Énoncés de recherche</span>
          </div>
          <div>
            <strong>{status?.corpus?.imported ?? 0}</strong>
            <span>Captures importées</span>
          </div>
          <div>
            <strong>{dataset?.strict_approved_pairs ?? 0}</strong>
            <span>Validations LSF locales</span>
          </div>
        </div>
        {status?.corpus && (
          <div className="corpus-progress">
            <span>
              {status.corpus.status === "running"
                ? "Extraction des vidéos en cours"
                : "Extraction terminée"}{" "}
              · {status.corpus.processed}/{status.corpus.total}
            </span>
            <progress
              max={status.corpus.total}
              value={status.corpus.processed}
            />
          </div>
        )}
        <p className="research-note">
          Les signes isolés servent à la bibliothèque de mouvements. Les énoncés
          annotés servent à des expériences textuelles distinctes ; les
          traductions françaises libres ne constituent pas un alignement mot à
          mot.
        </p>
        <div className="source-links">
          <a
            href="https://github.com/UniversalDependencies/UD_French_Sign_Language-STKAutogramm"
            target="_blank"
            rel="noreferrer"
          >
            STKAutogramm · CC BY-SA 4.0 ↗
          </a>
          <a
            href="https://commons.wikimedia.org/wiki/Category:Vidéos_LSF_par_Laura_Jauvert"
            target="_blank"
            rel="noreferrer"
          >
            Vidéos LSF · licences par fichier ↗
          </a>
        </div>
      </div>
      <div className="card">
        <div className="card-heading">
          <span className="step">
            <FlaskConical size={15} />
          </span>
          <div>
            <h2>Expériences mesurées</h2>
            <p>Prédictions sur des groupes tenus à l’écart de l’entraînement</p>
          </div>
        </div>
        {!status?.runs.length && (
          <p className="empty-copy">
            Les expériences terminées apparaîtront ici avec leurs mesures.
          </p>
        )}
        {status?.runs.map((run) => {
          const model = run.metrics.test.model,
            base = run.metrics.test.retrieval;
          return (
            <article className="experiment" key={run.run_id}>
              <div className="experiment-title">
                <strong>
                  {run.model_base.split("/").pop()} <small>LoRA</small>
                </strong>
                <span>
                  <Check size={11} />
                  Exécuté
                </span>
              </div>
              <p className="run-id">
                {run.run_id} · {run.epochs} époques ·{" "}
                {run.train_seconds.toFixed(1)} s
              </p>
              <div className="experiment-metrics">
                <div>
                  <strong>{(model.exact_match * 100).toFixed(1)} %</strong>
                  <span>Séquences exactes</span>
                </div>
                <div>
                  <strong>{(model.token_wer * 100).toFixed(1)} %</strong>
                  <span>WER · plus bas = mieux</span>
                </div>
                <div>
                  <strong>{model.corpus_bleu.toFixed(2)}</strong>
                  <span>BLEU · échelle 0–100</span>
                </div>
              </div>
              <div className="baseline-comparison">
                <span>
                  {model.examples} exemples de test · retrieval WER{" "}
                  {(base.token_wer * 100).toFixed(1)} %
                </span>
                <strong>
                  {model.exact_match === 0
                    ? "Traduction non démontrée"
                    : model.token_wer < base.token_wer
                      ? "WER inférieure au retrieval"
                      : "Le retrieval reste meilleur"}
                </strong>
              </div>
              {run.cuda_device && (
                <p className="run-hardware">
                  {run.cuda_device} · VRAM pic{" "}
                  {Math.round(run.peak_allocated_mib || 0)} Mio
                </p>
              )}
            </article>
          );
        })}
        <p className="research-note">
          Ce corpus est réduit et les cibles proviennent d’annotations publiées,
          sans nouvelle validation linguistique locale. Ces mesures ne
          démontrent ni une traduction générale ni un résultat de l’état de
          l’art.
        </p>
      </div>
      <div className="card">
        <div className="card-heading">
          <span className="step">TEST</span>
          <div>
            <h2>Interrogez un checkpoint</h2>
            <p>Une sortie candidate, distincte du moteur d’animation</p>
          </div>
        </div>
        <label className="research-label">
          Modèle local
          <select
            aria-label="Modèle de recherche"
            value={selected}
            onChange={(e) => {
              setSelected(e.target.value);
              setPrediction(null);
            }}
          >
            <option value="">Choisir une expérience</option>
            {status?.runs.map((run) => (
              <option
                key={run.run_id}
                value={run.run_id}
                disabled={!run.inference_ready}
              >
                {run.run_id}
                {!run.inference_ready ? " · checkpoint absent" : ""}
              </option>
            ))}
          </select>
        </label>
        <label className="research-label">
          Phrase à explorer
          <textarea
            aria-label="Phrase de recherche"
            value={text}
            maxLength={500}
            onChange={(e) => setText(e.target.value)}
            rows={3}
          />
        </label>
        <button
          className="primary-button"
          onClick={() => void predict()}
          disabled={!selected || !text.trim() || busy}
        >
          {busy ? (
            <Loader2 size={16} className="spin" />
          ) : (
            <FlaskConical size={16} />
          )}
          {busy ? "Inférence locale…" : "Générer des glosses candidates"}
          <ArrowRight size={16} />
        </button>
        {error && (
          <p role="alert" className="research-error">
            {error}
          </p>
        )}
        {prediction && (
          <div className="candidate-result">
            <span className="section-label">
              SORTIE BRUTE DU MODÈLE · NON VALIDÉE
            </span>
            <p>{prediction.gloss_sequence.join(" ") || "(sortie vide)"}</p>
            <small>
              Les glosses inconnues ne sont pas remplacées par des gestes
              ressemblants.
            </small>
          </div>
        )}
      </div>
    </>
  );
}
