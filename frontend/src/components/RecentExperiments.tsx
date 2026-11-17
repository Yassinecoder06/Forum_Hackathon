import { useState } from "react";

export type ExperimentLedger = {
  available: boolean;
  entries: {
    run_id: string;
    model_base: string;
    sign_language: "fsl" | "sfb";
    representation: string;
    evidence_status: string;
    inference_ready: false;
    train_examples?: number;
    input_provenance?: string;
    metrics: null | {
      examples: number | null;
      exact: number | null;
      wer: number | null;
      triple_f1: number | null;
      valid_graphs: number | null;
      multi_sign_examples: number | null;
      multi_sign_exact: number | null;
    };
  }[];
};
const percentage = (value: number | null) => value === null ? "—" : `${(value * 100).toFixed(2)} %`;
const tasks: Record<string, string> = {
  lexical_projection: "Projection lexicale LSFB",
  temporal_graph: "Graphe temporel LSFB",
  source_annotated_gloss: "Annotations glossées LSF",
};
export function RecentExperiments({ ledger }: { ledger?: ExperimentLedger }) {
  const [language, setLanguage] = useState("all");
  const entries = ledger?.entries.filter((r) => language === "all" || r.sign_language === language) ?? [];
  return <section className="card" aria-label="Bilan des expériences récentes">
    <h2>Bilan des expériences récentes</h2>
    <p>Résultats de développement, avec sélection sur ces données. Aucune validation linguistique locale ; ces modèles ne sont pas activés pour l’avatar.</p>
    <label>Langue des expériences{" "}
      <select aria-label="Langue des expériences" value={language} onChange={(e) => setLanguage(e.target.value)}>
        <option value="all">Toutes les langues</option>
        <option value="fsl">LSF — France</option>
        <option value="sfb">LSFB — Belgique</option>
      </select>
    </label>
    {!ledger?.available ? <p>Le catalogue récent n’est pas disponible dans cette installation.</p> : !entries.length ? <p>Aucune expérience enregistrée pour cette langue.</p> :
      <div style={{ display: "grid", gap: "1rem", marginTop: "1rem" }}>
        {entries.map((run) => <article key={run.run_id} data-run-id={run.run_id} style={{ border: "1px solid var(--line, #dce3d3)", borderRadius: "12px", padding: "1rem", minWidth: 0 }}>
          <h3>{run.sign_language === "fsl" ? "LSF — France" : "LSFB — Belgique"} · {run.model_base.split("/").pop()}</h3>
          <p>{tasks[run.representation] ?? run.representation}</p>
          <small style={{ overflowWrap: "anywhere" }}>{run.run_id}</small>
          {run.evidence_status !== "verified" || !run.metrics ? <p>Preuve absente ou modifiée : mesures masquées.</p> : <>
            <p>{run.train_examples ?? "—"} exemples d’entraînement · {run.metrics.examples ?? "—"} de développement</p>
            <div className="experiment-metrics">
              <div><strong>{percentage(run.metrics.exact)}</strong><span>Séquences exactes</span></div>
              <div><strong>{run.representation === "temporal_graph" ? `F1 : ${percentage(run.metrics.triple_f1)}` : `WER : ${percentage(run.metrics.wer)}`}</strong><span>{run.representation === "temporal_graph" ? "Recouvrement des triplets" : "Erreur sur les glosses"}</span></div>
            </div>
            <p>{run.representation === "temporal_graph" ? `${run.metrics.valid_graphs ?? "—"}/${run.metrics.examples ?? "—"} graphes valides` : run.metrics.multi_sign_examples !== null ? `${percentage(run.metrics.multi_sign_exact)} exacts sur ${run.metrics.multi_sign_examples} exemples multi-signes` : "Multi-signes : non mesurés séparément"}</p>
          </>}
        </article>)}
      </div>}
    <p className="research-note">Les tâches et jeux de données diffèrent : les scores ne forment pas un classement. WER peut dépasser 100 %. Les graphes GeoQuery utilisent un français traduit automatiquement par DeepL. Une structure valide ne garantit pas le sens ni une animation correcte.</p>
  </section>;
}
