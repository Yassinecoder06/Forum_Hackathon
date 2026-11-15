import { useEffect, useState } from "react";
import { ArrowDown, ArrowUp, Plus, X } from "lucide-react";
import { request } from "../services/api";
import type { AnimationPlan, AudioFeatures } from "../types/contract";
type Concept = { id: string; concept: string };
export function ConceptComposer({
  text,
  audio,
  onPlan,
}: {
  text: string;
  audio: AudioFeatures | null;
  onPlan: (plan: AnimationPlan) => Promise<void>;
}) {
  const [vocabulary, setVocabulary] = useState<Concept[]>([]),
    [ids, setIds] = useState<string[]>([]);
  const [search, setSearch] = useState(""),
    [error, setError] = useState(""),
    [busy, setBusy] = useState(false);
  const [unsupported, setUnsupported] = useState<string[]>([]);
  useEffect(() => {
    void request<{ concepts: Concept[] }>("/concepts/vocabulary")
      .then((v) => setVocabulary(v.concepts))
      .catch(() => setError("Bibliothèque indisponible."));
  }, []);
  useEffect(() => {
    setIds([]);
    setUnsupported([]);
  }, [text]);
  async function compose() {
    setBusy(true);
    setError("");
    try {
      const plan = await request<AnimationPlan>("/compose-concepts", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          concept_ids: ids,
          unsupported: unsupported.filter((s) => s.trim()),
          audio_features: audio,
        }),
      });
      await onPlan(plan);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Composition impossible.");
    } finally {
      setBusy(false);
    }
  }
  function exportDraft() {
    const draft = {
      schema_version: "concept-review-v1",
      source_text: text,
      concepts: ids.map((id) => vocabulary.find((c) => c.id === id)),
      unsupported: unsupported.filter((s) => s.trim()),
      origin: "interactive_concept_draft",
      linguistic_review: "pending",
      translation_validated: false,
      created_at: new Date().toISOString(),
    };
    const url = URL.createObjectURL(
      new Blob([JSON.stringify(draft, null, 2)], { type: "application/json" }),
    );
    const link = document.createElement("a");
    link.href = url;
    link.download = "signaculture-fiche-revue.json";
    link.click();
    setTimeout(() => URL.revokeObjectURL(url), 10000);
  }
  function move(index: number, offset: number) {
    setIds((current) => {
      const next = [...current];
      [next[index], next[index + offset]] = [next[index + offset], next[index]];
      return next;
    });
  }
  return (
    <details className="concept-composer">
      <summary>Préparer et corriger les concepts</summary>
      <p>
        Choisissez les signes disponibles et leur ordre. Cette sélection reste
        un brouillon à vérifier, notamment pour les noms propres, la négation,
        les questions et les quantités. Elle ne valide pas la grammaire LSF.
      </p>
      <label>
        Rechercher un signe
        <input
          aria-label="Rechercher un concept"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Ex. bonjour, venir, demain…"
        />
      </label>
      <div className="concept-options">
        {vocabulary
          .filter(
            (c) =>
              search.trim() &&
              c.concept
                .toLocaleLowerCase("fr")
                .includes(search.toLocaleLowerCase("fr")),
          )
          .slice(0, 12)
          .map((c) => (
            <button
              key={c.id}
              onClick={() => {
                setIds((current) => [...current, c.id]);
                setSearch("");
              }}
              disabled={ids.length >= 40}
            >
              <Plus size={12} />
              {c.concept}
            </button>
          ))}
      </div>
      <ol className="concept-sequence">
        {ids.map((id, i) => (
          <li key={`${id}-${i}`}>
            <span>{vocabulary.find((c) => c.id === id)?.concept || id}</span>
            <button
              aria-label={`Monter le concept ${i + 1}`}
              disabled={i === 0}
              onClick={() => move(i, -1)}
            >
              <ArrowUp size={13} />
            </button>
            <button
              aria-label={`Descendre le concept ${i + 1}`}
              disabled={i === ids.length - 1}
              onClick={() => move(i, 1)}
            >
              <ArrowDown size={13} />
            </button>
            <button
              aria-label={`Retirer le concept ${i + 1}`}
              onClick={() =>
                setIds((current) => current.filter((_, j) => i !== j))
              }
            >
              <X size={13} />
            </button>
          </li>
        ))}
      </ol>
      <label>
        Sens non représenté (un élément par ligne)
        <textarea
          aria-label="Sens non représenté"
          value={unsupported.join("\n")}
          onChange={(e) => setUnsupported(e.target.value.split("\n"))}
        />
      </label>
      {unsupported.length > 0 && (
        <p role="status">Sens à traiter : {unsupported.join(" · ")}</p>
      )}
      {error && (
        <p role="alert" className="research-error">
          {error}
        </p>
      )}
      <button
        className="primary-button"
        disabled={!ids.length || busy}
        onClick={() => void compose()}
      >
        Prévisualiser cette sélection
      </button>
      <button
        className="secondary-button"
        disabled={!ids.length}
        onClick={exportDraft}
      >
        Télécharger la fiche de revue
      </button>
    </details>
  );
}
