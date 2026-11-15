import { useEffect, useRef, useState } from "react";
import { ArrowDown, ArrowUp, Loader2, Plus, X } from "lucide-react";
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
  const version = useRef(0);
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
    version.current += 1;
    setIds([]);
    setUnsupported([]);
  }, [text]);
  async function suggest() {
    const current = version.current;
    setBusy(true);
    setError("");
    try {
      const result = await request<{
        concept_ids: string[];
        unsupported: string[];
      }>("/concepts/propose", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text }),
      });
      if (current !== version.current) return;
      setIds(result.concept_ids);
      setUnsupported(result.unsupported);
    } catch (e) {
      setError(e instanceof Error ? e.message : "Proposition indisponible.");
    } finally {
      setBusy(false);
    }
  }
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
          unsupported,
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
      unsupported,
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
        Choisissez les signes disponibles et leur ordre. Une proposition
        automatique reste un brouillon ; elle ne valide pas la grammaire LSF.
        Vérifiez notamment les noms propres, la négation, les questions et les
        quantités : leur détection automatique reste imparfaite.
      </p>
      <button
        className="secondary-button"
        onClick={() => void suggest()}
        disabled={busy || !text.trim() || text.length > 500}
      >
        {busy ? <Loader2 size={14} className="spin" /> : null}Proposer des
        concepts
      </button>
      {text.length > 500 && (
        <p>
          La proposition automatique accepte 500 caractères. La sélection
          manuelle reste disponible.
        </p>
      )}
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
