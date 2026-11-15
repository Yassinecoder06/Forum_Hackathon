import React, { useState, useRef } from 'react';
import {
  Send,
  Sparkles,
  HelpCircle,
  PlayCircle,
  FileCheck2,
  AlertTriangle,
  Clock,
  Layers,
  Film,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import { AnimationPlan, TranslationDraft } from '../../types/contract';
import { Badge } from '../Common/Badge';
import { ApiErrorAlert } from '../Common/ApiErrorAlert';

interface TextToSignStudioProps {
  onLoadPlanSequence: (plan: AnimationPlan) => Promise<void>;
  onLoadSingleSegment: (motionId: string, assetUrl?: string, label?: string) => Promise<void>;
  onClearMotion: () => void;
  onPlanCreated?: (plan: AnimationPlan | null) => void;
}

const EXAMPLE_PHRASES = [
  'bonjour',
  'merci',
  'bonjour merci',
  'bienvenue au musée',
  'culture et art',
];

export const TextToSignStudio: React.FC<TextToSignStudioProps> = ({
  onLoadPlanSequence,
  onLoadSingleSegment,
  onClearMotion,
  onPlanCreated,
}) => {
  const [inputText, setInputText] = useState('bonjour merci');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [draft, setDraft] = useState<TranslationDraft | null>(null);
  const [plan, setPlan] = useState<AnimationPlan | null>(null);

  // Request-generation guard: prevents out-of-order responses from replacing newer state
  const requestIdRef = useRef(0);

  const handleCompose = async (textToUse?: string) => {
    if (loading) return; // Prevent concurrent submissions

    const text = (textToUse ?? inputText).trim();
    if (!text) return;

    const currentReqId = ++requestIdRef.current;

    // Immediately clear loaded motion/plan/play state on new request
    onClearMotion();
    setDraft(null);
    setPlan(null);
    if (onPlanCreated) {
      onPlanCreated(null);
    }

    setLoading(true);
    setError(null);

    try {
      // 1. Gloss lookup / text analysis
      const draftResult = await apiClient.glossLookup(text);
      if (requestIdRef.current !== currentReqId) return;
      setDraft(draftResult);

      // 2. Compose animation plan
      const planResult = await apiClient.composeAnimation(text);
      if (requestIdRef.current !== currentReqId) return;
      setPlan(planResult);
      if (onPlanCreated) {
        onPlanCreated(planResult);
      }

      // 3. Multi-clip sequence playback: stitch and play the full plan across all segments
      // If zero segments (unknown-only composition), player remains cleared.
      if (planResult.segments && planResult.segments.length > 0) {
        await onLoadPlanSequence(planResult);
      } else {
        onClearMotion();
      }
    } catch (err: unknown) {
      if (requestIdRef.current === currentReqId) {
        setError(err instanceof Error ? err : new Error(String(err)));
        onClearMotion();
      }
    } finally {
      if (requestIdRef.current === currentReqId) {
        setLoading(false);
      }
    }
  };

  const handleSelectExample = (phrase: string) => {
    if (loading) return;
    setInputText(phrase);
    handleCompose(phrase);
  };

  return (
    <div className="space-y-6">
      {/* Input Header & Form */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-sky-500/10 border border-sky-500/20 rounded-xl text-sky-400">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-base">Texte Français vers Gloss & Animation 3D</h3>
              <p className="text-xs text-slate-400">
                Lookup déterministe et composition multi-clips sur rig v1 (P0/P1).
              </p>
            </div>
          </div>

          <Badge variant="warning" size="sm">
            Statut : Non validé LSF
          </Badge>
        </div>

        {/* Input Text Form with Accessible Label */}
        <div className="space-y-1.5">
          <label htmlFor="text-to-sign-input" className="block text-xs font-medium text-slate-300">
            Saisissez votre texte français pour générer la séquence 3D :
          </label>
          <div className="flex flex-col sm:flex-row gap-3">
            <div className="relative flex-1">
              <input
                id="text-to-sign-input"
                type="text"
                value={inputText}
                disabled={loading}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !loading) {
                    handleCompose();
                  }
                }}
                placeholder="Ex : bonjour merci, bienvenue au musée, art..."
                aria-label="Texte français à transcrire en LSF"
                className="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-4 py-3 text-slate-100 text-sm placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/50 focus:border-sky-500 disabled:opacity-50 disabled:cursor-not-allowed"
              />
            </div>

            <button
              onClick={() => handleCompose()}
              disabled={loading || !inputText.trim()}
              className="px-5 py-3 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-slate-950 font-bold text-sm flex items-center justify-center gap-2 shadow-lg shadow-sky-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed shrink-0"
            >
              {loading ? (
                <>
                  <div className="w-4 h-4 border-2 border-slate-950 border-t-transparent rounded-full animate-spin" />
                  <span>Composition en cours...</span>
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" />
                  <span>Analyser & Composer</span>
                </>
              )}
            </button>
          </div>
        </div>

        {/* Quick Examples */}
        <div className="flex flex-wrap items-center gap-2 pt-1 text-xs text-slate-400">
          <span className="text-slate-500 font-medium">Exemples contrôlés :</span>
          {EXAMPLE_PHRASES.map((phrase) => (
            <button
              key={phrase}
              disabled={loading}
              onClick={() => handleSelectExample(phrase)}
              className="px-2.5 py-1 rounded-lg bg-slate-800/80 hover:bg-slate-700 hover:text-sky-300 border border-slate-700/60 text-slate-300 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {phrase}
            </button>
          ))}
        </div>
      </div>

      {/* API Error Notification */}
      {error && (
        <ApiErrorAlert
          error={error}
          onRetry={() => handleCompose()}
          title="Échec de l'appel API /api/gloss-lookup ou /api/compose-animation"
        />
      )}

      {/* Analysis Draft & Gloss Chips */}
      {draft && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <h4 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
              <Layers className="w-4 h-4 text-sky-400" />
              Séquence de Gloss Candidats (Baseline NLP)
            </h4>
            <span className="text-xs text-amber-400 font-medium">
              Validation linguistique : <strong className="underline">Non approuvée</strong>
            </span>
          </div>

          {/* Tokens & Lemmas Breakdown */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
              <span className="text-slate-400 block mb-1">Tokens analysés :</span>
              <div className="flex flex-wrap gap-1.5 font-mono">
                {draft.tokens.map((token, i) => (
                  <span key={i} className="px-2 py-0.5 rounded bg-slate-800 text-slate-200 border border-slate-700">
                    {token}
                  </span>
                ))}
              </div>
            </div>

            <div className="bg-slate-950/60 p-3 rounded-xl border border-slate-800/80">
              <span className="text-slate-400 block mb-1">Lemmes extraits :</span>
              <div className="flex flex-wrap gap-1.5 font-mono">
                {draft.lemmas.map((lemma, i) => (
                  <span key={i} className="px-2 py-0.5 rounded bg-slate-800 text-sky-300 border border-slate-700">
                    {lemma}
                  </span>
                ))}
              </div>
            </div>
          </div>

          {/* Gloss Sequence Chips */}
          <div className="space-y-2">
            <span className="text-xs text-slate-400 font-medium block">
              Glosses LSF résolus ({draft.gloss_sequence.length}) :
            </span>
            <div className="flex flex-wrap gap-2">
              {draft.gloss_sequence.length > 0 ? (
                draft.gloss_sequence.map((g, idx) => (
                  <div
                    key={`${g.gloss_id}-${idx}`}
                    className="flex items-center gap-2 px-3 py-2 rounded-xl bg-slate-950 border border-indigo-500/40 shadow-sm"
                  >
                    <div className="font-mono font-bold text-sm text-indigo-300">
                      [{g.gloss}]
                    </div>
                    <span className="text-[11px] text-slate-400">({g.concept_fr})</span>
                    <Badge variant="warning" size="sm">
                      {g.review_status}
                    </Badge>
                  </div>
                ))
              ) : (
                <div className="text-xs text-amber-300 italic py-2">
                  Aucun gloss contrôlé n'a pu être apparié pour cette phrase. Le lecteur 3D est réinitialisé.
                </div>
              )}
            </div>
          </div>

          {/* Unknowns Alert */}
          {draft.unknowns && draft.unknowns.length > 0 && (
            <div className="p-3 rounded-xl bg-amber-950/30 border border-amber-500/30 text-amber-200 text-xs flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
              <div>
                <strong className="text-amber-300 block">Termes hors lexique contrôlé :</strong>
                <p>
                  Les tokens suivants n'ont aucune correspondance dans la bibliothèque de mouvements autorisés :{' '}
                  <span className="font-mono font-bold text-amber-100">
                    {draft.unknowns.join(', ')}
                  </span>
                  . Aucune animation n'a été inventée pour ces termes.
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Animation Plan Breakdown */}
      {plan && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
          <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-slate-800 pb-3 gap-2">
            <div className="flex items-center gap-2">
              <FileCheck2 className="w-4 h-4 text-emerald-400" />
              <h4 className="font-semibold text-slate-200 text-sm">
                Plan d'Animation Généré (ID : <code className="text-sky-300 font-mono">{plan.plan_id}</code>)
              </h4>
            </div>
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5 text-xs font-mono text-slate-400">
                <Clock className="w-3.5 h-3.5 text-slate-500" />
                <span>Durée totale : {plan.duration_ms}ms</span>
              </div>

              {plan.segments.length > 0 && (
                <button
                  onClick={() => onLoadPlanSequence(plan)}
                  className="px-3 py-1 rounded-lg bg-sky-500 hover:bg-sky-400 text-slate-950 font-bold text-xs flex items-center gap-1.5 shadow-sm transition-all"
                >
                  <Film className="w-3.5 h-3.5" />
                  Rejouer la séquence complète
                </button>
              )}
            </div>
          </div>

          {/* Segments Timeline list */}
          <div className="space-y-2">
            <span className="text-xs text-slate-400 font-medium block">
              Segments temporels ({plan.segments.length}) :
            </span>
            {plan.segments.length === 0 ? (
              <div className="p-3 bg-slate-950 rounded-xl border border-slate-800 text-xs text-amber-300 italic">
                Aucun segment de mouvement disponible : tous les termes de la phrase sont inconnus ou non enregistrés.
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-2">
                {plan.segments.map((seg, idx) => (
                  <div
                    key={`${seg.motion_id}-${idx}`}
                    className="flex items-center justify-between p-3 rounded-xl bg-slate-950 border border-slate-800 hover:border-slate-700 transition-colors"
                  >
                    <div className="flex items-center gap-3">
                      <span className="w-6 h-6 rounded-full bg-slate-800 text-slate-300 text-xs flex items-center justify-center font-bold font-mono">
                        {idx + 1}
                      </span>
                      <div>
                        <div className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                          <span>Gloss ID : {seg.gloss_id}</span>
                          <span className="text-slate-500 font-normal">|</span>
                          <span className="font-mono text-sky-400">{seg.motion_id}</span>
                        </div>
                        <div className="text-[11px] text-slate-400 font-mono">
                          Début : {seg.start_ms}ms — Durée : {seg.duration_ms}ms
                        </div>
                      </div>
                    </div>

                    <button
                      onClick={() => onLoadSingleSegment(seg.motion_id, seg.asset_url, seg.gloss_id)}
                      className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-sky-300 hover:text-white border border-slate-700 text-xs font-medium flex items-center gap-1.5 transition-colors"
                      title="Prévisualiser ce clip isolé"
                    >
                      <PlayCircle className="w-3.5 h-3.5" />
                      Clip isolé
                    </button>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Publication Gating notice */}
          <div className="p-3 rounded-xl bg-slate-950 border border-slate-800/80 text-xs text-slate-400 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <HelpCircle className="w-4 h-4 text-slate-500" />
              <span>
                Export pour publication :{' '}
                <strong className={plan.render_allowed_for_publication ? 'text-emerald-400' : 'text-rose-400'}>
                  {plan.render_allowed_for_publication ? 'Autorisé' : 'Bloqué (validation requise)'}
                </strong>
              </span>
            </div>
            <span className="text-[11px] text-slate-500 italic">
              Voir l'onglet "Revue & Export" pour le flux de validation.
            </span>
          </div>
        </div>
      )}
    </div>
  );
};
