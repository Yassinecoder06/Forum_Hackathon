import React, { useState } from 'react';
import {
  ShieldCheck,
  Lock,
  UserCheck,
  CheckCircle,
  XCircle,
  Clock,
  Eye,
  Download,
  FileText,
  RefreshCw,
  Film,
  Info,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import { AnimationPlan, RenderResponse, ReviewRecord } from '../../types/contract';
import { Badge } from '../Common/Badge';
import { ApiErrorAlert } from '../Common/ApiErrorAlert';

interface HumanReviewStudioProps {
  currentPlan: AnimationPlan | null;
  onPlanUpdated?: (updatedPlan: AnimationPlan) => void;
}

export const HumanReviewStudio: React.FC<HumanReviewStudioProps> = ({
  currentPlan,
  onPlanUpdated,
}) => {
  // Reviewer fields start empty (no fabricated persona or notes; explicit entry required)
  const [reviewerName, setReviewerName] = useState('');
  const [reviewStatus, setReviewStatus] = useState<'approved' | 'rejected'>('approved');
  const [reviewNotes, setReviewNotes] = useState('');
  const [submittingReview, setSubmittingReview] = useState(false);
  const [reviewResult, setReviewResult] = useState<ReviewRecord | null>(null);
  const [reviewError, setReviewError] = useState<Error | null>(null);

  // Render / Export gating state
  const [renderLoading, setRenderLoading] = useState(false);
  const [renderResponse, setRenderResponse] = useState<RenderResponse | null>(null);
  const [renderError, setRenderError] = useState<Error | null>(null);

  // If no plan has been composed yet, disable review and render
  if (!currentPlan) {
    return (
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-8 shadow-xl text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center mx-auto">
          <Info className="w-6 h-6" />
        </div>
        <div>
          <h3 className="font-bold text-slate-100 text-base">Aucun plan d'animation actif</h3>
          <p className="text-xs text-slate-400 mt-1 max-w-md mx-auto leading-relaxed">
            Pour évaluer et procéder à la revue humaine d'une animation LSF, veuillez d'abord composer une séquence depuis l'onglet{' '}
            <strong className="text-sky-300">"Studio Texte-à-Signe"</strong>.
          </p>
        </div>
      </div>
    );
  }

  const planId = currentPlan.plan_id;

  // Submit review and refresh authoritative plan from /render preview
  const handleSubmitReview = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!reviewerName.trim() || !reviewNotes.trim()) return;

    setSubmittingReview(true);
    setReviewError(null);

    try {
      // 1. Submit review to backend SQLite
      const record = await apiClient.submitReview({
        plan_id: planId,
        reviewer: reviewerName.trim(),
        status: reviewStatus,
        notes: reviewNotes.trim(),
      });
      setReviewResult(record);

      // 2. Query /render preview to fetch authoritative verified plan and assets
      const renderRes = await apiClient.requestRender({
        plan_id: planId,
        purpose: 'preview',
      });
      setRenderResponse(renderRes);

      // Authoritative update from backend response
      if (renderRes.plan && onPlanUpdated) {
        onPlanUpdated(renderRes.plan);
      }
    } catch (err: unknown) {
      setReviewError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setSubmittingReview(false);
    }
  };

  const handleRequestRender = async (purpose: 'preview' | 'publication') => {
    setRenderLoading(true);
    setRenderError(null);
    setRenderResponse(null);

    try {
      const res = await apiClient.requestRender({
        plan_id: planId,
        purpose,
      });
      setRenderResponse(res);

      // Authoritative update from backend response
      if (res.plan && onPlanUpdated) {
        onPlanUpdated(res.plan);
      }
    } catch (err: unknown) {
      setRenderError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setRenderLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Overview & Checklist */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-emerald-500/10 border border-emerald-500/20 rounded-xl text-emerald-400">
              <ShieldCheck className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-base">
                Revue Humaine & Gating de Publication Éthique
              </h3>
              <p className="text-xs text-slate-400">
                Triple validation impérative : linguistique, animation et droits d'auteur (P0).
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => handleRequestRender('preview')}
              disabled={renderLoading}
              title="Rafraîchir le plan d'animation depuis /render preview"
              className="px-2.5 py-1 text-xs rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 border border-slate-700 flex items-center gap-1 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={`w-3 h-3 ${renderLoading ? 'animate-spin' : ''}`} />
              Rafraîchir statut
            </button>
            <Badge variant="purple" size="sm">
              Plan ID : {planId}
            </Badge>
          </div>
        </div>

        {/* Triple Gate Status Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          {/* 1. Linguistic Review */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <span className="text-xs text-slate-400 font-medium block">
              1. Validation Linguistique LSF
            </span>
            <div className="flex items-center gap-2">
              {currentPlan.linguistic_review === 'approved' ? (
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              ) : currentPlan.linguistic_review === 'rejected' ? (
                <XCircle className="w-4 h-4 text-rose-400" />
              ) : (
                <Clock className="w-4 h-4 text-amber-400" />
              )}
              <span className="text-sm font-semibold text-slate-200">
                {currentPlan.linguistic_review === 'approved'
                  ? 'Validé par signant'
                  : currentPlan.linguistic_review === 'rejected'
                  ? 'Rejeté'
                  : 'En attente (pending)'}
              </span>
            </div>
            <p className="text-[11px] text-slate-500">
              Conformité grammaticale et syntaxe spatiale.
            </p>
          </div>

          {/* 2. Animation Quality Review */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <span className="text-xs text-slate-400 font-medium block">
              2. Fidélité d'Animation
            </span>
            <div className="flex items-center gap-2">
              {currentPlan.animation_review === 'approved' ? (
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              ) : (
                <Clock className="w-4 h-4 text-amber-400" />
              )}
              <span className="text-sm font-semibold text-slate-200">
                {currentPlan.animation_review === 'approved'
                  ? 'Validé'
                  : 'En attente (pending)'}
              </span>
            </div>
            <p className="text-[11px] text-slate-500">
              Absence de collisions, continuité et lisibilité.
            </p>
          </div>

          {/* 3. Rights & Licenses */}
          <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
            <span className="text-xs text-slate-400 font-medium block">
              3. Droits & Consentements
            </span>
            <div className="flex items-center gap-2">
              {currentPlan.rights_status === 'cleared' ? (
                <CheckCircle className="w-4 h-4 text-emerald-400" />
              ) : (
                <Clock className="w-4 h-4 text-amber-400" />
              )}
              <span className="text-sm font-semibold text-slate-200">
                {currentPlan.rights_status === 'cleared'
                  ? 'Droits vérifiés'
                  : 'En attente de libération'}
              </span>
            </div>
            <p className="text-[11px] text-slate-500">
              Licence CC / accord de performance vérifié.
            </p>
          </div>
        </div>
      </div>

      {/* Review Submission Form with empty initial inputs */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <h4 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
          <UserCheck className="w-4 h-4 text-sky-400" />
          Formulaire de Revue Humaine & Évaluation Qualité
        </h4>

        <form onSubmit={handleSubmitReview} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label htmlFor="reviewer-name-input" className="block text-xs font-medium text-slate-300 mb-1.5">
                Nom ou identifiant de l'évaluateur (locuteur / signant LSF) * :
              </label>
              <input
                id="reviewer-name-input"
                type="text"
                value={reviewerName}
                onChange={(e) => setReviewerName(e.target.value)}
                required
                placeholder="Ex : Camille (Signante LSF certifiée)"
                className="w-full bg-slate-950 border border-slate-700/80 rounded-xl px-3.5 py-2.5 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/50"
              />
            </div>

            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Décision de validation :
              </label>
              <div className="flex items-center gap-3">
                <button
                  type="button"
                  onClick={() => setReviewStatus('approved')}
                  className={`flex-1 py-2.5 px-3 rounded-xl border text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                    reviewStatus === 'approved'
                      ? 'bg-emerald-950 border-emerald-500 text-emerald-300 shadow-md shadow-emerald-950/50'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <CheckCircle className="w-3.5 h-3.5" />
                  Approuver (Validé)
                </button>
                <button
                  type="button"
                  onClick={() => setReviewStatus('rejected')}
                  className={`flex-1 py-2.5 px-3 rounded-xl border text-xs font-semibold flex items-center justify-center gap-1.5 transition-all ${
                    reviewStatus === 'rejected'
                      ? 'bg-rose-950 border-rose-500 text-rose-300 shadow-md shadow-rose-950/50'
                      : 'bg-slate-950 border-slate-800 text-slate-400 hover:text-slate-200'
                  }`}
                >
                  <XCircle className="w-3.5 h-3.5" />
                  Rejeter
                </button>
              </div>
            </div>
          </div>

          <div>
            <label htmlFor="reviewer-notes-input" className="block text-xs font-medium text-slate-300 mb-1.5">
              Remarques linguistiques, éthiques et observations de révision * :
            </label>
            <textarea
              id="reviewer-notes-input"
              rows={3}
              value={reviewNotes}
              onChange={(e) => setReviewNotes(e.target.value)}
              required
              className="w-full bg-slate-950 border border-slate-700/80 rounded-xl p-3 text-sm text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-sky-500/50"
              placeholder="Indiquez vos observations sur la configuration gestuelle, le repère spatial et le respect des marqueurs non manuels..."
            />
          </div>

          <button
            type="submit"
            disabled={submittingReview || !reviewerName.trim() || !reviewNotes.trim()}
            className="px-5 py-2.5 rounded-xl bg-sky-500 hover:bg-sky-400 active:bg-sky-600 text-slate-950 font-bold text-xs flex items-center gap-2 shadow-md shadow-sky-500/20 transition-all disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {submittingReview ? 'Enregistrement en cours...' : 'Enregistrer la décision de revue (POST /review)'}
          </button>
        </form>

        {reviewError && (
          <ApiErrorAlert
            error={reviewError}
            title="Erreur lors de l'enregistrement de la revue"
          />
        )}

        {reviewResult && (
          <div className="p-3 bg-emerald-950/40 border border-emerald-500/30 rounded-xl text-emerald-200 text-xs flex items-center gap-2">
            <CheckCircle className="w-4 h-4 text-emerald-400 shrink-0" />
            <span>
              Revue enregistrée pour le plan {reviewResult.plan_id} (Réviseur : {reviewResult.reviewer}).
            </span>
          </div>
        )}
      </div>

      {/* Preview & Publication Gating Action Panel */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <h4 className="font-semibold text-slate-200 text-sm flex items-center gap-2">
          <FileText className="w-4 h-4 text-indigo-400" />
          Rendu d'Aperçu & Export de Publication
        </h4>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {/* 1. Preview button */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <h5 className="font-semibold text-slate-200 text-xs flex items-center gap-2">
              <Eye className="w-4 h-4 text-sky-400" />
              Aperçu Local Studio (Preview)
            </h5>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Vérifie le plan complet et les assets enregistrés autorisés pour l'aperçu local.
            </p>
            <button
              onClick={() => handleRequestRender('preview')}
              disabled={renderLoading}
              className="w-full mt-2 py-2 px-3 rounded-lg bg-slate-800 hover:bg-slate-700 text-sky-300 hover:text-white border border-slate-700 text-xs font-semibold flex items-center justify-center gap-2 transition-colors disabled:opacity-50"
            >
              <Eye className="w-3.5 h-3.5" />
              Générer / Vérifier l'Aperçu Local
            </button>
          </div>

          {/* 2. Publication button (Gated) */}
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <h5 className="font-semibold text-slate-200 text-xs flex items-center gap-2">
                <Download className="w-4 h-4 text-emerald-400" />
                Publication Officielle
              </h5>
              <Badge variant={currentPlan.render_allowed_for_publication ? 'success' : 'danger'} size="sm">
                {currentPlan.render_allowed_for_publication ? 'Débloqué' : 'Verrouillé'}
              </Badge>
            </div>
            <p className="text-[11px] text-slate-400 leading-relaxed">
              Bloqué tant que toutes les validations (linguistique, animation, droits) ne sont pas satisfaites.
            </p>
            <button
              onClick={() => handleRequestRender('publication')}
              disabled={renderLoading}
              className={`w-full mt-2 py-2 px-3 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 transition-colors disabled:opacity-50 ${
                currentPlan.render_allowed_for_publication
                  ? 'bg-emerald-600 hover:bg-emerald-500 text-white'
                  : 'bg-rose-950/40 hover:bg-rose-900/60 border border-rose-700/60 text-rose-300'
              }`}
            >
              <Lock className="w-3.5 h-3.5" />
              Demander l'Export de Publication
            </button>
          </div>
        </div>

        {/* Authoritative Render Response Details */}
        {renderResponse && (
          <div className="p-4 rounded-xl bg-slate-950 border border-slate-800 space-y-3 text-xs">
            <div className="flex items-center justify-between border-b border-slate-800 pb-2">
              <div className="flex items-center gap-2 font-bold text-slate-200">
                <Film className="w-4 h-4 text-sky-400" />
                <span>Assets Vérifiés par le Backend ({renderResponse.assets?.length || 0}) :</span>
              </div>
              <Badge variant={renderResponse.video_exported ? 'success' : 'warning'} size="sm">
                video_exported : {renderResponse.video_exported ? 'true' : 'false'}
              </Badge>
            </div>

            {/* Notice regarding video_exported false */}
            {!renderResponse.video_exported && (
              <p className="text-[11px] text-amber-300/90 italic">
                Aperçu interactif 3D validé. Aucun fichier vidéo synthétique fictif n'est revendiqué (video_exported: false conformément au contrat d'intégration).
              </p>
            )}

            {/* Assets List */}
            {renderResponse.assets && renderResponse.assets.length > 0 && (
              <div className="space-y-1.5 pt-1">
                {renderResponse.assets.map((asset) => (
                  <div
                    key={asset.motion_id}
                    className="flex items-center justify-between p-2 rounded-lg bg-slate-900 border border-slate-800 text-[11px] font-mono"
                  >
                    <span className="text-sky-300">{asset.motion_id}</span>
                    <span className="text-slate-400">Droits : {asset.rights_status}</span>
                    <span className="text-slate-400">Revue anim : {asset.animation_review}</span>
                    <Badge variant={asset.synthetic ? 'purple' : 'success'} size="sm">
                      {asset.synthetic ? 'synthétique' : 'réel'}
                    </Badge>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {renderError && (
          <ApiErrorAlert
            error={renderError}
            title="Erreur lors de la requête de rendu /api/render"
          />
        )}
      </div>
    </div>
  );
};
