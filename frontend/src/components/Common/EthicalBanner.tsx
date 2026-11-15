import React, { useState } from 'react';
import { ShieldAlert, Info, ChevronDown, ChevronUp } from 'lucide-react';

export const EthicalBanner: React.FC = () => {
  const [expanded, setExpanded] = useState(false);

  return (
    <aside
      aria-label="Avertissement méthodologique et éthique"
      className="border-b border-amber-500/30 bg-amber-950/30 text-amber-100 text-xs px-4 py-2.5 transition-all shadow-sm"
    >
      <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-start md:items-center justify-between gap-2">
        <div className="flex items-center gap-2 text-amber-200">
          <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0" />
          <p className="font-medium">
            <span className="font-bold uppercase tracking-wide text-amber-300 mr-1.5">
              Prototype Hackathon Éthique 360 :
            </span>
            La lemmatisation n'est pas une traduction LSF. Les mouvements synthétiques sont des fixtures techniques non validées.
          </p>
        </div>

        <button
          onClick={() => setExpanded(!expanded)}
          className="text-amber-300 hover:text-amber-200 underline flex items-center gap-1 font-medium shrink-0 ml-auto md:ml-0"
          aria-expanded={expanded}
        >
          <Info className="w-3.5 h-3.5" />
          {expanded ? 'Masquer les engagements' : 'Détails des 6 piliers éthiques'}
          {expanded ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
        </button>
      </div>

      {expanded && (
        <div className="max-w-7xl mx-auto mt-3 pt-3 border-t border-amber-500/20 grid grid-cols-1 md:grid-cols-3 gap-3 text-amber-200/90 leading-relaxed text-xs">
          <div className="p-2.5 rounded bg-amber-950/40 border border-amber-500/20">
            <strong className="text-amber-300 block mb-1">1. Transparence & Vérité</strong>
            Aucun faux score ou vidéo simulée. Les prédictions d'émotion sont désactivées sans modèle vérifié en local.
          </div>
          <div className="p-2.5 rounded bg-amber-950/40 border border-amber-500/20">
            <strong className="text-amber-300 block mb-1">2. Primauté Linguistique LSF</strong>
            Les marqueurs grammaticaux (visage, espace, tempo) priment rigoureusement sur toute modulation sonore.
          </div>
          <div className="p-2.5 rounded bg-amber-950/40 border border-amber-500/20">
            <strong className="text-amber-300 block mb-1">3. Gating & Validation Humaine</strong>
            La publication officielle est bloquée tant que des signants LSF compétents n'ont pas validé les séquences.
          </div>
        </div>
      )}
    </aside>
  );
};
