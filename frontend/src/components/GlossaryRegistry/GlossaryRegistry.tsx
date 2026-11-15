import React, { useState, useEffect } from 'react';
import {
  BookOpen,
  Search,
  Filter,
  PlayCircle,
  RefreshCw,
  ShieldAlert,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import { GlossEntry } from '../../types/contract';
import { Badge } from '../Common/Badge';
import { ApiErrorAlert } from '../Common/ApiErrorAlert';

interface GlossaryRegistryProps {
  onSelectGloss: (motionId: string, label: string) => void;
}

export const GlossaryRegistry: React.FC<GlossaryRegistryProps> = ({ onSelectGloss }) => {
  const [entries, setEntries] = useState<GlossEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<Error | null>(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [statusFilter, setStatusFilter] = useState<string>('all');

  const fetchGlossary = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await apiClient.getGlossary();
      setEntries(data.entries || []);
    } catch (err: unknown) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGlossary();
  }, []);

  const filteredEntries = entries.filter((item) => {
    const matchesSearch =
      item.gloss.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.concept_fr.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.gloss_id.toLowerCase().includes(searchTerm.toLowerCase());

    const matchesStatus =
      statusFilter === 'all' || item.review_status === statusFilter;

    return matchesSearch && matchesStatus;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-purple-500/10 border border-purple-500/20 rounded-xl text-purple-400">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-base">
                Registre & Glossaire LSF Contrôlé
              </h3>
              <p className="text-xs text-slate-400">
                Périmètre restreint de concepts culturels traçables avec licences et statuts (P0).
              </p>
            </div>
          </div>

          <button
            onClick={fetchGlossary}
            disabled={loading}
            className="px-3 py-1.5 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white border border-slate-700 text-xs font-medium flex items-center gap-1.5 transition-colors"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} />
            Actualiser
          </button>
        </div>

        {/* Search & Filter Bar */}
        <div className="flex flex-col sm:flex-row gap-3 pt-2">
          <div className="relative flex-1">
            <Search className="w-4 h-4 text-slate-500 absolute left-3.5 top-3" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Rechercher par concept français, gloss ou ID..."
              className="w-full bg-slate-950 border border-slate-700/80 rounded-xl pl-10 pr-4 py-2.5 text-xs text-slate-200 placeholder:text-slate-500 focus:outline-none focus:ring-2 focus:ring-purple-500/50"
            />
          </div>

          <div className="flex items-center gap-2 bg-slate-950 px-3 py-2 rounded-xl border border-slate-700/80">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-transparent text-xs text-slate-200 focus:outline-none cursor-pointer"
            >
              <option value="all" className="bg-slate-900">Tous les statuts</option>
              <option value="pending" className="bg-slate-900">En attente (pending)</option>
              <option value="approved" className="bg-slate-900">Approuvé (approved)</option>
              <option value="rejected" className="bg-slate-900">Rejeté (rejected)</option>
            </select>
          </div>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <ApiErrorAlert
          error={error}
          onRetry={fetchGlossary}
          title="Erreur lors de la récupération du glossaire (/api/glossary)"
        />
      )}

      {/* Loading state */}
      {loading && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-8 text-center space-y-3">
          <div className="w-8 h-8 border-3 border-purple-400 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-sm text-slate-300 font-medium">Chargement du glossaire contrôlé...</p>
        </div>
      )}

      {/* Cards Grid */}
      {!loading && !error && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredEntries.map((entry) => (
            <div
              key={entry.gloss_id}
              className="bg-slate-900/80 border border-slate-800 hover:border-slate-700 rounded-2xl p-4 shadow-lg transition-all flex flex-col justify-between space-y-3"
            >
              <div className="space-y-2">
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <span className="font-mono text-lg font-extrabold text-purple-300 block">
                      [{entry.gloss}]
                    </span>
                    <span className="text-xs font-semibold text-slate-200">
                      Concept : {entry.concept_fr}
                    </span>
                  </div>
                  <Badge variant={entry.review_status === 'approved' ? 'success' : 'warning'} size="sm">
                    {entry.review_status}
                  </Badge>
                </div>

                <div className="text-[11px] text-slate-400 space-y-1 bg-slate-950/60 p-2.5 rounded-xl border border-slate-800/80">
                  <div className="flex justify-between">
                    <span className="text-slate-500">ID :</span>
                    <span className="font-mono text-slate-300">{entry.gloss_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Source :</span>
                    <span className="font-mono text-slate-300">{entry.source_id}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-slate-500">Motion ID :</span>
                    <span className="font-mono text-sky-400 font-semibold">{entry.motion_id || 'N/A'}</span>
                  </div>
                  {entry.variant && (
                    <div className="pt-1 border-t border-slate-800 text-[10px] text-amber-300/90 italic">
                      {entry.variant}
                    </div>
                  )}
                </div>
              </div>

              {/* Action */}
              {entry.motion_id && (
                <button
                  onClick={() => onSelectGloss(entry.motion_id!, entry.gloss)}
                  className="w-full py-2 px-3 rounded-xl bg-purple-950/60 hover:bg-purple-900/80 border border-purple-500/40 text-purple-200 hover:text-white text-xs font-semibold flex items-center justify-center gap-1.5 transition-colors shadow-sm"
                >
                  <PlayCircle className="w-3.5 h-3.5" />
                  Tester ce signe dans le lecteur 3D
                </button>
              )}
            </div>
          ))}

          {filteredEntries.length === 0 && (
            <div className="col-span-full bg-slate-900/60 border border-slate-800 rounded-2xl p-8 text-center text-xs text-slate-400">
              Aucun gloss ne correspond à votre recherche.
            </div>
          )}
        </div>
      )}

      {/* Methodology notice */}
      <div className="p-4 rounded-xl bg-slate-900/60 border border-slate-800 text-xs text-slate-400 flex items-start gap-2.5">
        <ShieldAlert className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
        <p className="leading-relaxed">
          <strong>Gouvernance des données (Team A) :</strong> Ces concepts proviennent d'une bibliothèque restreinte de démonstration. L'appariement d'un mot français vers un gloss ne garantit pas la correction de l'agencement syntaxique LSF dans une phrase complète.
        </p>
      </div>
    </div>
  );
};
