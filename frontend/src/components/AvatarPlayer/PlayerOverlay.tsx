import React from 'react';
import { AlertCircle, Film, Sparkles, User } from 'lucide-react';
import { Badge } from '../Common/Badge';

interface PlayerOverlayProps {
  motionId?: string;
  rigVersion?: string;
  fps?: number;
  durationMs?: number;
  activeGloss?: string;
  isSynthetic?: boolean;
}

export const PlayerOverlay: React.FC<PlayerOverlayProps> = ({
  motionId,
  rigVersion = 'signaculture-v1',
  fps = 10,
  durationMs = 1000,
  activeGloss,
  isSynthetic = true,
}) => {
  return (
    <div className="pointer-events-none absolute inset-x-0 top-0 p-3 flex flex-wrap items-start justify-between gap-2 z-10">
      {/* Dynamic Synthetic Badge vs Verified Clearance */}
      <div className="flex flex-col gap-1.5 pointer-events-auto">
        {isSynthetic ? (
          <Badge
            variant="purple"
            size="md"
            icon={<AlertCircle className="w-3.5 h-3.5 text-purple-400" />}
          >
            Fixture synthétique technique — Non validée LSF
          </Badge>
        ) : (
          <Badge
            variant="success"
            size="md"
            icon={<User className="w-3.5 h-3.5 text-emerald-400" />}
          >
            Mouvement réel enregistré
          </Badge>
        )}

        <div className="bg-slate-900/90 border border-slate-700/60 rounded-lg px-2.5 py-1 text-[10px] text-amber-300 shadow backdrop-blur max-w-xs">
          Corps & bras uniquement (sans articulation digitale). Fixture technique préliminaire.
        </div>
      </div>

      {/* Active Motion & Gloss Info */}
      <div className="flex flex-col items-end gap-1.5 ml-auto pointer-events-auto">
        {activeGloss && (
          <Badge variant="info" size="md" icon={<Sparkles className="w-3.5 h-3.5" />}>
            Segment actif : <span className="font-bold ml-1">{activeGloss}</span>
          </Badge>
        )}

        {motionId && (
          <div className="bg-slate-900/80 border border-slate-700/60 rounded-full px-2.5 py-0.5 text-[11px] font-mono text-slate-300 flex items-center gap-1.5 shadow backdrop-blur">
            <Film className="w-3 h-3 text-slate-400" />
            <span>{motionId}</span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400">{rigVersion}</span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400">{fps} fps</span>
            <span className="text-slate-500">|</span>
            <span className="text-slate-400">{durationMs}ms</span>
          </div>
        )}
      </div>
    </div>
  );
};
