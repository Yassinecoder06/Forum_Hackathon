import React from 'react';
import {
  Play,
  Pause,
  RotateCcw,
  Repeat,
  Gauge,
  Activity,
} from 'lucide-react';
import { formatTimeMs } from '../../utils/formatters';

interface TimelineControlsProps {
  isPlaying: boolean;
  currentTimeMs: number;
  durationMs: number;
  speed: number;
  loop: boolean;
  showSkeletonDebug: boolean;
  onPlayToggle: () => void;
  onSeek: (ms: number) => void;
  onReset: () => void;
  onSpeedChange: (speed: number) => void;
  onLoopToggle: () => void;
  onSkeletonToggle: () => void;
}

const SPEED_OPTIONS = [0.5, 0.75, 1.0, 1.25, 1.5];

export const TimelineControls: React.FC<TimelineControlsProps> = ({
  isPlaying,
  currentTimeMs,
  durationMs,
  speed,
  loop,
  showSkeletonDebug,
  onPlayToggle,
  onSeek,
  onReset,
  onSpeedChange,
  onLoopToggle,
  onSkeletonToggle,
}) => {
  const safeDuration = Math.max(durationMs, 100);
  const progressPercent = Math.min(100, Math.max(0, (currentTimeMs / safeDuration) * 100));

  return (
    <div className="bg-slate-900/90 border border-slate-800 rounded-xl p-4 shadow-xl backdrop-blur space-y-3">
      {/* Scrubber Slider & Progress Track */}
      <div className="space-y-1.5">
        <div className="flex items-center justify-between text-xs font-mono text-slate-400">
          <span className="text-sky-400 font-semibold">{formatTimeMs(currentTimeMs)}</span>
          <span className="text-slate-500">{formatTimeMs(durationMs)}</span>
        </div>

        <div className="relative flex items-center group">
          <input
            type="range"
            min={0}
            max={safeDuration}
            step={10}
            value={currentTimeMs}
            onChange={(e) => onSeek(Number(e.target.value))}
            aria-label="Position temporelle de l'animation"
            className="w-full h-2 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-sky-400 focus:outline-none focus:ring-2 focus:ring-sky-500/50"
          />
          {/* Visual gradient fill behind slider */}
          <div
            className="absolute left-0 top-0 h-2 bg-gradient-to-r from-sky-500 to-indigo-500 rounded-lg pointer-events-none transition-all"
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Main Buttons Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-3 pt-1">
        {/* Playback Transport Buttons */}
        <div className="flex items-center gap-2">
          <button
            onClick={onReset}
            title="Revenir au début (0ms)"
            className="p-2 rounded-lg bg-slate-800 hover:bg-slate-700 text-slate-300 hover:text-white transition-colors border border-slate-700/50"
            aria-label="Revenir au début"
          >
            <RotateCcw className="w-4 h-4" />
          </button>

          <button
            onClick={onPlayToggle}
            title={isPlaying ? 'Pause' : 'Lecture'}
            className="px-4 py-2 rounded-lg bg-sky-500 hover:bg-sky-400 active:bg-sky-600 text-slate-950 font-semibold flex items-center gap-2 transition-all shadow-md shadow-sky-500/20"
            aria-label={isPlaying ? 'Mettre en pause' : 'Lancer la lecture'}
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current" />}
            <span className="text-xs">{isPlaying ? 'Pause' : 'Lecture'}</span>
          </button>

          <button
            onClick={onLoopToggle}
            title={loop ? 'Boucle active' : 'Boucle désactivée'}
            className={`p-2 rounded-lg border transition-colors ${
              loop
                ? 'bg-indigo-600/30 border-indigo-500 text-indigo-300'
                : 'bg-slate-800 border-slate-700/50 text-slate-400 hover:text-slate-200'
            }`}
            aria-label="Activer ou désactiver la boucle"
          >
            <Repeat className="w-4 h-4" />
          </button>
        </div>

        {/* Speed Selector */}
        <div className="flex items-center gap-1.5 bg-slate-800/80 px-2 py-1 rounded-lg border border-slate-700/50 text-xs">
          <Gauge className="w-3.5 h-3.5 text-slate-400 shrink-0" />
          <span className="text-slate-400 text-[11px] mr-1">Vitesse :</span>
          {SPEED_OPTIONS.map((val) => (
            <button
              key={val}
              onClick={() => onSpeedChange(val)}
              className={`px-1.5 py-0.5 rounded text-[11px] font-medium transition-colors ${
                speed === val
                  ? 'bg-sky-500 text-slate-950 font-bold'
                  : 'text-slate-300 hover:text-white hover:bg-slate-700'
              }`}
            >
              {val}x
            </button>
          ))}
        </div>

        {/* Debug Wireframe Toggle */}
        <button
          onClick={onSkeletonToggle}
          title="Afficher les coordonnées et le squelette de debug"
          className={`px-2.5 py-1.5 rounded-lg border text-xs flex items-center gap-1.5 transition-colors ${
            showSkeletonDebug
              ? 'bg-purple-950/60 border-purple-500/70 text-purple-300 font-semibold'
              : 'bg-slate-800 border-slate-700/50 text-slate-400 hover:text-slate-200'
          }`}
          aria-label="Basculer le squelette de débogage"
        >
          <Activity className="w-3.5 h-3.5" />
          <span>Squelette brut</span>
        </button>
      </div>
    </div>
  );
};
