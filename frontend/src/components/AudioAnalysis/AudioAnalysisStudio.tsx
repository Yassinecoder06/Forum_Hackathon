import React, { useState, useRef } from 'react';
import {
  UploadCloud,
  FileAudio,
  Activity,
  AlertCircle,
  Volume2,
  Clock,
  Sparkles,
  Sliders,
  CheckCircle2,
} from 'lucide-react';
import { apiClient } from '../../api/client';
import { AudioFeatures } from '../../types/contract';
import { Badge } from '../Common/Badge';
import { ApiErrorAlert } from '../Common/ApiErrorAlert';

interface AudioAnalysisStudioProps {
  onAudioFeaturesLoaded?: (features: AudioFeatures) => void;
  onExpressiveToggle?: (enabled: boolean, energy: number) => void;
}

export const AudioAnalysisStudio: React.FC<AudioAnalysisStudioProps> = ({
  onAudioFeaturesLoaded,
  onExpressiveToggle,
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [audioUrl, setAudioUrl] = useState<string | null>(null);
  const [audioFeatures, setAudioFeatures] = useState<AudioFeatures | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<Error | null>(null);
  const [currentTimeMs, setCurrentTimeMs] = useState(0);

  // Expressive comparison states
  const [isExpressiveMode, setIsExpressiveMode] = useState(false);
  const [expressiveEnergy, setExpressiveEnergy] = useState(0.5);

  const audioRef = useRef<HTMLAudioElement | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setAudioUrl(URL.createObjectURL(file));
      handleUpload(file);
    }
  };

  const handleUpload = async (fileToUpload?: File) => {
    const file = fileToUpload || selectedFile;
    if (!file) return;

    setLoading(true);
    setError(null);

    try {
      const features = await apiClient.analyzeAudio(file);
      setAudioFeatures(features);
      if (onAudioFeaturesLoaded) {
        onAudioFeaturesLoaded(features);
      }
    } catch (err: unknown) {
      setError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setLoading(false);
    }
  };

  const handleAudioTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTimeMs(audioRef.current.currentTime * 1000);
    }
  };

  const handleExpressiveChange = (enabled: boolean, energy?: number) => {
    setIsExpressiveMode(enabled);
    const eng = energy ?? expressiveEnergy;
    if (energy !== undefined) {
      setExpressiveEnergy(eng);
    }
    if (onExpressiveToggle) {
      onExpressiveToggle(enabled, eng);
    }
  };

  // Pre-calculate SVG points for curves
  const renderCurvesSvg = () => {
    if (!audioFeatures || audioFeatures.frames.length === 0) return null;

    const frames = audioFeatures.frames;
    const totalMs = audioFeatures.duration_ms || frames[frames.length - 1].end_ms;
    const width = 800;
    const height = 180;

    // Max pitch for scaling (e.g. 400Hz or max detected)
    const pitches = frames.map((f) => f.pitch_hz).filter((p): p is number => p !== null && p > 0);
    const maxPitch = pitches.length > 0 ? Math.max(...pitches, 300) : 350;

    // Max energy
    const energies = frames.map((f) => f.energy);
    const maxEnergy = Math.max(...energies, 0.001);

    // Build pitch points
    const pitchPoints: string[] = [];
    // Build energy points
    const energyPoints: string[] = [`0,${height}`];

    frames.forEach((frame) => {
      const xMid = ((frame.start_ms + frame.end_ms) / 2 / totalMs) * width;
      const xStart = (frame.start_ms / totalMs) * width;
      const xEnd = (frame.end_ms / totalMs) * width;

      // Energy curve (lower half)
      const normalizedEnergy = frame.energy / maxEnergy;
      const yEnergy = height - normalizedEnergy * (height * 0.45);
      energyPoints.push(`${xStart},${yEnergy}`);
      energyPoints.push(`${xEnd},${yEnergy}`);

      // Pitch curve (upper half)
      if (frame.pitch_hz !== null && !frame.is_pause) {
        const normalizedPitch = Math.min(1, frame.pitch_hz / maxPitch);
        const yPitch = height * 0.45 - normalizedPitch * (height * 0.4);
        pitchPoints.push(`${xMid.toFixed(1)},${yPitch.toFixed(1)}`);
      }
    });

    energyPoints.push(`${width},${height}`);

    const cursorX = (currentTimeMs / totalMs) * width;

    return (
      <div className="relative bg-slate-950 border border-slate-800 rounded-xl p-4 overflow-hidden shadow-inner">
        <svg
          viewBox={`0 0 ${width} ${height}`}
          className="w-full h-44 select-none"
          preserveAspectRatio="none"
        >
          {/* Pause regions (silences) shaded */}
          {frames
            .filter((f) => f.is_pause)
            .map((f, i) => {
              const x = (f.start_ms / totalMs) * width;
              const w = ((f.end_ms - f.start_ms) / totalMs) * width;
              return (
                <rect
                  key={i}
                  x={x}
                  y={0}
                  width={Math.max(w, 2)}
                  height={height}
                  fill="#1e293b"
                  opacity={0.65}
                />
              );
            })}

          {/* Grid lines */}
          <line x1={0} y1={height * 0.48} x2={width} y2={height * 0.48} stroke="#334155" strokeDasharray="3 3" />

          {/* Energy Filled Polygon */}
          <polygon
            points={energyPoints.join(' ')}
            fill="url(#energyGradient)"
            opacity={0.7}
          />

          {/* Pitch Path */}
          {pitchPoints.length > 1 && (
            <path
              d={`M ${pitchPoints.join(' L ')}`}
              fill="none"
              stroke="#38bdf8"
              strokeWidth="2.5"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          )}

          {/* Gradients */}
          <defs>
            <linearGradient id="energyGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#818cf8" stopOpacity="0.8" />
              <stop offset="100%" stopColor="#4f46e5" stopOpacity="0.1" />
            </linearGradient>
          </defs>

          {/* Real-time playback cursor */}
          <line
            x1={cursorX}
            y1={0}
            x2={cursorX}
            y2={height}
            stroke="#f43f5e"
            strokeWidth="2"
          />
        </svg>

        {/* Legend */}
        <div className="flex flex-wrap items-center justify-between text-[11px] text-slate-400 mt-2 pt-2 border-t border-slate-800">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-sky-400 inline-block" />
              Pitch F0 (Hauteur tonale Hz)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 rounded-full bg-indigo-500 inline-block" />
              Énergie RMS (Intensité)
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 bg-slate-700 inline-block rounded-sm" />
              Pauses acoustiques
            </span>
          </div>

          <span className="font-mono text-slate-400">
            Cursor : {Math.round(currentTimeMs)}ms / {totalMs}ms
          </span>
        </div>
      </div>
    );
  };

  return (
    <div className="space-y-6">
      {/* Upload Zone */}
      <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="p-2 bg-indigo-500/10 border border-indigo-500/20 rounded-xl text-indigo-400">
              <FileAudio className="w-5 h-5" />
            </div>
            <div>
              <h3 className="font-bold text-slate-100 text-base">
                Analyse Audio & Prosodie Vocale (WAV)
              </h3>
              <p className="text-xs text-slate-400">
                Extraction du signal réel (F0, énergie, pauses). Aucune simulation non vérifiée.
              </p>
            </div>
          </div>

          <Badge variant="info" size="sm">
            Expérimental (P1)
          </Badge>
        </div>

        {/* File Drag-and-drop / Input */}
        <div className="border-2 border-dashed border-slate-700/80 hover:border-indigo-500/60 rounded-xl p-6 text-center transition-colors bg-slate-950/40">
          <input
            type="file"
            id="audio-upload"
            accept=".wav,audio/wav"
            onChange={handleFileChange}
            className="hidden"
          />
          <label htmlFor="audio-upload" className="cursor-pointer flex flex-col items-center gap-2">
            <UploadCloud className="w-8 h-8 text-indigo-400 animate-pulse" />
            <span className="text-sm font-medium text-slate-200">
              {selectedFile ? selectedFile.name : 'Cliquez pour sélectionner un fichier audio WAV'}
            </span>
            <span className="text-xs text-slate-500">
              Format WAV standard recommandé. Le backend effectuera un traitement déterministe.
            </span>
          </label>
        </div>

        {/* Audio Player if URL available */}
        {audioUrl && (
          <div className="bg-slate-950 p-3 rounded-xl border border-slate-800 flex items-center gap-3">
            <Volume2 className="w-5 h-5 text-indigo-400 shrink-0" />
            <audio
              ref={audioRef}
              src={audioUrl}
              controls
              onTimeUpdate={handleAudioTimeUpdate}
              className="w-full h-8"
            />
          </div>
        )}
      </div>

      {/* Error alert */}
      {error && (
        <ApiErrorAlert
          error={error}
          onRetry={() => handleUpload()}
          title="Erreur lors de l'appel à /api/analyze-audio"
        />
      )}

      {/* Loading state */}
      {loading && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-8 text-center space-y-3">
          <div className="w-8 h-8 border-3 border-indigo-400 border-t-transparent rounded-full animate-spin mx-auto" />
          <p className="text-sm text-slate-300 font-medium">
            Calcul de la prosodie et segmentation temporelle en cours sur le backend...
          </p>
        </div>
      )}

      {/* Real Curves & Analysis Card */}
      {audioFeatures && !loading && (
        <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-5 shadow-xl space-y-5">
          <div className="flex items-center justify-between border-b border-slate-800 pb-3">
            <div className="flex items-center gap-2">
              <Activity className="w-4 h-4 text-sky-400" />
              <h4 className="font-semibold text-slate-200 text-sm">
                Courbes Temporelles Réelles du Signal Audio
              </h4>
            </div>
            <div className="flex items-center gap-2 text-xs font-mono text-slate-400">
              <Clock className="w-3.5 h-3.5 text-slate-500" />
              <span>Durée : {audioFeatures.duration_ms}ms</span>
            </div>
          </div>

          {/* Render Visual Curves SVG */}
          {renderCurvesSvg()}

          {/* Model info & Emotion status card */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <span className="font-semibold text-slate-300 block">
                Hypothèse d'Émotion Vocale :
              </span>
              {audioFeatures.emotion_hypothesis ? (
                <div className="text-slate-200 font-mono">
                  {JSON.stringify(audioFeatures.emotion_hypothesis)}
                </div>
              ) : (
                <div className="flex items-start gap-2 text-amber-300/90 bg-amber-950/30 p-2.5 rounded-lg border border-amber-500/20">
                  <AlertCircle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5" />
                  <div>
                    <span className="font-bold block">Non disponible (null) :</span>
                    Conformément au contrat d'intégration, aucun modèle d'émotion lourd non vérifié n'est exécuté sans configuration locale validée.
                  </div>
                </div>
              )}
            </div>

            <div className="bg-slate-950 p-4 rounded-xl border border-slate-800 space-y-2">
              <span className="font-semibold text-slate-300 block">
                Métadonnées & Avertissements :
              </span>
              <p className="text-slate-400">
                Version du modèle d'analyse :{' '}
                <span className="font-mono text-slate-200">{audioFeatures.model_version}</span>
              </p>
              {audioFeatures.warnings && audioFeatures.warnings.length > 0 ? (
                <div className="space-y-1 pt-1">
                  {audioFeatures.warnings.map((w, i) => (
                    <div key={i} className="text-amber-300 text-[11px] flex items-center gap-1.5">
                      <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
                      {w}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-emerald-400 text-[11px] flex items-center gap-1 pt-1">
                  <CheckCircle2 className="w-3.5 h-3.5" />
                  Aucun avertissement de traitement audio.
                </p>
              )}
            </div>
          </div>

          {/* Neutral vs Expressive Comparison Module */}
          <div className="p-4 rounded-xl bg-gradient-to-br from-indigo-950/40 to-slate-950 border border-indigo-500/30 space-y-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Sliders className="w-4 h-4 text-indigo-400" />
                <h5 className="font-bold text-slate-200 text-sm">
                  Comparaison Expérimentale : Neutre vs Expressif Contraint
                </h5>
              </div>

              <div className="flex items-center gap-2">
                <button
                  onClick={() => handleExpressiveChange(false)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
                    !isExpressiveMode
                      ? 'bg-slate-800 text-slate-100 border border-slate-600'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Mode Neutre (Base)
                </button>
                <button
                  onClick={() => handleExpressiveChange(true)}
                  className={`px-3 py-1 rounded-lg text-xs font-semibold transition-colors ${
                    isExpressiveMode
                      ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/30'
                      : 'text-slate-400 hover:text-slate-200'
                  }`}
                >
                  Mode Expressif
                </button>
              </div>
            </div>

            {/* Energy Amplitude Slider */}
            {isExpressiveMode && (
              <div className="space-y-2 pt-2 border-t border-indigo-500/20">
                <div className="flex items-center justify-between text-xs">
                  <span className="text-indigo-200 font-medium">
                    Amplitude gestuelle manuelle/acoustique (expérimentale, sans garantie) :
                  </span>
                  <span className="font-mono font-bold text-indigo-300">
                    {(expressiveEnergy * 100).toFixed(0)}%
                  </span>
                </div>
                <input
                  type="range"
                  min="0"
                  max="1"
                  step="0.05"
                  value={expressiveEnergy}
                  onChange={(e) => handleExpressiveChange(true, Number(e.target.value))}
                  className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-indigo-400"
                />
              </div>
            )}

            {/* Strict Ethical Disclaimer */}
            <div className="text-[11px] text-amber-200/90 bg-amber-950/30 p-3 rounded-lg border border-amber-500/20 leading-relaxed flex items-start gap-2">
              <Sparkles className="w-3.5 h-3.5 text-amber-400 shrink-0 mt-0.5" />
              <p>
                <strong>Garde-fou éthique :</strong> Modulation expérimentale de l'amplitude gestuelle uniquement (variation manuelle/acoustique sans garantie linguistique). La modulation de tempo et la coarticulation temporelle constituent des travaux futurs. Aucune préservation automatique de la syntaxe spatiale LSF n'est certifiée sur cette couche.
              </p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
