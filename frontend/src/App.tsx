import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Header, TabType } from './components/Header';
import { EthicalBanner } from './components/Common/EthicalBanner';
import { AvatarCanvas } from './components/AvatarPlayer/AvatarCanvas';
import { TimelineControls } from './components/AvatarPlayer/TimelineControls';
import { PlayerOverlay } from './components/AvatarPlayer/PlayerOverlay';
import { TextToSignStudio } from './components/TextToSign/TextToSignStudio';
import { AudioAnalysisStudio } from './components/AudioAnalysis/AudioAnalysisStudio';
import { HumanReviewStudio } from './components/HumanReview/HumanReviewStudio';
import { GlossaryRegistry } from './components/GlossaryRegistry/GlossaryRegistry';
import { ApiErrorAlert } from './components/Common/ApiErrorAlert';
import { apiClient } from './api/client';
import { AnimationPlan, AudioFeatures, MotionFile } from './types/contract';
import { stitchAnimationPlan } from './utils/planStitcher';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<TabType>('studio');

  // Backend Health state
  const [apiConnected, setApiConnected] = useState<boolean | null>(null);
  const [globalApiError, setGlobalApiError] = useState<Error | null>(null);

  // 3D Motion state
  const [currentMotion, setCurrentMotion] = useState<MotionFile | null>(null);
  const [activeMotionId, setActiveMotionId] = useState<string | undefined>(undefined);
  const [activeGlossLabel, setActiveGlossLabel] = useState<string | undefined>(undefined);
  const [isSyntheticAsset, setIsSyntheticAsset] = useState<boolean>(true);
  const [currentPlan, setCurrentPlan] = useState<AnimationPlan | null>(null);

  // Playback state
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [currentTimeMs, setCurrentTimeMs] = useState<number>(0);
  const [playbackSpeed, setPlaybackSpeed] = useState<number>(1.0);
  const [loop, setLoop] = useState<boolean>(true);
  const [showSkeletonDebug, setShowSkeletonDebug] = useState<boolean>(false);

  // Expressive Modulation state (Audio-conditioned)
  const [expressiveModulation, setExpressiveModulation] = useState<boolean>(false);
  const [energyFactor, setEnergyFactor] = useState<number>(0.5);

  const lastAnimTimeRef = useRef<number | null>(null);
  const animFrameIdRef = useRef<number | null>(null);

  // 1. Health check
  const checkHealth = useCallback(async () => {
    setGlobalApiError(null);
    try {
      const res = await apiClient.getHealth();
      setApiConnected(res.status === 'ok');
    } catch (err: unknown) {
      setApiConnected(false);
      setGlobalApiError(err instanceof Error ? err : new Error(String(err)));
    }
  }, []);

  useEffect(() => {
    checkHealth();
  }, [checkHealth]);

  // 2. Clear Motion callback
  const handleClearMotion = useCallback(() => {
    setCurrentMotion(null);
    setActiveMotionId(undefined);
    setActiveGlossLabel(undefined);
    setCurrentTimeMs(0);
    setIsPlaying(false);
  }, []);

  // 3. Multi-clip sequence loader: stitches all segments of an AnimationPlan
  const handleLoadPlanSequence = useCallback(
    async (plan: AnimationPlan) => {
      setGlobalApiError(null);
      if (!plan.segments || plan.segments.length === 0) {
        handleClearMotion();
        return;
      }

      try {
        const motionMap = new Map<string, MotionFile>();
        let hasSynthetic = false;

        // Fetch all segment files and metadata concurrently
        await Promise.all(
          plan.segments.map(async (seg) => {
            const urlToFetch = seg.asset_url || `/assets/${seg.motion_id}.json`;
            const [motionFile, meta] = await Promise.all([
              apiClient.getMotionFile(urlToFetch),
              apiClient.getMotion(seg.motion_id).catch(() => null),
            ]);

            motionMap.set(seg.motion_id, motionFile);
            if (meta?.synthetic || meta?.synthetic === undefined) {
              hasSynthetic = true;
            }
          })
        );

        // Stitch motions across plan start_ms boundaries
        const stitchedMotion = stitchAnimationPlan(plan, motionMap);

        setCurrentMotion(stitchedMotion);
        setActiveMotionId(
          plan.segments.length === 1
            ? plan.segments[0].motion_id
            : `Séquence (${plan.segments.length} clips)`
        );
        setIsSyntheticAsset(hasSynthetic);
        setCurrentTimeMs(0);
        setIsPlaying(true);
      } catch (err: unknown) {
        const error = err instanceof Error ? err : new Error(String(err));
        setGlobalApiError(error);
        handleClearMotion();
      }
    },
    [handleClearMotion]
  );

  // 4. Single segment preview loader
  const handleLoadSingleSegment = useCallback(
    async (motionId: string, assetUrl?: string, label?: string) => {
      setGlobalApiError(null);
      try {
        const urlToFetch = assetUrl || `/assets/${motionId}.json`;
        const [motion, meta] = await Promise.all([
          apiClient.getMotionFile(urlToFetch),
          apiClient.getMotion(motionId).catch(() => null),
        ]);

        setCurrentMotion(motion);
        setActiveMotionId(motionId);
        setActiveGlossLabel(label);
        setIsSyntheticAsset(meta?.synthetic ?? true);
        setCurrentTimeMs(0);
        setIsPlaying(true);
      } catch (err: unknown) {
        // No silent fallback per review: propagate error to visible UI
        const error = err instanceof Error ? err : new Error(String(err));
        setGlobalApiError(error);
        handleClearMotion();
      }
    },
    [handleClearMotion]
  );

  // 5. Load Explicit Offline Synthetic Fixture (for offline demo testing)
  const handleLoadOfflineFixture = useCallback(
    async (fixtureId: string) => {
      setGlobalApiError(null);
      try {
        const localUrl = `/fixtures/${fixtureId}.json`;
        const res = await fetch(localUrl);
        if (res.ok) {
          const motion = (await res.json()) as MotionFile;
          setCurrentMotion(motion);
          setActiveMotionId(fixtureId);
          setActiveGlossLabel(`${fixtureId} (Fixture locale)`);
          setIsSyntheticAsset(true);
          setCurrentTimeMs(0);
          setIsPlaying(true);
        } else {
          throw new Error(`Fixture locale introuvable (${res.status})`);
        }
      } catch (err: unknown) {
        const error = err instanceof Error ? err : new Error(String(err));
        setGlobalApiError(error);
      }
    },
    []
  );

  // 6. Active Gloss tracker during playback across multi-clip plans
  useEffect(() => {
    if (!currentPlan?.segments || currentPlan.segments.length === 0) return;

    const activeSeg = currentPlan.segments.find(
      (s) => currentTimeMs >= s.start_ms && currentTimeMs < s.start_ms + s.duration_ms
    );

    if (activeSeg) {
      setActiveGlossLabel(activeSeg.gloss_id);
    }
  }, [currentTimeMs, currentPlan]);

  // 7. Playback Animation Frame Loop
  useEffect(() => {
    if (!isPlaying || !currentMotion) {
      lastAnimTimeRef.current = null;
      return;
    }

    const duration = currentMotion.duration_ms || 1000;

    const tick = (now: number) => {
      if (lastAnimTimeRef.current === null) {
        lastAnimTimeRef.current = now;
      }
      const deltaMs = now - lastAnimTimeRef.current;
      lastAnimTimeRef.current = now;

      setCurrentTimeMs((prev) => {
        let next = prev + deltaMs * playbackSpeed;
        if (next >= duration) {
          if (loop) {
            next = next % duration;
          } else {
            next = duration;
            setIsPlaying(false);
          }
        }
        return next;
      });

      animFrameIdRef.current = requestAnimationFrame(tick);
    };

    animFrameIdRef.current = requestAnimationFrame(tick);

    return () => {
      if (animFrameIdRef.current) {
        cancelAnimationFrame(animFrameIdRef.current);
      }
    };
  }, [isPlaying, currentMotion, playbackSpeed, loop]);

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans">
      {/* 1. Ethical & Methodological Alert Banner */}
      <EthicalBanner />

      {/* 2. Header & Navigation */}
      <Header
        activeTab={activeTab}
        onTabChange={setActiveTab}
        apiConnected={apiConnected}
        onCheckHealth={checkHealth}
        onLoadOfflineFixture={handleLoadOfflineFixture}
      />

      {/* Global API Connection Warning Alert */}
      {globalApiError && (
        <div className="max-w-7xl mx-auto w-full px-4 pt-4">
          <ApiErrorAlert
            error={globalApiError}
            onRetry={checkHealth}
            title="Problème de communication avec le serveur API"
          />
        </div>
      )}

      {/* 3. Main Studio Workspace */}
      <main className="max-w-7xl mx-auto w-full px-4 py-6 flex-1">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Active Tab Studio Panel (7 cols on desktop) */}
          <section className="lg:col-span-7 space-y-6">
            {activeTab === 'studio' && (
              <TextToSignStudio
                onLoadPlanSequence={handleLoadPlanSequence}
                onLoadSingleSegment={handleLoadSingleSegment}
                onClearMotion={handleClearMotion}
                onPlanCreated={setCurrentPlan}
              />
            )}

            {activeTab === 'audio' && (
              <AudioAnalysisStudio
                onAudioFeaturesLoaded={(features: AudioFeatures) => {
                  if (currentPlan) {
                    setCurrentPlan({ ...currentPlan, audio_features: features });
                  }
                }}
                onExpressiveToggle={(enabled, energy) => {
                  setExpressiveModulation(enabled);
                  setEnergyFactor(energy);
                }}
              />
            )}

            {activeTab === 'review' && (
              <HumanReviewStudio
                currentPlan={currentPlan}
                onPlanUpdated={setCurrentPlan}
              />
            )}

            {activeTab === 'glossary' && (
              <GlossaryRegistry
                onSelectGloss={(motionId, label) => {
                  handleLoadSingleSegment(motionId, undefined, label);
                  setActiveTab('studio');
                }}
              />
            )}
          </section>

          {/* Right Column: 3D Joint Avatar Player (5 cols on desktop, sticky) */}
          <aside
            aria-label="Lecteur 3D d'Avatar SignaCulture"
            className="lg:col-span-5 space-y-4 lg:sticky lg:top-24"
          >
            <div className="bg-slate-900/80 border border-slate-800 rounded-2xl p-4 shadow-2xl space-y-3">
              <div className="flex items-center justify-between border-b border-slate-800 pb-2.5">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full bg-sky-400 animate-pulse" />
                  <h2 className="font-bold text-slate-100 text-sm">
                    Lecteur 3D Avatar (Three.js WebGL)
                  </h2>
                </div>

                <span className="text-[11px] font-mono text-slate-400">
                  {currentMotion ? `${currentMotion.fps} fps` : 'En attente'}
                </span>
              </div>

              {/* 3D Canvas with overlay */}
              <div className="relative">
                <PlayerOverlay
                  motionId={activeMotionId}
                  rigVersion={currentMotion?.rig_version || 'signaculture-v1'}
                  fps={currentMotion?.fps || 10}
                  durationMs={currentMotion?.duration_ms || 1000}
                  activeGloss={activeGlossLabel}
                  isSynthetic={isSyntheticAsset}
                />

                <AvatarCanvas
                  motion={currentMotion}
                  currentTimeMs={currentTimeMs}
                  expressiveModulation={expressiveModulation}
                  energyFactor={energyFactor}
                  showSkeletonDebug={showSkeletonDebug}
                />
              </div>

              {/* Timeline Transport Controls */}
              <TimelineControls
                isPlaying={isPlaying}
                currentTimeMs={currentTimeMs}
                durationMs={currentMotion?.duration_ms || 1000}
                speed={playbackSpeed}
                loop={loop}
                showSkeletonDebug={showSkeletonDebug}
                onPlayToggle={() => setIsPlaying(!isPlaying)}
                onSeek={(ms) => {
                  setCurrentTimeMs(ms);
                  lastAnimTimeRef.current = null;
                }}
                onReset={() => {
                  setCurrentTimeMs(0);
                  setIsPlaying(false);
                  lastAnimTimeRef.current = null;
                }}
                onSpeedChange={setPlaybackSpeed}
                onLoopToggle={() => setLoop(!loop)}
                onSkeletonToggle={() => setShowSkeletonDebug(!showSkeletonDebug)}
              />

              {/* Instructions footer */}
              <div className="text-[11px] text-slate-400 flex items-center justify-between pt-1 px-1">
                <span>Clic-glisser pour pivoter • Molette pour zoomer</span>
                <span className="text-slate-400">Coords métriques (Y up)</span>
              </div>
            </div>
          </aside>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-800/80 bg-slate-950 py-4 text-center text-xs text-slate-400">
        <p>
          SignaCulture AI — Hackathon IA Éthique 360 (13–25 nov 2026). Équipe d'ingénierie collaborative.
        </p>
      </footer>
    </div>
  );
};

export default App;
