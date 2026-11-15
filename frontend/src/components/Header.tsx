import React from 'react';
import {
  Sparkles,
  Layers,
  FileAudio,
  ShieldCheck,
  BookOpen,
  Wifi,
  WifiOff,
  RefreshCw,
  Box,
} from 'lucide-react';

export type TabType = 'studio' | 'audio' | 'review' | 'glossary';

interface HeaderProps {
  activeTab: TabType;
  onTabChange: (tab: TabType) => void;
  apiConnected: boolean | null;
  onCheckHealth: () => void;
  onLoadOfflineFixture: (fixtureId: string) => void;
}

const OFFLINE_FIXTURES = [
  { id: 'SYNTHETIC_001', label: '001 - Bonjour' },
  { id: 'SYNTHETIC_002', label: '002 - Merci' },
  { id: 'SYNTHETIC_003', label: '003 - Culture' },
  { id: 'SYNTHETIC_004', label: '004 - Musée' },
  { id: 'SYNTHETIC_005', label: '005 - Art' },
  { id: 'SYNTHETIC_006', label: '006 - Bienvenue' },
];

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  onTabChange,
  apiConnected,
  onCheckHealth,
  onLoadOfflineFixture,
}) => {
  return (
    <header className="border-b border-slate-800 bg-slate-900/90 backdrop-blur sticky top-0 z-30 shadow-md">
      <div className="max-w-7xl mx-auto px-4 py-3 flex flex-col md:flex-row items-center justify-between gap-3">
        {/* Brand & Title */}
        <div className="flex items-center gap-3 w-full md:w-auto justify-between md:justify-start">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-500 via-indigo-600 to-purple-600 flex items-center justify-center text-white shadow-lg shadow-sky-500/20">
              <Sparkles className="w-5 h-5" />
            </div>
            <div>
              <div className="flex items-center gap-2">
                <span className="font-extrabold text-slate-100 text-base tracking-tight">
                  SignaCulture AI
                </span>
                <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-sky-950 text-sky-400 border border-sky-800/60 uppercase">
                  Studio LSF
                </span>
              </div>
              <p className="text-[11px] text-slate-400 hidden sm:block">
                Studio culturel & avatar 3D éthique — Langue des Signes Française
              </p>
            </div>
          </div>

          {/* Backend Status Badge (Mobile) */}
          <div className="flex items-center md:hidden">
            <button
              onClick={onCheckHealth}
              className={`px-2 py-1 rounded-full text-[10px] font-semibold border flex items-center gap-1.5 ${
                apiConnected === true
                  ? 'bg-emerald-950/80 text-emerald-300 border-emerald-600/60'
                  : apiConnected === false
                  ? 'bg-rose-950/80 text-rose-300 border-rose-600/60'
                  : 'bg-slate-800 text-slate-400 border-slate-700'
              }`}
            >
              {apiConnected === true ? <Wifi className="w-3 h-3" /> : <WifiOff className="w-3 h-3" />}
              <span>{apiConnected === true ? 'API 8000 OK' : 'API Hors-ligne'}</span>
            </button>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="flex items-center gap-1 bg-slate-950/80 p-1 rounded-xl border border-slate-800 overflow-x-auto max-w-full">
          <button
            onClick={() => onTabChange('studio')}
            className={`px-3.5 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === 'studio'
                ? 'bg-sky-500 text-slate-950 shadow-md shadow-sky-500/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <Layers className="w-3.5 h-3.5" />
            Studio Texte-à-Signe
          </button>

          <button
            onClick={() => onTabChange('audio')}
            className={`px-3.5 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === 'audio'
                ? 'bg-indigo-600 text-white shadow-md shadow-indigo-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <FileAudio className="w-3.5 h-3.5" />
            Audio & Prosodie
          </button>

          <button
            onClick={() => onTabChange('review')}
            className={`px-3.5 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === 'review'
                ? 'bg-emerald-600 text-white shadow-md shadow-emerald-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
            Revue & Export
          </button>

          <button
            onClick={() => onTabChange('glossary')}
            className={`px-3.5 py-2 rounded-lg text-xs font-semibold flex items-center gap-1.5 transition-all shrink-0 ${
              activeTab === 'glossary'
                ? 'bg-purple-600 text-white shadow-md shadow-purple-600/20'
                : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
            }`}
          >
            <BookOpen className="w-3.5 h-3.5" />
            Glossaire LSF
          </button>
        </nav>

        {/* Backend Status & Offline Fixture Selector */}
        <div className="hidden md:flex items-center gap-2.5">
          {/* Quick Offline Fixture Dropdown */}
          <div className="flex items-center gap-1 bg-slate-950 px-2.5 py-1 rounded-xl border border-slate-800 text-xs text-slate-300">
            <Box className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-[11px] text-slate-400 mr-1">Fixture locale :</span>
            <select
              onChange={(e) => {
                if (e.target.value) onLoadOfflineFixture(e.target.value);
              }}
              defaultValue=""
              className="bg-transparent text-[11px] text-sky-400 font-mono focus:outline-none cursor-pointer"
            >
              <option value="" disabled className="bg-slate-900 text-slate-400">
                Choisir fixture...
              </option>
              {OFFLINE_FIXTURES.map((f) => (
                <option key={f.id} value={f.id} className="bg-slate-900 text-slate-200">
                  {f.label}
                </option>
              ))}
            </select>
          </div>

          {/* Health Pill Button */}
          <button
            onClick={onCheckHealth}
            title="Tester la connexion au serveur FastAPI (GET /api/health)"
            className={`px-3 py-1.5 rounded-xl text-xs font-semibold border flex items-center gap-2 transition-all ${
              apiConnected === true
                ? 'bg-emerald-950/60 hover:bg-emerald-900/60 text-emerald-300 border-emerald-600/50'
                : apiConnected === false
                ? 'bg-rose-950/60 hover:bg-rose-900/60 text-rose-300 border-rose-600/50'
                : 'bg-slate-950 hover:bg-slate-800 text-slate-400 border-slate-700'
            }`}
          >
            {apiConnected === true ? (
              <>
                <span className="relative flex h-2 w-2">
                  <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
                  <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
                </span>
                <span>API Port 8000 OK</span>
              </>
            ) : apiConnected === false ? (
              <>
                <WifiOff className="w-3.5 h-3.5 text-rose-400" />
                <span>API Déconnectée</span>
                <RefreshCw className="w-3 h-3 text-rose-400 ml-0.5" />
              </>
            ) : (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Vérification...</span>
              </>
            )}
          </button>
        </div>
      </div>
    </header>
  );
};
