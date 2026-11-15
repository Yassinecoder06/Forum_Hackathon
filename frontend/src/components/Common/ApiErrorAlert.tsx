import React from 'react';
import { AlertTriangle, RefreshCw, ServerOff } from 'lucide-react';
import { ApiError } from '../../api/client';

interface ApiErrorAlertProps {
  error: Error | ApiError | null;
  onRetry?: () => void;
  title?: string;
}

export const ApiErrorAlert: React.FC<ApiErrorAlertProps> = ({
  error,
  onRetry,
  title = 'Problème de connexion API',
}) => {
  if (!error) return null;

  const endpoint = error instanceof ApiError ? error.endpoint : undefined;
  const status = error instanceof ApiError ? error.status : undefined;

  return (
    <div
      role="alert"
      className="p-4 rounded-xl border border-rose-500/40 bg-rose-950/40 text-rose-200 shadow-lg backdrop-blur-sm transition-all"
    >
      <div className="flex items-start gap-3">
        <div className="p-2 bg-rose-900/60 rounded-lg text-rose-300 shrink-0 mt-0.5">
          {status === undefined ? (
            <ServerOff className="w-5 h-5" />
          ) : (
            <AlertTriangle className="w-5 h-5" />
          )}
        </div>
        <div className="flex-1 space-y-1">
          <div className="flex items-center justify-between gap-2">
            <h4 className="font-semibold text-rose-100">{title}</h4>
            {status && (
              <span className="text-xs px-2 py-0.5 bg-rose-900/80 rounded border border-rose-700/50 text-rose-200 font-mono">
                Code HTTP {status}
              </span>
            )}
          </div>
          <p className="text-sm text-rose-200/90 leading-relaxed">{error.message}</p>
          {endpoint && (
            <p className="text-xs font-mono text-rose-300/70 pt-1">
              Endpoint concerné : <span className="underline">{endpoint}</span>
            </p>
          )}
          <p className="text-xs text-rose-300/80 pt-1 italic">
            Assurez-vous que le backend FastAPI est démarré sur le port 8000 (ex: <code>uvicorn backend.app.main:app --port 8000</code>).
          </p>
        </div>
        {onRetry && (
          <button
            onClick={onRetry}
            className="px-3 py-1.5 bg-rose-800 hover:bg-rose-700 active:bg-rose-900 text-white text-xs font-medium rounded-lg flex items-center gap-1.5 transition-colors shrink-0 shadow-sm"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Réessayer
          </button>
        )}
      </div>
    </div>
  );
};
