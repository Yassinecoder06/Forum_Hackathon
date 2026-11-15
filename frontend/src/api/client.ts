import {
  AnimationPlan,
  AudioFeatures,
  GlossaryResponse,
  HealthResponse,
  MotionAsset,
  MotionFile,
  RenderRequest,
  RenderResponse,
  ReviewRecord,
  TranslationDraft,
} from '../types/contract';

/**
 * Standard API error carrying detailed backend diagnostics.
 */
export class ApiError extends Error {
  public status?: number;
  public endpoint: string;
  public details?: unknown;

  constructor(message: string, endpoint: string, status?: number, details?: unknown) {
    super(message);
    this.name = 'ApiError';
    this.endpoint = endpoint;
    this.status = status;
    this.details = details;
  }
}

const API_BASE = '/api';

async function handleResponse<T>(res: Response, endpoint: string): Promise<T> {
  if (!res.ok) {
    let errorDetail: unknown = null;
    try {
      errorDetail = await res.json();
    } catch {
      try {
        errorDetail = await res.text();
      } catch {
        errorDetail = null;
      }
    }

    const message =
      (errorDetail && typeof errorDetail === 'object' && 'detail' in errorDetail && typeof (errorDetail as Record<string, unknown>).detail === 'string' && (errorDetail as Record<string, unknown>).detail) ||
      (typeof errorDetail === 'string' && errorDetail) ||
      `Erreur serveur (${res.status} ${res.statusText}) sur l'endpoint ${endpoint}`;

    throw new ApiError(String(message), endpoint, res.status, errorDetail);
  }
  return res.json() as Promise<T>;
}

export const apiClient = {
  /**
   * Health check for FastAPI backend.
   * GET /api/health (proxied to /health)
   */
  async getHealth(): Promise<HealthResponse> {
    const endpoint = `${API_BASE}/health`;
    try {
      const res = await fetch(endpoint, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      return await handleResponse<HealthResponse>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        'Impossible de joindre le serveur backend sur localhost:8000 via /api/health.',
        endpoint,
        undefined,
        err
      );
    }
  },

  /**
   * Fetch controlled glossary entries.
   * GET /api/glossary (proxied to /glossary)
   */
  async getGlossary(): Promise<GlossaryResponse> {
    const endpoint = `${API_BASE}/glossary`;
    try {
      const res = await fetch(endpoint, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      return await handleResponse<GlossaryResponse>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        'Impossible de récupérer le glossaire LSF contrôlé.',
        endpoint,
        undefined,
        err
      );
    }
  },

  /**
   * Look up gloss sequence and lemmas for French input text.
   * POST /api/gloss-lookup (or /api/analyze-text)
   */
  async glossLookup(text: string): Promise<TranslationDraft> {
    const endpoint = `${API_BASE}/gloss-lookup`;
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ text }),
      });
      return await handleResponse<TranslationDraft>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        `Échec de la recherche de gloss pour : "${text}".`,
        endpoint,
        undefined,
        err
      );
    }
  },

  /**
   * Analyze French text with NLP lemmatizer / expression matcher.
   * POST /api/analyze-text
   */
  async analyzeText(text: string): Promise<TranslationDraft> {
    const endpoint = `${API_BASE}/analyze-text`;
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify({ text }),
      });
      return await handleResponse<TranslationDraft>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        `Échec de l'analyse linguistique pour : "${text}".`,
        endpoint,
        undefined,
        err
      );
    }
  },

  /**
   * Compose an animation plan from text and optional audio prosodic features.
   * POST /api/compose-animation
   */
  async composeAnimation(
    text: string,
    audioFeatures?: AudioFeatures
  ): Promise<AnimationPlan> {
    const endpoint = `${API_BASE}/compose-animation`;
    try {
      const payload: { text: string; audio_features?: AudioFeatures } = { text };
      if (audioFeatures) {
        payload.audio_features = audioFeatures;
      }
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(payload),
      });
      return await handleResponse<AnimationPlan>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        'Échec de la composition du plan d\'animation.',
        endpoint,
        undefined,
        err
      );
    }
  },

  /**
   * Fetch motion asset metadata by motion_id.
   * GET /api/motions/{motion_id}
   */
  async getMotion(motionId: string): Promise<MotionAsset> {
    const endpoint = `${API_BASE}/motions/${encodeURIComponent(motionId)}`;
    try {
      const res = await fetch(endpoint, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      return await handleResponse<MotionAsset>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        `Mouvement introuvable pour motion_id : "${motionId}".`,
        endpoint,
        undefined,
        err
      );
    }
  },

  /**
   * Fetch 3D Motion JSON file.
   * Handles /assets/{filename}, data/approved_motion_assets/{filename}, or direct URLs.
   * CRITICAL: Ensures /assets/... from backend compose is routed through /api/assets/...
   * to be proxied by Vite to http://localhost:8000/assets/...
   */
  async getMotionFile(urlOrPath: string): Promise<MotionFile> {
    let cleanUrl = urlOrPath;
    if (cleanUrl.startsWith('data/approved_motion_assets/')) {
      const filename = cleanUrl.replace('data/approved_motion_assets/', '');
      cleanUrl = `${API_BASE}/assets/${filename}`;
    } else if (cleanUrl.startsWith('/assets/')) {
      cleanUrl = `${API_BASE}${cleanUrl}`;
    } else if (!cleanUrl.startsWith('/') && !cleanUrl.startsWith('http')) {
      cleanUrl = `${API_BASE}/assets/${cleanUrl}`;
    }

    try {
      const res = await fetch(cleanUrl, {
        method: 'GET',
        headers: { Accept: 'application/json' },
      });
      return await handleResponse<MotionFile>(res, cleanUrl);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        `Impossible de charger les données 3D depuis : ${cleanUrl}.`,
        cleanUrl,
        undefined,
        err
      );
    }
  },

  /**
   * Analyze audio WAV file for pitch, energy, pauses and emotion hypothesis.
   * POST /api/analyze-audio (multipart/form-data)
   */
  async analyzeAudio(file: File): Promise<AudioFeatures> {
    const endpoint = `${API_BASE}/analyze-audio`;
    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        body: formData,
      });
      return await handleResponse<AudioFeatures>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        `Erreur lors de l'analyse audio de ${file.name}.`,
        endpoint,
        undefined,
        err
      );
    }
  },

  /**
   * Submit human review record.
   * POST /api/review
   */
  async submitReview(review: {
    plan_id: string;
    reviewer: string;
    status: 'approved' | 'rejected';
    notes: string;
  }): Promise<ReviewRecord> {
    const endpoint = `${API_BASE}/review`;
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(review),
      });
      return await handleResponse<ReviewRecord>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        'Erreur lors de la soumission de la revue humaine.',
        endpoint,
        undefined,
        err
      );
    }
  },

  /**
   * Request render for preview or publication.
   * POST /api/render
   */
  async requestRender(request: RenderRequest): Promise<RenderResponse> {
    const endpoint = `${API_BASE}/render`;
    try {
      const res = await fetch(endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(request),
      });
      return await handleResponse<RenderResponse>(res, endpoint);
    } catch (err) {
      if (err instanceof ApiError) throw err;
      throw new ApiError(
        `Erreur lors de la requête de rendu (${request.purpose}).`,
        endpoint,
        undefined,
        err
      );
    }
  },
};
