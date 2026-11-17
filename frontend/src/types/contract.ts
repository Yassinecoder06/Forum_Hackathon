/**
 * SignaCulture AI — Integration Contract Types (docs/INTEGRATION_CONTRACT.md)
 * Contract is authoritative for all integration schemas and field types.
 * All timestamps are in milliseconds (ms). Keys are snake_case.
 */

export type ReviewStatus = "pending" | "approved" | "rejected";
export type RightsStatus = "cleared" | "pending";
export type RigVersion = "signaculture-v1";

export interface GlossEntry {
  gloss_id: string;
  concept_fr: string;
  gloss: string;
  language: "LSF";
  variant?: string;
  source_id: string;
  motion_id?: string;
  aliases: string[];
  review_status: ReviewStatus;
}

export interface MotionAsset {
  motion_id: string;
  gloss_id: string;
  path: string;
  rig_version: RigVersion;
  fps: number;
  duration_ms: number;
  license: string;
  quality_status: string;
  rights_status: RightsStatus;
  animation_review: ReviewStatus;
  synthetic: boolean;
}

export interface TranslationDraft {
  source_text: string;
  tokens: string[];
  lemmas: string[];
  gloss_sequence: GlossEntry[];
  unknowns: string[];
  linguistic_review: ReviewStatus | string;
  translation_validated: boolean;
}

export interface AnimationPlanSegment {
  gloss_id: string;
  motion_id: string;
  start_ms: number;
  duration_ms: number;
  asset_url?: string;
  /** Part of the source clip to play (its signing core, on captured frame times); absent = whole clip. */
  clip_start_ms?: number;
  clip_end_ms?: number;
}

export interface AudioFrame {
  start_ms: number;
  end_ms: number;
  pitch_hz: number | null;
  energy: number;
  is_pause: boolean;
}

export interface AudioFeatures {
  duration_ms: number;
  frames: AudioFrame[];
  emotion_hypothesis: null | Record<string, number | string>;
  model_version: string;
  warnings: string[];
  transcription?: null | {
    text: string;
    language: "fr";
    model_id: string;
    model_revision: string;
    device: "cuda" | "cpu";
    compute_type: string;
    segments: { start_ms: number; end_ms: number; text: string }[];
  };
}

export interface AnimationPlan {
  transition_ms?: number;
  timing_mode?: "original" | "audio_voiced";
  plan_id: string;
  source_text: string;
  target_language: "LSF";
  segments: AnimationPlanSegment[];
  unknowns: string[];
  duration_ms: number;
  linguistic_review: ReviewStatus | string;
  animation_review: ReviewStatus | string;
  rights_status: RightsStatus | string;
  render_allowed_for_publication: boolean;
  audio_features?: AudioFeatures;
}

export interface MotionFrame {
  playback_quality?: Record<string, boolean | string>;
  playback_origin?: "source" | "transition" | "audio_sample" | "endpoint_hold";
  time_ms: number;
  joints: Record<string, [number, number, number]>;
  face?: Record<string, number> | null;
  hand_tracking?: Record<string, { observed: boolean; source: string }>;
}

export interface MotionAttribution {
  credit: string;
  source_url: string;
  license: string;
  license_url?: string;
  changes?: string;
}

export interface MotionFile {
  playback_window?: { start_ms: number; end_ms: number; method?: string };
  face_calibration?: { brow_left: number; brow_right: number };
  synthetic?: boolean;
  attribution?: MotionAttribution;
  attributions?: MotionAttribution[];
  rig_version: RigVersion | string;
  fps: number;
  duration_ms: number;
  joints: string[];
  frames: MotionFrame[];
}

export interface ReviewRecord {
  reviewer: string;
  status: "approved" | "rejected";
  notes: string;
  version: string;
  date: string;
}

export interface RenderRequest {
  plan_id: string;
  purpose: "preview" | "publication";
}

/**
 * Backend /render response per INTEGRATION_CONTRACT.md line 9:
 * preview returns verified plan and assets, publication blocked until ALL approvals and rights checked.
 * No claim video exported unless actual video file generated.
 */
export interface RenderResponse {
  plan: AnimationPlan;
  assets: MotionAsset[];
  video_exported: boolean;
}

export interface HealthResponse {
  status: "ok" | string;
}

export interface GlossaryResponse {
  entries: GlossEntry[];
}
