/**
 * SignaCulture AI — Integration Contract Types (docs/architecture/INTEGRATION_CONTRACT.md)
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
  /** Voice prosody for this sign (timing_mode "audio_words"); acoustic estimates. */
  emphasis?: number | null;
  liveliness?: number | null;
  hold_ms?: number | null;
  word_start_ms?: number | null;
  word_end_ms?: number | null;
}

export interface AudioFrame {
  start_ms: number;
  end_ms: number;
  pitch_hz: number | null;
  energy: number;
  is_pause: boolean;
}

export interface EmotionWindow {
  start_ms: number;
  end_ms: number;
  scores: Record<string, number>;
}

export interface ProsodyWord {
  text: string;
  start_ms: number;
  end_ms: number;
  prominence: number;
  emphasized: boolean;
  syl_per_s: number;
  liveliness: number;
}

export interface AudioFeatures {
  duration_ms: number;
  frames: AudioFrame[];
  emotion_hypothesis: null | Record<string, number | string>;
  /** Word-level prominence, pace and pitch liveliness (needs transcript word timings). */
  prosody?: null | { pace_syl_per_s: number; words: ProsodyWord[] };
  /** Overlapping ~2 s windows of emotion2vec hypotheses, ordered by start_ms. */
  emotion_timeline?: null | EmotionWindow[];
  model_version: string;
  warnings: string[];
  transcription?: null | {
    text: string;
    language: "fr";
    model_id: string;
    model_revision: string;
    device: "cuda" | "cpu";
    compute_type: string;
    segments: {
      start_ms: number;
      end_ms: number;
      text: string;
      words?: { start_ms: number; end_ms: number; text: string; probability: number }[] | null;
    }[];
  };
}

export interface AnimationPlan {
  transition_ms?: number;
  timing_mode?: "original" | "audio_voiced" | "audio_words";
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
  /** Head-nod curve (0..1) on emphasised signs; drives an additive head pitch. */
  nod?: number;
}

export interface MotionAttribution {
  credit: string;
  source_url: string;
  license: string;
  license_url?: string;
  changes?: string;
}

/** Present on motions produced by POST /lsf/generate (experimental engines). */
export interface MotionGeneration {
  engine: "c0" | "b2" | string;
  lemmas?: string[];
  unknown?: string[];
  seconds?: number;
  /** B2 word order: French text order, rule-based LSF order, or a gloss sequence edited by the user. */
  order?: "french" | "lsf" | "edited";
  gloss_analysis?: GlossAnalysis;
  /** Voice emotion applied to the speed and size of the manual signing (never the face). Heuristic. */
  emotion?: { label: string; confidence: number; speed: number; amplitude: number };
  /** Stressed words (voice or written cues): larger movement, a short hold and a head nod on their signs. */
  prosody?: { stressed: { word: string; cue: string; emphasis: number; start_ms: number; end_ms: number }[] };
  translation_validated: false;
  linguistic_review: string;
}

/** Rule-based LSF word order (POST /lsf/order): simplified rules, not checked by an LSF reader. */
export type NonManualType = "question_yn" | "question_wh" | "negation" | "emphasis";
export interface GlossClause {
  text: string;
  glosses: string[];
  markers: NonManualType[];
  rules: string[];
  /** [film-speakers] quoted / reported speech: the clause is signed in the role of this referent. */
  role_shift?: { referent: string; label: string; kind: "quote" | "reported"; locus?: string | null } | null;
  /** [film-speakers] referent id of the voice of the clause (speaker of the cue or of the quote). */
  voice?: string | null;
  /** [film-speakers] introductions, mentions, pronoun indexes and role shifts, for a later pointing layer. */
  events?: { type: "introduce" | "mention" | "index" | "role_shift"; referent: string | null; token?: string; gloss_index?: number | null }[];
}
/** [film-speakers] One entry of the scene's referent table (lsf_order). Locus in the viewer's frame. */
export interface Referent {
  id: string;
  label: string;
  kind: "person" | "group" | "place" | "thing" | string;
  gender: "m" | "f" | null;
  number: "sg" | "pl";
  locus: "L" | "C" | "R" | null;
  speaker: boolean;
  mentions: number;
  last_mention: number;
}
export interface GlossAnalysis {
  clauses: GlossClause[];
  glosses: string[];
  gloss_text: string;
  method: string;
  validated: false;
  referents?: Referent[];
}

export interface MotionFile {
  generation?: MotionGeneration;
  playback_window?: { start_ms: number; end_ms: number; method?: string };
  /** Neutral face channels (brows; engines also give mouth_width, mouth_corner, eye_open_left/right). */
  face_calibration?: { brow_left: number; brow_right: number; [channel: string]: number };
  /** Grammatical non-manual markers baked into the face channels (nm_*) of each frame. */
  nonmanual?: { start_ms: number; end_ms: number; type: NonManualType; strength?: number }[];
  /** [film-speakers] Clauses signed in the role of a quoted / reported speaker (times inside this motion). */
  role_shift?: { start_ms: number; end_ms: number; referent: string; label: string; locus?: string | null; kind: string }[];
  /** [pointing] Index-finger points toward the locus (L / C / R, viewer's frame) of a person / thing the text refers to (times inside this motion). */
  pointing?: { start_ms: number; end_ms: number; locus: "L" | "C" | "R"; referent?: string | null; label?: string | null; kind?: "pronoun" | "addressee" | "mention" | "deictic"; token?: string | null; clause?: number; gloss_index?: number | null }[];
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
