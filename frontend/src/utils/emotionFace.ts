// Voice-emotion hypotheses -> ARKit blendshape weights for the avatar face.
// Scores come from emotion2vec+ and are unvalidated model hypotheses. While the avatar
// signs, LSF eyebrows/head carry grammar: emotion is capped there and only nudges brows.
import type { EmotionWindow } from "../types/contract";

export type EmotionScores = Record<string, number>;
export type FaceMode = "rest" | "signing";

/** Eyes/cheeks/mouth-corner cap while signing (jaw and head are never touched; brows are only nudged). */
export const SIGNING_FACE_CAP = 0.6;
/** Additive brow nudge cap while signing; captured grammatical brows stay dominant. */
export const SIGNING_BROW_CAP = 0.1;
/** Probability floor ignored as noise, so near-neutral voices give a neutral face. */
export const NOISE_FLOOR = 0.15;
/** Perceptual curve: a clear but not overwhelming score (e.g. 0.5) should read clearly on the face. */
export const STRENGTH_GAMMA = 0.6;
/** Exponential smoothing time constant for expression changes. */
export const SMOOTHING_MS = 300;

// FACS-inspired recipes using the ARKit names the avatar exposes; weights at full intensity.
const pair = (base: string, w: number) => ({ [`${base}Left`]: w, [`${base}Right`]: w });
export const EMOTION_RECIPES: Record<string, Record<string, number>> = {
  // Tuned on CaFE clips with the ReadyEva avatar: eye squint lowered so happy eyes stay open;
  // sad and disgust raised because their mouth/nose shapes are subtle on this mesh.
  happy: { ...pair("mouthSmile", 0.75), ...pair("cheekSquint", 0.45), ...pair("eyeSquint", 0.1), ...pair("browOuterUp", 0.15) },
  sad: { ...pair("mouthFrown", 0.95), mouthShrugLower: 0.35, ...pair("mouthPress", 0.2), browInnerUp: 0.9, ...pair("browDown", 0.15), ...pair("eyeSquint", 0.2) },
  angry: { ...pair("browDown", 0.65), ...pair("noseSneer", 0.45), ...pair("mouthPress", 0.5), ...pair("eyeSquint", 0.5) },
  // Eyes carry surprise/fear while signing, when brows are only nudged and the jaw is left to mouthing.
  surprised: { browInnerUp: 0.6, ...pair("browOuterUp", 0.6), ...pair("eyeWide", 0.9), jawOpen: 0.25 },
  fearful: { browInnerUp: 0.7, ...pair("eyeWide", 0.85), ...pair("mouthStretch", 0.6), ...pair("browDown", 0.1) },
  disgusted: { ...pair("noseSneer", 1), ...pair("mouthUpperUp", 0.75), ...pair("browDown", 0.5), ...pair("mouthFrown", 0.5), ...pair("cheekSquint", 0.35), ...pair("eyeSquint", 0.3) },
};

export const EMOTION_KEYS = [...new Set(Object.values(EMOTION_RECIPES).flatMap(Object.keys))];
const isBrow = (name: string) => name.startsWith("brow");
const clamp01 = (v: number) => (Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : 0);

function toScores(raw: Record<string, number | string> | null | undefined): EmotionScores | null {
  if (!raw) return null;
  const out: EmotionScores = {};
  for (const [name, value] of Object.entries(raw)) out[name] = clamp01(Number(value));
  return out;
}

/**
 * Anchor a short-window score to the whole-clip score. Short windows hear little speech and
 * are noisy (on CaFE, a correctly classified fearful clip had every window say "happy"), so a
 * window can only strengthen the emotions it shares with the clip:
 *   agreement = Σ min(clip, window)                      (distribution overlap, 0..1)
 *   out = clip × (0.5 + 0.5·agreement) + window × 0.5·agreement
 * A window that disagrees leaves the clip's emotion at half strength instead of flipping it.
 */
export function anchorToClip(windowScores: EmotionScores, clip: EmotionScores): EmotionScores {
  const names = new Set([...Object.keys(clip), ...Object.keys(windowScores)]);
  let agreement = 0;
  for (const name of names) agreement += Math.min(clip[name] ?? 0, windowScores[name] ?? 0);
  agreement = clamp01(agreement);
  const out: EmotionScores = {};
  for (const name of names)
    out[name] = clamp01((clip[name] ?? 0) * (0.5 + 0.5 * agreement) + (windowScores[name] ?? 0) * 0.5 * agreement);
  return out;
}

function interpolateTimeline(timeline: EmotionWindow[], timeMs: number): EmotionScores {
  const centre = (w: EmotionWindow) => (w.start_ms + w.end_ms) / 2;
  if (timeMs <= centre(timeline[0])) return toScores(timeline[0].scores)!;
  const last = timeline[timeline.length - 1];
  if (timeMs >= centre(last)) return toScores(last.scores)!;
  let i = 0;
  while (centre(timeline[i + 1]) < timeMs) i++;
  const a = timeline[i], b = timeline[i + 1];
  const t = (timeMs - centre(a)) / Math.max(1, centre(b) - centre(a));
  const out: EmotionScores = {};
  for (const name of new Set([...Object.keys(a.scores), ...Object.keys(b.scores)]))
    out[name] = clamp01((a.scores[name] ?? 0) * (1 - t) + (b.scores[name] ?? 0) * t);
  return out;
}

/**
 * Scores at a playback time: windows interpolated between their centres and anchored to the
 * whole-clip score; the clip score alone when there is no timeline.
 */
export function sampleEmotion(
  timeline: EmotionWindow[] | null | undefined,
  utterance: Record<string, number | string> | null | undefined,
  timeMs: number,
): EmotionScores | null {
  const clip = toScores(utterance);
  if (!timeline?.length) return clip;
  const windowScores = interpolateTimeline(timeline, timeMs);
  return clip ? anchorToClip(windowScores, clip) : windowScores;
}

/** Blendshape weights for the given scores. Brow keys while signing are an additive nudge. */
export function emotionToBlendshapes(
  scores: EmotionScores | null,
  mode: FaceMode,
  intensity = 1,
): Record<string, number> {
  const weights: Record<string, number> = {};
  if (!scores) return weights;
  const gain = clamp01(intensity);
  for (const [emotion, recipe] of Object.entries(EMOTION_RECIPES)) {
    const strength = (Math.max(0, (scores[emotion] ?? 0) - NOISE_FLOOR) / (1 - NOISE_FLOOR)) ** STRENGTH_GAMMA;
    if (!strength) continue;
    for (const [name, w] of Object.entries(recipe)) weights[name] = (weights[name] ?? 0) + strength * w * gain;
  }
  for (const name of Object.keys(weights)) {
    let value = clamp01(weights[name]);
    if (mode === "signing") {
      // Mouth opening is lexical mouthing while signing; leave it to captured data.
      if (name === "jawOpen") value = 0;
      value = Math.min(value, isBrow(name) ? SIGNING_BROW_CAP : SIGNING_FACE_CAP);
    }
    weights[name] = value;
  }
  return weights;
}

/** Frame-rate independent exponential easing of blendshape weights. */
export function smoothWeights(
  previous: Record<string, number>,
  target: Record<string, number>,
  dtMs: number,
  tauMs = SMOOTHING_MS,
): Record<string, number> {
  const k = dtMs <= 0 ? 0 : 1 - Math.exp(-dtMs / tauMs);
  const out: Record<string, number> = {};
  for (const name of new Set([...Object.keys(previous), ...Object.keys(target)])) {
    const from = previous[name] ?? 0, to = target[name] ?? 0;
    const value = from + (to - from) * k;
    if (value > 1e-4 || to > 0) out[name] = clamp01(value);
  }
  return out;
}

export function dominantEmotion(scores: EmotionScores | null): string | null {
  if (!scores) return null;
  let best: string | null = null, bestScore = 0;
  for (const [name, v] of Object.entries(scores)) if (v > bestScore) { best = name; bestScore = v; }
  return best;
}
