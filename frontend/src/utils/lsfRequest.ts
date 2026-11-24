import type { AudioFeatures, NonManualType } from "../types/contract";

export type LsfEngine = "b2" | "c0";

export const MARKER_LABEL: Record<NonManualType, string> = {
  question_yn: "Question oui/non · sourcils levés",
  question_wh: "Question ouverte · sourcils froncés",
  negation: "Négation · hochement de tête",
  emphasis: "Mot accentué · geste appuyé, hochement",
};

export const EMOTION_LABEL: Record<string, string> = {
  neutral: "neutre", happy: "joie", angry: "colère", sad: "tristesse", surprised: "surprise", fearful: "peur", disgusted: "dégoût",
};

/** emotion2vec scores (numbers only) from an AudioFeatures.emotion_hypothesis, or null. */
export function voiceScores(hypothesis?: Record<string, number | string> | null) {
  if (!hypothesis) return null;
  const scores = Object.fromEntries(Object.entries(hypothesis).filter(([, v]) => typeof v === "number" && Number.isFinite(v))) as Record<string, number>;
  return Object.keys(scores).length ? scores : null;
}

/** Dominant emotional label of voice scores (neutral when neutral/other/unknown wins). */
export function dominantEmotion(scores?: Record<string, number> | null) {
  if (!scores) return null;
  const [label, value] = Object.entries(scores).sort((a, b) => b[1] - a[1])[0] ?? [];
  if (!label) return null;
  return { label: label in EMOTION_LABEL ? label : "neutral", confidence: value };
}

/** Body of POST /lsf/generate for the current choices (Studio panel, film and live modes). */
export function generateBody(text: string, engine: LsfEngine, lsfOrder: boolean, nonmanual: boolean, edited?: string[][] | null,
  emotion?: Record<string, number> | null, voice?: Pick<AudioFeatures, "prosody" | "emotion_timeline"> | null,
  focus?: "left" | "centre" | "right" | null) {
  return {
    text,
    engine,
    order: lsfOrder ? "lsf" : "french",
    nonmanual,
    ...(engine === "b2" && lsfOrder && edited ? { glosses: edited } : {}),
    ...(emotion ? { emotion } : {}),
    // Words the speaker stressed (and the emotion over time): the same step also reads "!", capitals and "très" in the text.
    ...(voice?.prosody ? { prosody: voice.prosody } : {}),
    ...(voice?.emotion_timeline?.length ? { emotion_timeline: voice.emotion_timeline.slice(0, 64) } : {}),
    // [pointing] companion: where the thing talked about is in the picture; demonstratives and pronouns point there
    ...(focus ? { focus } : {}),
  };
}
