import type { MotionFile } from "../types/contract";

/** How the person said it. */
export type Via = "signes" | "voix" | "texte";

export type Turn = {
  id: number;
  role: "user" | "assistant";
  text: string;
  via?: Via;
  motion?: MotionFile;
  pending?: boolean;
  failed?: boolean;
  shown?: boolean;
  ms?: { recognition?: number; brain?: number; signing?: number };
};

export type Candidate = { gloss: string; label: string; score: number; distance?: number };

/** POST /sign/recognize response. */
export type Recognition = {
  glosses: { gloss: string; label: string; score: number }[];
  segments: { start_ms: number; end_ms: number; candidates: Candidate[] }[];
  text_guess: string;
  confidence: number;
  vocabulary: number;
  frames?: number;
  hands_seen?: number;
  ms: number;
  client_ms?: number;
};

/** The words picked for each recognised sign (index into its candidates), as a French phrase. */
export function choiceText(result: Recognition, picks: number[]) {
  const words = result.segments
    .map((s, k) => s.candidates[picks[k] ?? 0]?.label)
    .filter((w): w is string => Boolean(w));
  const text = words.join(" ");
  return text ? text[0].toUpperCase() + text.slice(1) : "";
}

/** Finished turns -> the brain's messages (failed or empty replies are left out; the last 12 turns are enough). */
export function historyMessages(turns: Turn[]) {
  return turns
    .filter((t) => t.text.trim() && !t.failed && !(t.role === "assistant" && t.pending && !t.text))
    .map((t) => ({ role: t.role, text: t.text.trim() }))
    .slice(-12);
}

/** GET /sign/vocabulary: lexicon size, everyday words, the person's own recorded signs. */
export type Vocabulary = {
  size: number;
  conversation: { gloss: string; label: string }[];
  user: { gloss: string; label: string; count: number }[];
};
