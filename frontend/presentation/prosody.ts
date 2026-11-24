// Word stress from the voice, for the live presentation. A line-by-line port of the server's analyse_prosody
// (backend/app/services/prosody.py, Hamza's word prominence: energy, pitch, lengthening), fed with the acoustic
// frames of POST /analyze-audio (fast, no transcription) and the word times the live stream already has, so Whisper
// never runs twice. Acoustic estimates, not linguistic stress annotation.
export type Frame = { start_ms: number; end_ms: number; pitch_hz: number | null; energy: number; is_pause: boolean };
export type TimedWord = { text: string; start_ms: number; end_ms: number };
export type ProsodyWord = TimedWord & { prominence: number; emphasized: boolean; syl_per_s: number; liveliness: number };
export type Prosody = { pace_syl_per_s: number; words: ProsodyWord[] };

type Key = "energy" | "pitch" | "length";
const WEIGHTS: Record<Key, number> = { energy: 0.45, pitch: 0.35, length: 0.2 };
const STD_FLOOR: Record<Key, number> = { energy: 2.0, pitch: 1.5, length: 0.04 }; // dB, semitones, s/syllable
const PROMINENCE_MIN = 1.0;
const RAW_MIN: Record<Key, number> = { energy: 3.0, pitch: 2.0, length: 1.3 };   // +dB, +semitones, x median
const PHRASE_PAUSE_MS = 300, MIN_PHRASE_FOR_TREND = 4;
const VOWELS = /[aeiouyàâäéèêëîïôöùûüÿœæ]+/gi;

const normalize = (t: string) => t.toLowerCase().replace(/[^\p{L}\p{N}_]+/gu, "");
export function syllables(text: string) {
  const word = text.toLowerCase();
  let count = (word.match(VOWELS) ?? []).length;
  if (count > 1 && /[^aeiouyàâäéèêëîïôöùûüÿœæ](e|es|ent)$/.test(word)) count--;
  return Math.max(1, count);
}
const mean = (v: number[]) => v.reduce((a, b) => a + b, 0) / v.length;
const median = (v: number[]) => { const s = [...v].sort((a, b) => a - b), m = s.length >> 1; return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2; };
const pstdev = (v: number[]) => { const m = mean(v); return Math.sqrt(mean(v.map((x) => (x - m) ** 2))); };
const framesIn = (frames: Frame[], start: number, end: number) => frames.filter((f) => f.end_ms > start && f.start_ms < end);

/** Value each word would have without stress: per-phrase declination trend (leave-one-out), else the median. */
function expected(values: number[], times: number[], phrases: number[][]) {
  const out = values.map(() => median(values));
  for (const phrase of phrases) {
    if (phrase.length < MIN_PHRASE_FOR_TREND) continue;
    for (const i of phrase) {
      const others = phrase.filter((j) => j !== i), ts = others.map((j) => times[j]), vs = others.map((j) => values[j]);
      const mt = mean(ts), mv = mean(vs), v = ts.reduce((a, t) => a + (t - mt) ** 2, 0);
      const slope = v ? ts.reduce((a, t, k) => a + (t - mt) * (vs[k] - mv), 0) / v : 0;
      out[i] = mv + slope * (times[i] - mt);
    }
  }
  return out;
}
const zscores = (values: number[], floor: number) => { const m = mean(values), s = Math.max(pstdev(values), floor); return values.map((v) => (v - m) / s); };

export function analyseProsody(frames: Frame[], input: TimedWord[]): Prosody | null {
  const words = input.filter((w) => normalize(w.text));
  if (!words.length) return null;
  const pitches = frames.flatMap((f) => (f.pitch_hz ? [f.pitch_hz] : []));
  const reference = pitches.length ? median(pitches) : null;
  const semitones = (hz: number) => 12 * Math.log2(hz / reference!);
  const raw = words.map((w) => {
    const { start_ms: start, end_ms: end } = w;
    const inside = framesIn(frames, start, end);
    const voicedOnly = inside.filter((f) => !f.is_pause);
    const voiced = voicedOnly.length ? voicedOnly : inside;
    const energy = voiced.length ? mean(voiced.map((f) => f.energy)) : 0;
    const wordPitch = voiced.flatMap((f) => (f.pitch_hz ? [f.pitch_hz] : []));
    const count = syllables(w.text), duration = Math.max(1, end - start) / 1000;
    const voicedS = inside.filter((f) => !f.is_pause).reduce((a, f) => a + Math.min(end, f.end_ms) - Math.max(start, f.start_ms), 0) / 1000;
    const centre = (start + end) / 2;
    const around = framesIn(frames, centre - 1000, centre + 1000).flatMap((f) => (f.pitch_hz ? [f.pitch_hz] : []));
    const spread = around.length > 2 && reference ? pstdev(around.map(semitones)) : 0;
    return { text: w.text, start_ms: start, end_ms: end, syllables: count, energy: 20 * Math.log10(Math.max(energy, 1e-4)),
      pitch: wordPitch.length && reference ? mean(wordPitch.map(semitones)) : null as number | null,
      length: Math.max(voicedS, 0.04) / count, syl_per_s: count / duration, liveliness: Math.min(1, Math.max(0, (spread - 1) / 4)) };
  });
  const known = raw.flatMap((r) => (r.pitch !== null ? [r.pitch] : []));
  const fill = known.length ? mean(known) : 0;
  const pitch = raw.map((r) => r.pitch ?? fill);
  const centres = raw.map((r) => (r.start_ms + r.end_ms) / 2000);
  const phrases: number[][] = [];
  let current = [0];
  for (let i = 1; i < raw.length; i++) {
    if (raw[i].start_ms - raw[i - 1].end_ms >= PHRASE_PAUSE_MS) { phrases.push(current); current = []; }
    current.push(i);
  }
  phrases.push(current);
  const relative = (values: number[]) => { const e = expected(values, centres, phrases); return values.map((v, i) => v - e[i]); };
  const level: Record<Key, number[]> = { energy: relative(raw.map((r) => r.energy)), pitch: relative(pitch), length: raw.map((r) => r.length) };
  const keys = Object.keys(WEIGHTS) as Key[];
  const z = Object.fromEntries(keys.map((k) => [k, zscores(level[k], STD_FLOOR[k])])) as Record<Key, number[]>;
  const medians = Object.fromEntries(keys.map((k) => [k, median(level[k])])) as Record<Key, number>;
  const prominence = raw.map((_, i) => keys.reduce((a, k) => a + WEIGHTS[k] * z[k][i], 0));
  const rawCue = raw.map((_, i) => level.energy[i] - medians.energy >= RAW_MIN.energy || level.pitch[i] - medians.pitch >= RAW_MIN.pitch
    || level.length[i] >= RAW_MIN.length * medians.length);
  const limit = Math.max(1, Math.ceil(raw.length / 4));
  const ranked = new Set(raw.map((_, i) => i).sort((a, b) => prominence[b] - prominence[a]).slice(0, limit));
  const totalSyllables = raw.reduce((a, r) => a + r.syllables, 0);
  const speakingS = raw.reduce((a, r) => a + Math.max(1, r.end_ms - r.start_ms), 0) / 1000;
  const round = (x: number, d: number) => Math.round(x * 10 ** d) / 10 ** d;
  return {
    pace_syl_per_s: round(totalSyllables / speakingS, 2),
    words: raw.map((r, i) => ({ text: r.text, start_ms: r.start_ms, end_ms: r.end_ms, prominence: round(prominence[i], 3),
      emphasized: raw.length >= 2 && ranked.has(i) && prominence[i] >= PROMINENCE_MIN && rawCue[i],
      syl_per_s: round(r.syl_per_s, 2), liveliness: round(r.liveliness, 3) })),
  };
}

/** The words of a piece of the sentence (word range [a, b)), with times from the piece's own start. */
export function sliceProsody(p: Prosody, a: number, b: number): Prosody | null {
  const words = p.words.slice(a, b);
  if (!words.length) return null;
  const t0 = words[0].start_ms;
  return { pace_syl_per_s: p.pace_syl_per_s, words: words.map((w) => ({ ...w, start_ms: w.start_ms - t0, end_ms: w.end_ms - t0 })) };
}
