/** Subtitle cues (SRT or WebVTT) and the placement of one generated sign motion per cue on the film timeline. */
export type Cue = { start_ms: number; end_ms: number; text: string };

const TIME = /(?:(\d{1,2}):)?(\d{1,2}):(\d{2})[,.](\d{1,3})/;

function ms(stamp: string) {
  const m = stamp.match(TIME);
  if (!m) return NaN;
  const [, h, min, s, frac] = m;
  return ((Number(h || 0) * 60 + Number(min)) * 60 + Number(s)) * 1000 + Number(frac.padEnd(3, "0"));
}

/** Parse SRT or WebVTT text. Tags (<i>, {\an8}, <c.x>) are removed, lines of one cue are joined. */
export function parseSubtitles(raw: string): Cue[] {
  const cues: Cue[] = [];
  const blocks = raw.replace(/\r/g, "").replace(new RegExp("^" + String.fromCharCode(0xfeff)), "").split(/\n{2,}/);
  for (const block of blocks) {
    const lines = block.split("\n").map((l) => l.trim()).filter(Boolean);
    const at = lines.findIndex((l) => l.includes("-->"));
    if (at < 0) continue;
    const [a, b] = lines[at].split("-->");
    const start = ms(a), end = ms(b);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) continue;
    const text = lines.slice(at + 1).join(" ").replace(/<[^>]+>/g, "").replace(/\{[^}]*\}/g, "").replace(/\s+/g, " ").trim();
    if (text) cues.push({ start_ms: start, end_ms: end, text });
  }
  return cues.sort((x, y) => x.start_ms - y.start_ms);
}

export type Placed = { start_ms: number; end_ms: number; delay_ms: number };

/** Each motion starts with its cue, or right after the previous motion when that one is still playing
 * (gap_ms of rest between signs); the delay accumulates and is reported, never hidden. */
export function placeMotions(cues: Cue[], durations: (number | undefined)[], gap_ms = 150): Placed[] {
  const out: Placed[] = [];
  let free = 0;
  cues.forEach((cue, i) => {
    const d = durations[i] ?? 0;
    const start = Math.max(cue.start_ms, free);
    out.push({ start_ms: start, end_ms: start + d, delay_ms: start - cue.start_ms });
    if (d) free = start + d + gap_ms;
  });
  return out;
}

/** Index of the motion to show at time t (the last one started; before the first: the first) and its local time. */
export function activeAt(placed: Placed[], ready: boolean[], t: number) {
  let index = -1;
  for (let i = 0; i < placed.length; i++) if (ready[i] && placed[i].start_ms <= t) index = i;
  if (index < 0) {
    const first = ready.findIndex(Boolean);
    return first < 0 ? null : { index: first, local_ms: 0 };
  }
  const p = placed[index];
  return { index, local_ms: Math.min(Math.max(0, t - p.start_ms), p.end_ms - p.start_ms) };
}
