/**
 * Who is speaking, from the subtitle file itself (film mode).
 *
 * Read in each cue, in priority order:
 * - WebVTT voice spans `<v Marie>…</v>` (also `<v.loud Marie>`), several per line allowed;
 * - name prefixes `MARIE : …`, `Marie: …`, `[MARIE] …`, `(Marie) …` (an all-caps name, or a capitalised name that is
 *   not a common French interjection such as "Attention :");
 * - a leading dash `-` / `–` / `—` = a new speaker inside one cue (French SRT convention). Unnamed dash turns get two
 *   alternating anonymous voices ("Voix 1" / "Voix 2"), so a two-person exchange still alternates sides;
 * - SDH colour `<font color="…">` when nothing else names the speaker;
 * - sound events (`[rires]`, `(musique)`) are kept apart: shown, never signed.
 * A cue with no speaker information continues the previous cue's speaker when that cue's sentence was not finished;
 * otherwise it has no speaker (narrator: the avatar signs frontally).
 *
 * The cast sheet gives each speaker a locus in signing space, in the viewer's frame (the side of the screen):
 * L (gauche), C (centre), R (droite), or N (narrateur, frontal).
 */
export type Locus = "L" | "C" | "R" | "N";
export type Turn = { text: string; speaker?: string; label?: string; sounds: string[] };
export type CastEntry = { id: string; label: string; locus: Locus; color: string };

const NOT_NAMES = new Set([
  "attention", "note", "remarque", "info", "infos", "rappel", "question", "réponse", "exemple", "résultat", "bilan",
  "conclusion", "alors", "bon", "bref", "voilà", "voici", "écoute", "écoutez", "regarde", "regardez", "oui", "non",
  "donc", "mais", "puis", "ensuite", "enfin", "important", "titre", "chapitre", "partie", "lieu", "date", "heure",
  "source", "sujet", "objet", "nb", "ps", "ok", "hein", "dis", "dites", "tiens", "tenez", "mesdames", "messieurs",
]);
const PALETTE = ["#b0563a", "#2f6f8f", "#6b8f2f", "#8f2f6b", "#8f6b2f", "#2f8f6b"];
export const ANONYMOUS = ["voix-1", "voix-2"];

const strip = (s: string) => s.replace(/<[^>]+>/g, "").replace(/\{[^}]*\}/g, "").replace(/\s+/g, " ").trim();
const fold = (s: string) => s.normalize("NFKD").replace(/[̀-ͯ]/g, "").toLowerCase();

/** Stable id of a speaker label: "MARIE", "Marie" and "marie" are the same person. */
export function speakerId(label: string) {
  return fold(label).replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "voix";
}

/** Display form of a name: "MARIE DUPONT" -> "Marie Dupont"; mixed case is kept. */
export function speakerLabel(name: string) {
  const clean = name.trim().replace(/\s+/g, " ");
  return clean === clean.toUpperCase()
    ? clean.toLowerCase().replace(/(^|[\s'’-])(\p{L})/gu, (_, sep: string, c: string) => sep + c.toUpperCase())
    : clean;
}

function isName(name: string) {
  const words = name.trim().split(/\s+/);
  if (!words.length || words.length > 3 || name.length > 28) return false;
  if (!/^\p{Lu}/u.test(name)) return false;
  if (name === name.toUpperCase() && /\p{L}{2,}/u.test(name)) return !NOT_NAMES.has(fold(words[0]));
  return words.every((w) => /^\p{Lu}[\p{L}'’.-]*$/u.test(w)) && !NOT_NAMES.has(fold(words[0]));
}

/** Leading "NOM :" / "[NOM]" / "(NOM)" on one tag-free line -> { name, rest } or null. */
export function namePrefix(line: string): { name: string; rest: string } | null {
  const bracket = line.match(/^[[(]([^\])]{1,30})[\])]\s*(.*)$/);
  if (bracket && bracket[2] && isName(bracket[1]) && bracket[1] === bracket[1].toUpperCase()) return { name: bracket[1].trim(), rest: bracket[2] };
  const colon = line.match(/^([\p{L}][\p{L}'’. -]{0,27}?)\s*:\s*(.*)$/u);
  if (colon && isName(colon[1]) && !/^\d/.test(colon[2])) return { name: colon[1].trim(), rest: colon[2] };
  return null;
}

const SOUND = /[[(]([^\])]{1,40})[\])]/g;

/** Split sound events off a tag-free line: "[rires] Oui !" -> { text: "Oui !", sounds: ["rires"] }. */
function soundsOff(line: string) {
  const sounds: string[] = [];
  const text = line.replace(SOUND, (all, inner: string) => {
    if (inner === inner.toUpperCase() && /\p{L}/u.test(inner) && isName(inner)) return all;   // [MARIE] is a name
    sounds.push(inner.trim());
    return " ";
  }).replace(/\s+/g, " ").trim();
  return { text, sounds };
}

/** The raw lines of one cue -> speaker turns (one turn when nobody is named and no dash starts a line).
 * An unnamed dash turn has speaker "" (resolved by assignVoices). */
export function cueTurns(lines: string[]): Turn[] {
  const turns: Turn[] = [];
  const last = () => turns[turns.length - 1] as Turn | undefined;
  const open = (speaker?: string, label?: string) => turns.push({ text: "", speaker, label, sounds: [] });
  const add = (line: string, before: string[] = []) => {
    const { text, sounds } = soundsOff(line);
    sounds.unshift(...before);
    if (!text && !sounds.length) return;
    if (!last()) open();
    const t = last()!;
    t.text = (t.text + " " + text).trim();
    t.sounds.push(...sounds);
  };
  for (const raw of lines) {
    // WebVTT voice spans: each <v Name> opens a turn for that name
    const voices = [...raw.matchAll(/<v(?:\.[^\s>]+)*\s+([^>]+)>/g)];
    if (voices.length) {
      add(strip(raw.slice(0, voices[0].index)));
      voices.forEach((v, k) => {
        const end = k + 1 < voices.length ? voices[k + 1].index : raw.length;
        const label = speakerLabel(v[1]);
        if (last()?.speaker !== speakerId(label)) open(speakerId(label), label);
        add(strip(raw.slice(v.index! + v[0].length, end)).replace(/^[-–—]\s*/, ""));
      });
      continue;
    }
    const color = raw.match(/<font[^>]*color\s*=\s*["']?([#\w]+)/i)?.[1];
    let line = strip(raw);
    if (!line) continue;
    const dash = /^[-–—]\s*/.test(line);
    if (dash) line = line.replace(/^[-–—]\s*/, "");
    // leading sound events ("[rires] MARIE : …") come off before the name is read
    const lead: string[] = [];
    for (let m = line.match(/^[[(]([^\])]{1,40})[\])]\s*/); m && !(m[1] === m[1].toUpperCase() && isName(m[1])); m = line.match(/^[[(]([^\])]{1,40})[\])]\s*/)) {
      lead.push(m[1].trim());
      line = line.slice(m[0].length);
    }
    const named = namePrefix(line);
    if (named) {
      const label = speakerLabel(named.name);
      open(speakerId(label), label);
      line = named.rest;
    } else if (dash) open("");
    else if (color && !last()?.speaker) {
      const id = "couleur-" + speakerId(color);
      if (last()?.speaker !== id) open(id, `Voix ${color.replace(/^#/, "")}`);
    }
    add(line, lead);
  }
  return turns.filter((t) => t.text || t.sounds.length);
}

/** Anonymous dash turns ("" speaker) -> alternating "Voix 1" / "Voix 2", across the whole file. */
export function assignVoices<T extends { speaker?: string; speakerLabel?: string }>(cues: T[], anonymous: boolean[]) {
  let last = -1;
  cues.forEach((cue, i) => {
    if (!anonymous[i]) return;
    last = (last + 1) % 2;
    cue.speaker = ANONYMOUS[last];
    cue.speakerLabel = `Voix ${last + 1}`;
  });
  return cues;
}

/** Speakers in order of first appearance, with the index of their first cue. */
export function speakersOf(cues: { speaker?: string; speakerLabel?: string }[]) {
  const seen = new Map<string, { id: string; label: string; first: number }>();
  cues.forEach((c, i) => {
    if (c.speaker && !seen.has(c.speaker)) seen.set(c.speaker, { id: c.speaker, label: c.speakerLabel || c.speaker, first: i });
  });
  return [...seen.values()];
}

/** Default cast sheet: first speaker on the left, second on the right, third centre, then left/right again;
 * an existing sheet keeps its edits (names, loci, colours). */
export function defaultCast(cues: { speaker?: string; speakerLabel?: string }[], previous: CastEntry[] = []): CastEntry[] {
  const order: Locus[] = ["L", "R", "C"];
  return speakersOf(cues).map((s, k) => previous.find((p) => p.id === s.id) ?? {
    id: s.id, label: s.label, locus: k < 3 ? order[k] : k % 2 ? "R" : "L", color: PALETTE[k % PALETTE.length],
  });
}

/** Cast entry of a speaker id, or of a name said in the text ("PAUL" from the referent table). */
export function castOf(cast: CastEntry[], speakerOrLabel?: string | null) {
  if (!speakerOrLabel) return undefined;
  const id = speakerId(speakerOrLabel);
  return cast.find((c) => c.id === speakerOrLabel || c.id === id || speakerId(c.label) === id);
}

export const LOCUS_LABEL: Record<Locus, string> = { L: "Gauche", C: "Centre", R: "Droite", N: "Narrateur (face)" };
