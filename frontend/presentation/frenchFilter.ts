// Filter between Whisper and the avatar: only French speech is signed.
// Whisper is already forced to French with a voice detector (server), but it still (1) invents text on music or noise,
// (2) writes low-confidence guesses, (3) writes other-language speech as it hears it. Each case is dropped here.
export type HeardWord = { text: string; probability?: number };
export type Verdict = { keep: boolean; why?: "noise" | "unsure" | "not-french" };

// Whisper's usual inventions on silence, music or noise (subtitle credits, sound labels). It joins its words with spaces
// ("Sous -titrage ST' 501"), so spaces around "-" and "." are allowed.
const INVENTED = /sous\s*-?\s*titr|amara\s*\.?\s*org|merci d.avoir regard|abonnez\s*-?\s*vous|\bST\s*'?\s*501\b|♪/i;
// A whole sentence that is only a sound label: "Musique", "(Applaudissements)", "[Rires]".
const SOUND_LABEL = /^[\s([*]*(musique|applaudissements?|rires?|bruits?|silence|toux|soupirs?|générique)[\s)\].!*…]*$/i;
const BRACKETED = /^\s*[[(][^\])]*[\])]\s*[.!]?\s*$/;

export const isNoise = (t: string) => !t.trim() || INVENTED.test(t) || SOUND_LABEL.test(t) || BRACKETED.test(t);

// Small words that tell French from English. Words that exist in both ("a", "on", "son", "but", "an") count as French.
const FR = new Set(("le la les l de des du d un une et est à au aux en que qu qui quoi je j tu il elle on nous vous ils elles ce c cet cette " +
  "ces ça pas ne n pour dans sur avec par mais ou où donc car se s sa son ses mon ma mes ton ta tes notre nos votre vos leur leurs y a ont " +
  "sont été être avoir fait très plus comme tout tous toute bien aussi bonjour merci oui non voici voilà alors parce chez").split(" "));
const EN = new Set(("the and is are was were of to you your i it it's this that what with for in my we they be have has do does don't not can " +
  "will would just so if there here hello thank thanks yes about from at how why who our their them me").split(" "));

const tokens = (t: string) => t.toLowerCase().replace(/[’]/g, "'").split(/[^\p{L}']+/u)
  .flatMap((w) => (w.includes("'") && !EN.has(w) ? w.split("'") : [w])).filter(Boolean);

/** Words of the sentence that are French / English small words. */
export function languageCounts(text: string) {
  let fr = 0, en = 0;
  for (const w of tokens(text)) { if (FR.has(w)) fr++; else if (EN.has(w)) en++; }
  return { fr, en };
}

/** Mean word confidence under this: Whisper was guessing (French speech: 0.8-0.95 on the demo narration). */
export const MIN_CONFIDENCE = 0.45;

export function checkSentence(text: string, words: HeardWord[] = []): Verdict {
  if (isNoise(text)) return { keep: false, why: "noise" };
  const p = words.map((w) => w.probability).filter((x): x is number => typeof x === "number");
  if (p.length && p.reduce((a, b) => a + b, 0) / p.length < MIN_CONFIDENCE) return { keep: false, why: "unsure" };
  const { fr, en } = languageCounts(text);
  if (en > fr && (en >= 2 || fr === 0)) return { keep: false, why: "not-french" };
  return { keep: true };
}
