// Wake word: the microphone listens in standby (nothing is signed) until the speaker says the word ("commencer" by
// default, or the address ?mot=...; ?mot=aucun = no standby). Checked on every partial transcription (~0.7 s), so it
// reacts before the sentence ends. Accents, capitals and punctuation are ignored; "commencer" also accepts "commencez".
// The pause word ("pause", address ?pause=..., ?pause=aucun = none) works like the standby button: active -> standby
// (the words before it are still signed), standby -> active.
export const plain = (t: string) => t.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^a-z0-9]/g, "");

export function wakeForms(word: string) {
  const w = plain(word);
  return w ? [...new Set([w, w.replace(/er$/, "ez")])] : [];
}

/** Index of the first word that is the wake word, or -1. */
export function findWake(text: string, forms: string[]) {
  return forms.length ? text.split(/\s+/).findIndex((w) => forms.includes(plain(w))) : -1;
}

/** What to sign from the sentence that woke the interpreter: only the words after the wake word. */
export function afterWake(text: string, forms: string[]) {
  const words = text.split(/\s+/), i = findWake(text, forms);
  return i < 0 ? text : words.slice(i + 1).join(" ").replace(/^[\s,.;:!?…-]+/, "");
}

/** What to sign from the sentence that paused the interpreter: only the words before the pause word. */
export function beforeWord(text: string, forms: string[]) {
  const words = text.split(/\s+/), i = findWake(text, forms);
  return i < 0 ? text : words.slice(0, i).join(" ").replace(/[\s,;:-]+$/, "");
}
