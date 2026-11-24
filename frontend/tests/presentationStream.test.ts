import assert from "node:assert/strict";
import { SentenceCutter, type Word } from "../presentation/liveStream";
import { checkSentence } from "../presentation/frenchFilter";
import { afterWake, beforeWord, findWake, wakeForms } from "../presentation/wakeWord";

// words with 300 ms each, from a start time (ms in the current buffer)
const words = (text: string, from = 0): Word[] => text.split(" ").map((t, i) => ({ text: t, start_ms: from + i * 300, end_ms: from + i * 300 + 280 }));

// 1. A word is sure only when two passes agree; a sentence closes on its final punctuation, mid-speech.
{
  const c = new SentenceCutter();
  assert.deepEqual(c.update(words("Bonjour à")).sentences, []);
  assert.equal(c.live.sure, "");
  assert.equal(c.live.tentative, "Bonjour à");
  let r = c.update(words("Bonjour à tous. Nous"));
  assert.deepEqual(r.sentences, []);
  assert.equal(c.live.sure, "Bonjour à");
  r = c.update(words("Bonjour à tous. Nous présentons"));
  assert.deepEqual(r.sentences, ["Bonjour à tous."], "sentence sent while still speaking");
  assert.equal(r.cut, 880, "audio cut right after « tous. »");
  assert.equal(c.live.sure, "Nous", "the next sentence keeps going");
  // the caller dropped 880 ms: the next pass has times relative to the new buffer start
  r = c.update(words("Nous présentons notre projet.", 20));
  assert.deepEqual(r.sentences, []);
  r = c.update(words("Nous présentons notre projet.", 20));
  assert.deepEqual(r.sentences, ["Nous présentons notre projet."], "no word lost or repeated at the seam");
}
// 1b. Nothing is confirmed on a very short buffer (Whisper has no context yet).
{
  const c = new SentenceCutter();
  c.update(words("S'il y a"), 900);
  c.update(words("S'il y a"), 1600);
  assert.equal(c.live.sure, "", "no word sure under 2 s of audio");
  c.update(words("Signaculture aide les"), 2400);
  c.update(words("Signaculture aide les créateurs"), 3100);
  assert.equal(c.live.sure, "Signaculture aide les");
}
// 2. A pause closes a sentence without punctuation; everything heard is kept.
{
  const c = new SentenceCutter();
  c.update(words("merci pour votre"));
  const r = c.flush(words("merci pour votre attention"));
  assert.deepEqual(r.sentences, ["merci pour votre attention"]);
  assert.equal(c.live.sure + c.live.tentative, "");
}
// 3. Whisper's inventions on noise are never signed.
{
  const c = new SentenceCutter();
  assert.deepEqual(c.flush(words("Sous-titrage ST' 501")).sentences, []);
  assert.deepEqual(c.flush(words("Sous -titrage ST' 501")).sentences, [], "also when Whisper splits the words");
  assert.deepEqual(c.flush(words("Abonnez -vous")).sentences, []);
  assert.deepEqual(c.flush(words("Nous sous-traitons le titre.")).sentences, ["Nous sous-traitons le titre."], "real speech is kept");
}
// 4. A very long sentence without punctuation is cut (after a comma when there is one).
{
  const c = new SentenceCutter(10);
  const long = "un deux trois quatre cinq six, sept huit neuf dix onze douze treize";
  c.update(words(long));
  const r = c.update(words(long));
  assert.deepEqual(r.sentences, ["un deux trois quatre cinq six,"]);
  assert.equal(c.live.sure, "sept huit neuf dix onze douze treize");
}
// 4a. Short phrases: closed at a comma once they have 5 words, not before.
{
  const c = new SentenceCutter(12, 2000, 5);
  const w = words("Bonjour, nous présentons notre projet, un avatar qui signe");
  c.update(w);
  assert.deepEqual(c.update(w).sentences, ["Bonjour, nous présentons notre projet,"]);
  assert.equal(c.live.sure, "un avatar qui signe");
}
// 4b. French spacing: a lone "!" from Whisper still ends the sentence.
{
  const c = new SentenceCutter();
  const w = words("Elle est très fière de son équipe !");
  c.update(w);
  assert.deepEqual(c.update(w).sentences, ["Elle est très fière de son équipe!"]);
}
// 5. Accents and punctuation differences between passes still agree (the newer spelling is kept).
{
  const c = new SentenceCutter();
  c.update(words("Ca va bien"));
  assert.deepEqual(c.update(words("Ça va, bien.")).sentences, ["Ça va, bien."]);
}
// 6. French filter: sound labels, unsure guesses and English are not signed; French (with loanwords) is.
{
  const why = (t: string, p?: number) => checkSentence(t, p === undefined ? [] : t.split(" ").map((x) => ({ text: x, probability: p }))).why;
  assert.equal(why("Musique"), "noise");
  assert.equal(why("(Applaudissements)"), "noise");
  assert.equal(why("[Rires]"), "noise");
  assert.equal(why("♪ La la la ♪"), "noise");
  assert.equal(why("Bonjour à tous, voici notre projet.", 0.3), "unsure");
  assert.equal(why("Thank you so much for coming tonight."), "not-french");
  assert.equal(why("What is this?"), "not-french");
  assert.equal(why("Hello"), "not-french");
  for (const ok of ["Bonjour à tous, voici notre projet.", "Le but du projet est simple.", "Merci.", "On utilise le design thinking.",
    "C'est un prototype open source.", "Marie présente le projet.", "SignaCulture traduit le français parlé."])
    assert.equal(checkSentence(ok, ok.split(" ").map((x) => ({ text: x, probability: 0.9 }))).keep, true, ok);
  const c = new SentenceCutter();
  assert.deepEqual(c.flush(words("Thank you very much.")).sentences, [], "English never reaches the avatar");
  assert.deepEqual(c.flush(words("Merci beaucoup.")).sentences, ["Merci beaucoup."]);
}
// 7. Wake word: accents, capitals, punctuation ignored; only what follows the word is signed.
{
  const f = wakeForms("commencer");
  assert.equal(findWake("Bon, on va Commencer !", f), 3);
  assert.equal(findWake("Commencez.", f), 0);
  assert.equal(findWake("Nous commençons le projet.", f), -1);
  assert.equal(afterWake("Commencer.", f), "");
  assert.equal(afterWake("On va commencer, bonjour à tous.", f), "bonjour à tous.");
  assert.deepEqual(wakeForms(""), []);
  const p = wakeForms("pause");
  assert.equal(beforeWord("Merci à tous, pause.", p), "Merci à tous");
  assert.equal(beforeWord("Pause !", p), "");
  assert.equal(findWake("On fait une PAUSE", p), 3);
}
console.log("Presentation live-stream checks passed: short phrases, wake and pause words, French filter (noise, unsure, English), two-pass agreement, sentence cut while speaking, seam without loss, pause flush, noise, long sentences.");
