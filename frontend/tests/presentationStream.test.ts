import assert from "node:assert/strict";
import { SentenceCutter, type Word } from "../presentation/liveStream";

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
// 5. Accents and punctuation differences between passes still agree (the newer spelling is kept).
{
  const c = new SentenceCutter();
  c.update(words("Ca va bien"));
  assert.deepEqual(c.update(words("Ça va, bien.")).sentences, ["Ça va, bien."]);
}
console.log("Presentation live-stream checks passed: two-pass agreement, sentence cut while speaking, seam without loss, pause flush, noise, long sentences.");
