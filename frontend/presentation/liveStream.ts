// Live transcription for the presentation: the microphone never stops. About every 0.7 s the audio not yet
// committed is transcribed again; a word is "sure" once two passes in a row agree on it (local agreement).
// A sentence goes to the signing model as soon as its sure text ends with . ? ! (or at a long pause, or when it
// grows too long); its audio is then cut away (word timestamps) and listening simply goes on.
import { CAPTURE_WORKLET, PhraseDetector, encodeWav } from "../src/utils/phraseDetector";

export type Word = { start_ms: number; end_ms: number; text: string };
export type LiveText = { sure: string; tentative: string };
/** A finished sentence with its words (times in the buffer at the moment it closed). */
export type Closed = { text: string; words: Word[] };

const norm = (w: string) => w.toLowerCase().normalize("NFD").replace(/[̀-ͯ]/g, "").replace(/[^\p{L}\p{N}]/gu, "");
const endsSentence = (w: string) => /[.?!…]["»)]*$/.test(w);
// Whisper's usual inventions on silence or noise (subtitle credits): never signed.
// Whisper joins its words with spaces ("Sous -titrage ST' 501"), so spaces around "-" and "." are allowed.
export const isNoise = (t: string) => !t.trim() || /sous\s*-?\s*titr|amara\s*\.?\s*org|merci d.avoir regard|abonnez\s*-?\s*vous|\bST\s*'?\s*501\b/i.test(t);

/** Pure sentence cutter over successive hypotheses of the same, growing audio buffer (times relative to it). */
export class SentenceCutter {
  private sentence: Word[] = []; // sure words of the sentence in progress
  private tail: Word[] = [];     // unsure words of the previous pass
  private committedEnd = 0;      // end (ms, in the buffer) of the last sure word
  // minCommitMs: on a very short buffer Whisper guesses (no context yet): confirm nothing before this much audio.
  constructor(private maxWords = 24, private minCommitMs = 2000) {}

  /** Words after what is already sure, without a repeated overlap at the seam. */
  private fresh(words: Word[]) {
    let out = words.filter((w) => w.start_ms >= this.committedEnd - 80 && norm(w.text));
    const done = this.sentence.slice(-3).map((w) => norm(w.text));
    for (let k = Math.min(done.length, out.length); k > 0; k--)
      if (out.slice(0, k).every((w, i) => norm(w.text) === done[done.length - k + i])) { out = out.slice(k); break; }
    return out;
  }

  /** Close sentences in the sure words; returns them and the cut point (ms) after the last closed one. */
  private close(force: boolean) {
    const sentences: string[] = [];
    const closed: Closed[] = [];
    let cut = 0;
    for (;;) {
      let end = this.sentence.findIndex((w) => endsSentence(w.text));
      if (end < 0 && this.sentence.length > this.maxWords) {
        // too long without punctuation: cut after the last comma, else everything
        const comma = this.sentence.slice(0, this.maxWords).map((w) => w.text.endsWith(",")).lastIndexOf(true);
        end = comma >= 5 ? comma : this.maxWords - 1;
      }
      if (end < 0 && force && this.sentence.length) end = this.sentence.length - 1;
      if (end < 0) break;
      const words = this.sentence.slice(0, end + 1), text = words.map((w) => w.text).join(" ");
      cut = this.sentence[end].end_ms;
      this.sentence = this.sentence.slice(end + 1);
      if (!isNoise(text)) { sentences.push(text); closed.push({ text, words }); }
    }
    if (cut) this.shift(cut);
    return { sentences, closed, cut };
  }

  private shift(ms: number) {
    const move = (w: Word) => ({ ...w, start_ms: w.start_ms - ms, end_ms: w.end_ms - ms });
    this.sentence = this.sentence.map(move);
    this.tail = this.tail.map(move);
    this.committedEnd = Math.max(0, this.committedEnd - ms);
  }

  /** One more pass over the buffer: commit what two passes agree on, close finished sentences. */
  update(words: Word[], bufferMs = Infinity) {
    const now = this.fresh(words);
    if (bufferMs < this.minCommitMs) { this.tail = now; return { sentences: [] as string[], closed: [] as Closed[], cut: 0 }; }
    let agree = 0;
    while (agree < now.length && agree < this.tail.length && norm(now[agree].text) === norm(this.tail[agree].text)) agree++;
    if (agree) {
      this.sentence.push(...now.slice(0, agree));
      this.committedEnd = now[agree - 1].end_ms;
    }
    this.tail = now.slice(agree);
    return this.close(false);
  }

  /** Last pass at a pause (or on stop): everything heard is sure; closes the sentence in progress. */
  flush(words: Word[]) {
    const now = this.fresh(words);
    this.sentence.push(...now);
    if (now.length) this.committedEnd = now[now.length - 1].end_ms;
    this.tail = [];
    const out = this.close(true);
    this.committedEnd = 0;
    return out;
  }

  get live(): LiveText {
    return { sure: this.sentence.map((w) => w.text).join(" "), tentative: this.tail.map((w) => w.text).join(" ") };
  }
}

type Options = {
  transcribe: (wav: ArrayBuffer) => Promise<Word[]>;
  /** A finished sentence, with its own audio and word times (for the voice stress). */
  onSentence: (text: string, voice: { wav: ArrayBuffer; words: Word[] }) => void;
  onLive: (live: LiveText) => void;
  onError?: (e: unknown) => void;
  intervalMs?: number;  // time between two passes
  pauseMs?: number;     // silence that closes a sentence without punctuation
  maxBufferMs?: number; // never send more audio than this
};

/** Open the microphone and stream it through Whisper. Returns the voice detector (level, speaking) and stop(). */
export async function openLiveStream(o: Options) {
  const intervalMs = o.intervalMs ?? 700, maxBufferMs = o.maxBufferMs ?? 15000;
  const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
  const context = new AudioContext();
  try {
    const url = URL.createObjectURL(new Blob([CAPTURE_WORKLET], { type: "application/javascript" }));
    await context.audioWorklet.addModule(url);
    URL.revokeObjectURL(url);
  } catch (e) {
    stream.getTracks().forEach((t) => t.stop());
    void context.close();
    throw e;
  }
  const rate = context.sampleRate, ms = (n: number) => (n * 1000) / rate;
  const cutter = new SentenceCutter();
  const chunks: Float32Array[] = [];
  let length = 0;
  let heard = false, pause = false, running = true;
  // The detector only says "speaking" and when a pause happened (its phrase end), the cutting is done on the text.
  const detector = new PhraseDetector({ sampleRate: rate, endSilenceMs: o.pauseMs ?? 900, maxPhraseMs: 1e9 }, () => { pause = true; });
  const node = new AudioWorkletNode(context, "signaculture-capture");
  const source = context.createMediaStreamSource(stream);
  node.port.onmessage = (e: MessageEvent<Float32Array>) => {
    chunks.push(e.data);
    length += e.data.length;
    detector.push(e.data);
    if (detector.speaking) heard = true;
    // nobody has spoken since the last cut: keep only a short pre-roll, so silence is never sent to Whisper
    if (!heard && ms(length) > 600) drop(length - Math.round(rate * 0.4));
  };
  source.connect(node);

  function drop(n: number) {
    n = Math.max(0, Math.min(n, length));
    while (n > 0 && chunks.length) {
      const c = chunks[0];
      if (c.length <= n) { chunks.shift(); length -= c.length; n -= c.length; }
      else { chunks[0] = c.subarray(n); length -= n; n = 0; }
    }
  }
  function snapshot() {
    const out = new Float32Array(length);
    let at = 0;
    for (const c of chunks) { out.set(c, at); at += c.length; }
    return out;
  }

  async function pass(final: boolean) {
    const audio = snapshot();
    const words = await o.transcribe(encodeWav(audio, rate));
    // each sentence gets its own stretch of audio (150 ms margin) and its word times from that stretch's start
    const send = ({ text, words: ws }: Closed) => {
      const from = Math.max(0, ws[0].start_ms - 150), to = ws[ws.length - 1].end_ms + 150;
      const clip = audio.subarray(Math.round((from * rate) / 1000), Math.min(audio.length, Math.round((to * rate) / 1000)));
      o.onSentence(text, { wav: encodeWav(clip, rate), words: ws.map((w) => ({ ...w, start_ms: w.start_ms - from, end_ms: w.end_ms - from })) });
    };
    if (final) {
      cutter.flush(words).closed.forEach(send);
      drop(audio.length); // what arrived during the request stays
      heard = detector.speaking;
    } else {
      const { closed, cut } = cutter.update(words, ms(audio.length));
      closed.forEach(send);
      if (cut) drop(Math.round((cut * rate) / 1000));
    }
    o.onLive(cutter.live);
  }

  const loop = (async () => {
    while (running) {
      const started = performance.now();
      try {
        if (pause) { pause = false; if (heard) await pass(true); }
        else if (heard && ms(length) >= 800) {
          await pass(ms(length) > maxBufferMs); // a very long buffer is closed as it is
        }
      } catch (e) {
        o.onError?.(e);
      }
      await new Promise((r) => setTimeout(r, Math.max(150, intervalMs - (performance.now() - started))));
    }
  })();

  async function stop() {
    if (!running) return;
    detector.flush();
    running = false;
    node.port.onmessage = null;
    source.disconnect();
    stream.getTracks().forEach((t) => t.stop());
    void context.close();
    await loop;
    try { if (heard && length) await pass(true); } catch (e) { o.onError?.(e); }
    o.onLive({ sure: "", tentative: "" });
  }
  return { detector, stop };
}
