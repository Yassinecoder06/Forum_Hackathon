/**
 * Live mode: cut the microphone stream into spoken phrases at natural pauses (energy voice-activity detection),
 * so each phrase is transcribed and signed as a unit instead of fixed 5 s chunks that split words.
 *
 * Frames of `frameMs`; the threshold follows the background noise (it only adapts while nobody speaks).
 * A phrase starts after 2 loud frames (with `preRollMs` of audio before it, so first syllables are kept) and
 * ends after `endSilenceMs` of quiet; very long speech is cut every `maxPhraseMs` so the avatar never waits.
 */
export type DetectorOptions = {
  sampleRate: number;
  frameMs?: number;
  minLevel?: number;
  startFactor?: number;
  endSilenceMs?: number;
  minSpeechMs?: number;
  maxPhraseMs?: number;
  preRollMs?: number;
};

export class PhraseDetector {
  readonly rate: number;
  private frame: number;
  private opts: Required<DetectorOptions>;
  private pending: number[] = [];
  private preRoll: Float32Array[] = [];
  private phrase: Float32Array[] = [];
  private noise = 0;
  private frames = 0;
  private loud = 0;
  private quiet = 0;
  private voiced = 0;
  speaking = false;
  level = 0;

  constructor(options: DetectorOptions, private onPhrase: (samples: Float32Array) => void) {
    this.opts = { frameMs: 20, minLevel: 0.012, startFactor: 3, endSilenceMs: 600, minSpeechMs: 300, maxPhraseMs: 8000, preRollMs: 300, ...options };
    this.rate = options.sampleRate;
    this.frame = Math.round((this.rate * this.opts.frameMs) / 1000);
  }

  private get threshold() {
    return Math.max(this.opts.minLevel, this.noise * this.opts.startFactor);
  }

  push(chunk: Float32Array) {
    for (let i = 0; i < chunk.length; i++) this.pending.push(chunk[i]);
    while (this.pending.length >= this.frame) this.step(Float32Array.from(this.pending.splice(0, this.frame)));
  }

  private step(f: Float32Array) {
    let sum = 0;
    for (let i = 0; i < f.length; i++) sum += f[i] * f[i];
    const rms = Math.sqrt(sum / f.length);
    this.level = rms;
    this.frames++;
    const { frameMs, preRollMs, endSilenceMs, maxPhraseMs } = this.opts;
    if (!this.speaking) {
      // background level: quick at first, then slow, and never from speech
      this.noise = this.frames <= 10 ? Math.max(this.noise, rms) : this.noise * 0.98 + Math.min(rms, this.threshold) * 0.02;
      this.preRoll.push(f);
      while (this.preRoll.length * frameMs > preRollMs) this.preRoll.shift();
      this.loud = rms > this.threshold && this.frames > 10 ? this.loud + 1 : 0;
      if (this.loud >= 2) {
        this.speaking = true;
        this.phrase = [...this.preRoll];
        this.preRoll = [];
        this.quiet = 0;
        this.voiced = this.loud;
      }
      return;
    }
    this.phrase.push(f);
    if (rms > this.threshold * 0.7) {
      this.quiet = 0;
      this.voiced++;
    } else this.quiet++;
    if (this.quiet * frameMs >= endSilenceMs) this.finish(Math.max(0, this.quiet - Math.round(150 / frameMs)));
    else if (this.phrase.length * frameMs >= maxPhraseMs) this.finish(0, true);
  }

  private finish(trimFrames: number, keepSpeaking = false) {
    const frames = trimFrames ? this.phrase.slice(0, this.phrase.length - trimFrames) : this.phrase;
    if (this.voiced * this.opts.frameMs >= this.opts.minSpeechMs) {
      const out = new Float32Array(frames.reduce((n, f) => n + f.length, 0));
      let o = 0;
      for (const f of frames) { out.set(f, o); o += f.length; }
      this.onPhrase(out);
    }
    this.phrase = [];
    this.voiced = 0;
    this.quiet = 0;
    this.speaking = keepSpeaking;
  }

  /** Emit the phrase in progress (stopping the microphone mid-sentence). */
  flush() {
    if (this.speaking) this.finish(0);
  }
}

/** Mono float samples -> 16 kHz 16-bit PCM WAV (what the live transcription endpoint reads directly). */
export function encodeWav(samples: Float32Array, fromRate: number, toRate = 16000): ArrayBuffer {
  const n = Math.max(1, Math.floor((samples.length * toRate) / fromRate));
  const out = new DataView(new ArrayBuffer(44 + n * 2));
  const text = (o: number, s: string) => { for (let i = 0; i < s.length; i++) out.setUint8(o + i, s.charCodeAt(i)); };
  text(0, "RIFF"); out.setUint32(4, 36 + n * 2, true); text(8, "WAVE"); text(12, "fmt ");
  out.setUint32(16, 16, true); out.setUint16(20, 1, true); out.setUint16(22, 1, true);
  out.setUint32(24, toRate, true); out.setUint32(28, toRate * 2, true); out.setUint16(32, 2, true); out.setUint16(34, 16, true);
  text(36, "data"); out.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) {
    const x = (i * fromRate) / toRate, k = Math.floor(x), t = x - k;
    const v = (samples[k] ?? 0) * (1 - t) + (samples[Math.min(k + 1, samples.length - 1)] ?? 0) * t;
    out.setInt16(44 + i * 2, Math.max(-1, Math.min(1, v)) * 0x7fff, true);
  }
  return out.buffer;
}
