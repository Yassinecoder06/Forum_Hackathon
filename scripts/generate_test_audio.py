#!/usr/bin/env python3
"""
Generate deterministic in-memory WAV bytes with alternating sine tones and silence.

Format:
- 16,000 Hz sample rate, 16-bit signed integer, mono
- 0.0s - 1.0s: 440 Hz sine wave (voiced tone A4)
- 1.0s - 2.0s: silence (amplitude 0.0, pause)
- 2.0s - 3.0s: 220 Hz sine wave (voiced tone A3)
Total duration: 3.0 seconds (3,000 ms)

Zero third-party dependencies: uses standard library wave, struct, math, io.
Audio is generated dynamically at test runtime; no binary WAV is committed to Git.
"""

import argparse
import io
import math
import struct
import wave


def generate_sine_silence_wav_bytes(
    sample_rate: int = 16000,
    tone1_freq: float = 440.0,
    tone1_duration: float = 1.0,
    silence_duration: float = 1.0,
    tone2_freq: float = 220.0,
    tone2_duration: float = 1.0,
    amplitude: float = 0.6,
) -> bytes:
    """Generate in-memory WAV bytes with tone, silence, and tone segments."""
    total_samples = int(sample_rate * (tone1_duration + silence_duration + tone2_duration))
    tone1_samples = int(sample_rate * tone1_duration)
    silence_samples = int(sample_rate * silence_duration)
    tone2_samples = total_samples - tone1_samples - silence_samples

    pcm_data = bytearray()
    max_amp = 32767.0 * amplitude

    # Segment 1: 440 Hz tone
    for i in range(tone1_samples):
        t = i / sample_rate
        val = int(max_amp * math.sin(2.0 * math.pi * tone1_freq * t))
        pcm_data.extend(struct.pack("<h", max(-32768, min(32767, val))))

    # Segment 2: Silence (amplitude 0)
    for _ in range(silence_samples):
        pcm_data.extend(struct.pack("<h", 0))

    # Segment 3: 220 Hz tone
    for i in range(tone2_samples):
        t = i / sample_rate
        val = int(max_amp * math.sin(2.0 * math.pi * tone2_freq * t))
        pcm_data.extend(struct.pack("<h", max(-32768, min(32767, val))))

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)

    return buf.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic sine+silence test WAV")
    parser.add_argument("--output", "-o", help="Optional target output path for writing WAV to disk")
    args = parser.parse_args()

    wav_bytes = generate_sine_silence_wav_bytes()
    if args.output:
        with open(args.output, "wb") as f:
            f.write(wav_bytes)
        print(f"[OK] Wrote {len(wav_bytes)} bytes WAV to {args.output}")
    else:
        print(f"[OK] In-memory WAV generation verified ({len(wav_bytes)} bytes, 3000 ms, 16000 Hz mono)")


if __name__ == "__main__":
    main()
