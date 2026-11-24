# Live LSF interpreter

A web page that listens to a French speaker and has the MetaHuman avatar sign what is said, live.
Two ways to show it:

| Page | What it shows |
|---|---|
| `/transcription/` | Whisper alone: what the interpreter hears (confirmed words, current guess, low-confidence words, sentences kept or dropped by the French filter). |
| `/interprete/` | The interpreter alone, filling the window: avatar, subtitle strip, mic button. Made to sit next to (or inside) a slide deck. |
| `/presentation/` | 13 French slides about SignaCulture with the same interpreter in the bottom-right corner. |

Code: `frontend/interprete/`, `frontend/presentation/` (`interpreter.tsx` is the box used by both pages).
It uses the same services as the Studio (one product): `/live/transcribe`, `/analyze-audio`, `/lsf/generate`.

## Chain

```
mic ──► live transcription (Whisper large-v3-turbo, GPU, every ~0.7 s) ──► sentence cut
    ──► French filter ──► voice analysis (stress + emotion2vec) ──► C0 + refiners (/lsf/generate)
    ──► avatar (sliced motion preparation, cross-fade to rest between sentences)
```

- **Live transcription** (`presentation/liveStream.ts`). The mic never stops. About every 0.7 s the audio not yet
  committed is transcribed again. A word is sure once two passes agree (nothing is confirmed under 2 s of audio).
  A sentence is sent as soon as its sure text ends with `. ? !`, at a 0.9 s pause, or after 24 words (cut at a comma
  if possible). Its audio is cut at the last word's end time and listening goes on.
- **French filter** (`presentation/frenchFilter.ts`). Whisper is forced to French with a voice detector on the server,
  but some things still get through. A sentence is not signed if it is:
  - a Whisper invention on noise or music (subtitle credits, "Musique", "(Applaudissements)", "[Rires]", ♪);
  - an unsure guess (mean word confidence < 0.45; French narration scores 0.8–0.95);
  - English (more English small words than French ones).
- **Voice analysis** (one `/analyze-audio?include_emotion=true` call per sentence, 0.1–0.4 s):
  - Stressed words come from the acoustic frames and the stream's word times. `presentation/prosody.ts` is a
    line-by-line port of the server's `prosody.analyse_prosody`.
  - emotion2vec gives scores for the sentence plus ~2 s windows. They change the speed and size of the signs on the
    server and the face and posture on the avatar. No emotion under 1.5 s of speech (unreliable on short clips).
  - These are model hypotheses about the voice, not a measure of the speaker's state.
- **Signing** (`/lsf/generate`, engine C0 + finger/face refiners).
  - Long sentences are split into chunks of ≤ 16 words (C0 writes at most 96 tokens). A chunk that hits the limit is
    cut in two. A piece that hits the limit is
    cut in two.
  - Each request carries the stressed words, the emotion, and the referent table of the previous sentences, so a
    pronoun ("Elle…") points to who was named before ("Marie…"). The table is forgotten after 90 s of silence.
- **Avatar.** Motions are prepared a few ms per frame ahead of time, so there is no stall at a sentence start.
  Signing runs at ×1.5. Between sentences and chunks there is about 0.1 s, with a short fade (0.3 s) toward rest. The arms are kept out of the torso (torso measured on the avatar's own meshes).

## Controls

| | |
|---|---|
| `M` / button | Mic on / off |
| `S` | Stop everything at once (`/interprete/`): the mic, the signing in progress and the phrases waiting; the avatar goes back to rest. `M` listens again. |
| `F` | Full screen (`/interprete/`) |
| `S`, `←` `→`, `Z`, `H` | Slides page: `S` signs the slide's sentence; change slide, box size, hide the box |

Address options:

| Option | Effect |
|---|---|
| `?vitesse=1.2` | Signing speed, 0.5–3 (default 1.5) |
| `?ecoute=pauses` | Earlier listening mode: one transcription per phrase, cut at pauses |

## Running it

The page is a Vite page of the frontend; `/api` must reach the SignaCulture API (Whisper worker, C0 + refiners,
emotion2vec loaded).

```bash
cd frontend
SIGNACULTURE_API_TARGET=http://<api-host>:8040 npx vite --host 0.0.0.0 --port 5202
# open http://localhost:5202/interprete/
```

- **Microphone and https.** Browsers give the microphone only on `localhost` or https. For other devices on the LAN,
  run a second Vite with a certificate (`SIGNACULTURE_DEV_CERT`, `SIGNACULTURE_DEV_KEY`) and open `https://<host>:<port>/interprete/`.
- **PowerPoint (WebView2 add-ins).**
  - The add-in has no "continue anyway" for a self-signed certificate: the presenting PC must trust it. Run this in
    PowerShell, as the normal user:
    `Import-Certificate -FilePath lan.crt -CertStoreLocation Cert:\CurrentUser\Root`
  - The add-in's frame must also allow the microphone. If it does not, keep the interpreter in its own window on top
    of the slides.
- The page sends no `X-Frame-Options` / `frame-ancestors`, so it can be embedded.

## Measured (automatic proxies, not validated by an LSF reader)

- Text → signing start: about 0.7 s (C0 with CUDA graphs: 0.06–0.08 s per sentence on the server).
- 30 s of continuous narration as a fake mic: sentences sent while the speaker kept talking, no word lost or doubled
  at the cuts. Last word → sent to signing: 1.4–4.1 s (median 2.3 s).
- Same 15 s of speech, signed until the end:
  - normal speed: 4 sentences in 45 s, still behind;
  - ×1.25 + catch-up: 6 sentences, all done by 45 s, 4–8 s each;
  - ×2: 6 sentences, all done by 33 s, 3–5 s each;
- Current setting (full sentences in chunks, ×1.5, 0.1 s gaps): every sentence signed, 0.0–0.1 s between them,
  everything signed by 39.9 s.
  - worst frame 34 ms.
- Arms inside the torso, on 13 test sentences, before → after the torso guard:
  - before: 1–89 % of frames (up to 18 cm);
  - after: 0 % on 12 of the 13 sentences, 0.6 % and 7.5 % (max 3.5 cm) on the two hardest.
- Pointing: "Marie présente le projet." then "Elle est très fière de son équipe !" → "Elle" points to Marie's locus.

## Limits

- The avatar signs more slowly than people speak. On long continuous speech it falls behind; sentences wait in order
  and none is dropped.
- The English filter is checked with unit tests only (no English recording was available).

## Tests

`frontend/tests/presentationStream.test.ts` checks:
- sentence cutting (two-pass agreement, seams, pauses, noise, long sentences);
- the French filter;
- (the wake-word helpers in `presentation/wakeWord.ts`, not used by the page now).

`frontend/tests/motion.test.ts` checks that the sliced motion preparation gives exactly the one-shot result.
