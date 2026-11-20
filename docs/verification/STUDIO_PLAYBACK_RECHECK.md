# Studio playback and export recheck

The current browser smoke suite passed against a local FastAPI server and
Vite. The first attempt found that `ReadyEva.glb`, the default avatar, was
missing and its endpoint returned 503. The passing run selected the local
Noa rig explicitly using `SIGNACULTURE_AVATAR=Noa`. Its SHA-256 matched the
pinned setup manifest. This run verifies Noa playback.

- Composed and sought a sequence using two registered motion assets.
- Exported a VP9 WebM. FFprobe decoded 36 frames at 1280 × 960.
- Analyzed a three-second WAV and checked pause-aware animation timing.
- Transcribed the public greeting as “Bonjour.” with the local French ASR
  and verified that the user had to accept it before it became the draft.
- Imported a synthetic JSON motion, registered its provenance, resolved it
  through lexical lookup and removed the test library entry on cleanup.
- Checked an unknown word, the mobile layout and offline handling.

Both audio requests used the local emotion2vec and transcription capability
flags. The smoke suite checks audio timing and the transcript, with no assertion
about emotion class accuracy. The capture step exercises JSON import. It does
not measure new video landmark extraction or signing intelligibility.

Evidence files are private under `.runtime/playback-recheck/`, including
`preview.webm` and `noa-backend.log`.

With the API launched on port 8001 using the Noa environment setting, point
Vite at it with `SIGNACULTURE_API_TARGET=http://127.0.0.1:8001`. From the
repository root, run:

```bash
APP_URL=http://127.0.0.1:5174 CHROME_PATH=/opt/google/chrome/chrome \
  E2E_OUTPUT=../.runtime/playback-recheck npm --prefix e2e test
ffprobe -v error -count_frames -select_streams v:0 \
  -show_entries stream=codec_name,width,height,nb_read_frames \
  .runtime/playback-recheck/preview.webm
```

## Default avatar selection follow-up

A separate Chromium run explicitly unset `SIGNACULTURE_AVATAR`. The
browser downloaded the Noa GLB, its SHA-256 matched the pinned digest,
the rig reached the ready state, and the studio showed the Noa credit.
The older export smoke above used an explicit Noa setting. The
follow-up covers default startup and on-screen attribution only.

Receipt: `.runtime/default-avatar-recheck.J6RahO/result.json`.

## Audio panel follow-up

Chromium uploaded a 100 ms silent WAV and the public Bonjour recording to
the real local API. Silence displayed unmeasured pitch, 100% pauses and
the low-signal explanation, with the emotion toggle hidden. The greeting
returned nine emotion scores, displayed the emotion preview toggle, and
showed “Bonjour.” in the transcript panel. The first browser attempt hit an
ambiguous text selector because both the textarea and transcript contained
the greeting. Scoping the check to the transcript panel resolved it.

Receipt: `.runtime/audio-panel.8QHPwQ/receipt.json`.
The two samples validate the interface behavior and local inference path.
They do not measure speech or emotion classification accuracy.
