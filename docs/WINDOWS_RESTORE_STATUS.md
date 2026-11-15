# Windows restore status (2026-11-15)

Native Windows 11 x64, no WSL. Bundle `SignaCulture_offline_linux_x86_64.zip` verified (SHA256 `403ec58a…f0ed4a`)
and extracted selectively: app, data, `.runtime` caches, models, avatar. Linux runtimes and virtualenvs were skipped.

## Launch
Double-click `start-windows.bat` (or run it from a terminal). Then:
- Showcase site: http://127.0.0.1:8080/
- Studio (live demo): http://127.0.0.1:8080/studio/ — `?text=...` opens and plays a sentence
- API: http://127.0.0.1:8080/api/ · Diagrams: http://127.0.0.1:8080/diagrams/

## What was tested here
| Feature | Status |
|---|---|
| Library | 1,446 corrected clips load (`/api/capabilities`) |
| French lookup | spaCy `fr_core_news_sm` 3.8 reused from the bundle; "veux" → vouloir; `j'`/`qu'` elision fix |
| 3D playback, seeking, WebM export | Works (Eva avatar, café stage); five showcase videos recorded by the real export |
| Sign-core trimming + continuous bridges | New: see below. 81 backend + all frontend motion tests pass |
| Voice timing (pitch, energy, pauses) | Works (FFmpeg 8.1 + DSP) |
| Emotion hypothesis (emotion2vec) | Not enabled: needs a CPU PyTorch + FunASR environment; weights are present in `.models/` |
| Video capture | Works: `.venv-motion-win` (Python 3.10, MediaPipe 0.10.21). Canary on CHAT reproduces the Linux capture exactly |
| `.pose` export | Works: `.venv-pose-win` (Python 3.12, pinned lock, included upstream at `fec7ac3`). Bonjour/Merci: 80 frames in 17 s; lexicon cache reused |
| Qwen proposal | Removed by design: `/api/concepts/propose` returns 404 |

## Windows-only changes
- `.venv-win`, `.venv-motion-win`, `.venv-pose-win` (new native environments; Linux ones untouched/skipped).
- `PYTHONUTF8=1` (set by the launcher): the data is UTF-8, Windows Python defaults to cp1252.
- `backend/app/services/capture_library.py`: `msvcrt` lock when `fcntl` is unavailable.
- `scripts/serve-site.py` (launched by `start-windows.bat`): native ffmpeg; serves showcase `/`, Studio `/studio/` (build it first: `npm --prefix frontend run build`), `/diagrams`, API `/api`. The bundle-only `integration/portable/serve.py` got the same mounts.
- `frontend` rebuilt with `base: /studio/`; the original prebuilt UI is kept in `.runtime/frontend-dist-original-backup`.

## Changes beyond the restore (requested by the lead)
- Avatar: Ready Eva by default (`SIGNACULTURE_AVATAR=Noa` restores Noa); café theme; idle breathing/blinking.
- Showcase site in `site/` (`node e2e/build-showcase.mjs` re-records its videos from the running app).
- Diagrams in `docs/diagrams/` built from the CHAT capture (`docs/diagrams/README.md`).
- **Interpolation between signs.** Dictionary clips start and end with the hands at rest, so the old
  333 ms bridge joined two rest poses and the avatar dropped its hands between every sign (and swung
  between different signers' rest poses, e.g. hands crossed vs arms hanging). Now
  `backend/app/services/sign_core.py` plays each clip's signing core: from the first to the last
  captured frame where a wrist has covered 40 % of that sign's own peak travel from rest (idle
  drift is 3–8 cm; signs move 10–50 cm). The sentence keeps the first approach and the last
  release. `frontend/src/utils/planStitcher.ts` bridges cores with a clamped Hermite curve (arms
  keep their measured speed) while finger points move relative to their wrist. Result on the five
  showcase phrases: 42–46 % shorter; hands at rest inside sentences 47–75 % → 2–22 %. The
  tracking warning now shows only for real tracking gaps, not for bridges. Source frames are
  never modified. Only 4 of 1,446 clips (no detectable movement) still play whole.
- **Not yet integrated:** the team's learned LSFB transition model (`signlib.TransitionModel`, exp 9/15).
  Its weights are on the team Drive, not on this PC; it also needs the exact 157-point face index
  mapping used by `extract_poses.py`.

## Limits
Lexical preview only, no LSF grammar; no deaf-signer review; Dicta-Sign video and avatar
redistribution rights pending: keep the site local.
