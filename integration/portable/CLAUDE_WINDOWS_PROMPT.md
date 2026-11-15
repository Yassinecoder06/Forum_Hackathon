# Claude: restore SignaCulture on this Windows PC using the supplied bundle

Act as the engineer on this Windows computer. Get the existing app running as quickly as possible by reusing the completed work in the phone's `forum` transfer folder. Inspect the real machine and execute the necessary setup and checks; do not stop at a plan. Start with a working text/dictionary → cached 3D motion preview, then enable the other features using compatible local resources.

## What has already been supplied

Locate the copied transfer folder. It contains:
- `SignaCulture_offline_linux_x86_64.zip` — 3,858,010,143 bytes, 71,869 ZIP entries; expands to about 7.05 GB.
- SHA256 `403ec58ac78fd1e765467db7c2cb70fdfd99439c3da71c543cbb60e2d2f0ed4a` and a `.sha256` sidecar.
- `BOOTSTRAP.py`, `BUNDLE_MANIFEST.json`, `THIRD_PARTY_RESOURCES.json`, and these instructions.

The app includes 1,446 processed motion clips, corrected raw landmark arrays, source videos, the existing GLB avatar, compiled React/Three.js UI, French dictionary/data, the spaCy French model, SQLite state, emotion2vec seed weights, pinned upstream pose-processing source, and a precomputed 1,447-entry France-LSF pose lexicon. These assets are the main time saving: preserve and reuse them.

The ready-made Python environments, Python executables, FFmpeg/FFprobe and native libraries are **Linux x86-64 binaries**, not native Windows executables. Platform-neutral models, JSON, `.pose`, `.glb`, media, SQLite and frontend assets remain reusable. This Windows destination has not been tested by the original engineer; validate it here and report exact limitations.

## First inspect, then choose the fastest route

1. Locate the folder on a local Windows disk, not an MTP URI. If still only on the phone, copy the complete `forum` folder using Explorer (This PC → phone → Internal shared storage → forum). Check common locations or ask for the actual folder; do not search unrelated private folders. Use a short local path such as `C:\forum_transfer`.
2. Check Windows architecture, available Python (`py -0p` if supported), free space and WSL using PowerShell `wsl --status` and `wsl --list --verbose`. Check whether a compatible **WSL2 x86-64** Linux distro is already available. Allow at least 12 GB free before copying the archive, or 8 GB additional free after the ZIP is already present.
3. Prefer the existing WSL2 route because it can reuse the packaged runtimes directly. If WSL2 is absent, compare installing it with the native route below. Explain any administrator/reboot requirement before changing Windows features; do not block an otherwise workable native core on optional features.

## Route A — compatible WSL2 already exists

- Resolve the Windows transfer folder correctly (`wslpath` if needed), quote paths, and copy the ZIP, checksum, `BOOTSTRAP.py` and documentation into a folder under the Linux home directory. Prefer the Linux filesystem for the many environment files.
- Inside that WSL distro, run `python3 BOOTSTRAP.py --check`, then `python3 BOOTSTRAP.py --start`. The companion script verifies the ZIP before extraction, fixes executable permissions (including FFprobe's ELF loader) and invokes the included relocation helper. It was tested after ZIP extraction on Linux.
- Use the Windows browser at `http://localhost:8080`. Inspect actual connectivity if localhost forwarding differs on this machine; do not disable the Windows firewall indiscriminately.
- If an import fails because a distro system library is missing, install only the identified dependency. Preserve the existing model/landmark/pose data. A linker error does not justify rerunning extraction or downloading weights.
- Stop only processes you started; use another free port if 8080 is occupied.

## Route B — native Windows salvage

If WSL2 is unavailable or inappropriate, reuse the data and code but create fresh Windows environments. Do not run the Linux bootstrap with native Windows Python or try to convert ELF/Linux `.so` libraries into Windows packages.

1. Verify the ZIP SHA256 (PowerShell `Get-FileHash -Algorithm SHA256`). Selectively extract app source, `frontend/dist`, `data`, `.models/emotion2vec-seed`, the complete active `.runtime` data/cache and documentation. Skip Linux `runtimes/`, `.venv*` and `integration/spoken_pipeline/.venv` executable environments. Retain the pure spaCy model directory and its matching `fr_core_news_sm-*.dist-info` from the bundled `.venv/lib/python3.13/site-packages` for reuse in a native environment. Validate ZIP paths and avoid changing source data while extracting. Python's `zipfile` is suitable for selective extraction and large ZIPs.
2. Use a compatible Windows Python, preferably 3.11 x64 when available. Create a new `.venv-win` without overwriting the supplied Linux environment folders. Install `backend/requirements.txt` and compatible spaCy 3.8 native wheels. Copy the bundled **pure model data/package plus distribution metadata** into this environment's `Lib/site-packages` and test `spacy.load('fr_core_news_sm')`. Install Windows library wheels as needed; do not download the same French model again if the bundled copy loads correctly.
3. Serve the **prebuilt UI** with the API. Inspect `integration/portable/serve.py`: it mounts the API under `/api` and `frontend/dist` under `/`. Adapt only the Windows executable/PATH handling so it does not prefer bundled Linux media binaries. Start it with `.venv-win\Scripts\python.exe integration\portable\serve.py`. A Node/npm rebuild is unnecessary for launch.
4. Preserve the portable `.runtime/library.json` bytes and its relative paths where possible. `SIGNACULTURE_STORAGE`, `SIGNACULTURE_DATA_ROOT` and `SIGNACULTURE_DB_PATH` can point to this extracted app. Start with the app directory as working directory. The library hash is used to recognise the precomputed pose cache: avoid invalidating it through cosmetic rewrites. If machine-specific metadata must be rebased, do so carefully and preserve all IDs, licences, rights, timestamps and review states.
5. Bring up cached-motion playback and manual concept editing first. Missing capture/audio/pose dependencies should be reported as temporarily unavailable, not used as a reason to reprocess the corpus.
6. For new video capture, create a separate native Windows Python 3.11 environment using `experiments/motion_extraction/requirements-optional.txt`. This code uses `mediapipe.solutions`; verify that API, starting with the recorded MediaPipe 0.10.21 version rather than blindly upgrading to a Tasks-only version. Set `SIGNACULTURE_MOTION_PYTHON` to its actual `Scripts\python.exe`.
7. For audio, create a separate Windows CPU environment with compatible PyTorch/torchaudio and FunASR dependencies. Reuse `.models/emotion2vec-seed/model.pt` and its config/token files; no new model download or CUDA setup is needed merely to run CPU inference. Set `SIGNACULTURE_AUDIO_PYTHON` to the actual native interpreter. Treat `portable/requirements-audio.txt` as a source-environment inventory, checking Windows wheel compatibility rather than copying Linux binaries.
8. For the pose exporter, create a native pose environment from `integration/spoken_pipeline/requirements.lock.txt` where compatible and install the **included** upstream source at `integration/spoken_pipeline/runtime/upstream`, with no dependencies reinstalled unnecessarily. Replace any historical editable `file:///home/jmal/...` requirement with that local path. Its pinned revision is `fec7ac393989f4b6541ab5f860819262a54f27c7`. Set `SIGNACULTURE_LEXICAL_PYTHON` to the Windows interpreter. Reuse `.runtime/lexical_pipeline/lexicon`; do not regenerate it solely because the OS changed.
9. Reuse an existing native `ffmpeg.exe`/`ffprobe.exe` if available. Otherwise obtain compatible Windows binaries from a reputable primary distribution, inspect their source/licence, and install them inside the project if practical. Set native PATH and, if necessary, `SIGNACULTURE_FFMPEG` to the Windows executable. Bundled Linux FFmpeg/FFprobe cannot run natively on Windows.
10. If this is Windows ARM64, inspect compatible Python/wheel options first; the supplied x86-64 Linux runtime is not an ARM64 runtime. Reuse the platform-neutral assets and provide an honest core-first fallback.

## Constraints that matter

- Qwen was explicitly removed. Do not restore its endpoint, proposal UI, GGUF weights, server, or any replacement LLM. Manual concept editing and French lexical lookup are the intended workflow.
- Do not retrain, scrape videos, rerun the 1,446-clip corpus extraction or rebuild all poses. Those tasks have already been performed. Preserve the expensive data and models.
- Another teammate owns MetaHuman. Leave the existing avatar model/geometry unchanged.
- Keep unobserved landmarks absent/masked. The 134 reconciled clips retain genuine original detector frames when newer extraction lost observations; that is not new human approval. All linguistic review remains pending.
- Use French France-LSF (`fsl`). No Swiss/Belgian/German/ASL fallback and no claim of grammatical French-to-LSF translation. The app is a lexical preview.
- Preserve source licences/credits and publication gates. This is a private research transfer; do not upload the runtime videos/models/avatar to GitHub.
- No credentials are supplied or required for the local app. Ask only for a genuinely necessary missing path or an administrator/reboot decision; continue independent setup meanwhile.

## Completion checks

Actually launch the app. Check health, the 1,446-entry corrected/reconciled library, French lookup, Bonjour/Merci 3D playback, manual ordering, deterministic seeking and a real WebM preview export. Verify `/api/concepts/propose` is 404. Where enabled, run one existing-clip capture canary, one local audio inference, and a cached Bonjour/Merci `.pose` plus diagnostic video export. Confirm the pose cache was reused. Report which features are working, which remain unavailable, exact commands and URLs, and any Windows-only changes. Finish with a reusable launcher and `WINDOWS_RESTORE_STATUS.md`; do not call the complete Windows installation verified until you actually test it.

Primary references if needed:
- Microsoft WSL commands: https://learn.microsoft.com/en-us/windows/wsl/basic-commands
- Microsoft WSL installation: https://learn.microsoft.com/en-us/windows/wsl/install
- Recorded MediaPipe release/wheels: https://pypi.org/project/mediapipe/0.10.21/
