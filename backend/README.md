Run from repository root: `pip install -r backend/requirements.txt` then `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --proxy-headers`.
Tests: `python -m pytest backend/tests` after installing requirements-dev.txt.
Configuration: SIGNACULTURE_DATA_ROOT (repository containing data/), SIGNACULTURE_DB_PATH (SQLite outside tracked files), SIGNACULTURE_CORS_ORIGINS (comma separated origins).
NLP uses Unicode token matching, longest expressions first, with optional installed spaCy French lemmatization. Lemmas in fallback are lowercase tokens, not linguistic lemmatization. Unknown words retain original spelling and duplicates. This is controlled vocabulary lookup, not validated translation.
Audio/video is decoded by FFmpeg to mono 16 kHz PCM (60 seconds maximum), then analyzed for RMS, autocorrelation F0 and pauses. `POST /analyze-audio?include_emotion=true&include_transcription=true` can add local emotion2vec and French Whisper-small transcription when their runtime/models exist. The transcript is a draft for correction, and emotion scores are hypotheses, not ground truth. `/compose-animation` can use voiced audio regions for approximate timing; it does not align words to signs. See `research/audio/ASR_MODEL_SETUP.md` for model provenance and setup.

`POST /captures` accepts authorized video, native motion JSON or Animics keyframes. The isolated MediaPipe worker reports coverage and missing hands. Register a ready capture via `/captures/{id}/register`; runtime library entries are persisted in `.runtime/` (override with SIGNACULTURE_STORAGE). `/library` exposes captured motions and the attributed Bonjour example. `/captures/{id}/source` serves the original for review. For newly processed video captures, `GET /captures/{id}/landmarks` returns gzip-compressed Animics-style landmarks. Reimport the `.json.gz` file directly through `POST /captures`. Compressed JSON is limited to 64 MiB after decompression. Missing landmarks stay absent, and the extractor provides approximate monocular depth.

The browser exports an actual WebM preview. `/render` itself returns a plan, not an encoded video. Publication requires separate linguistic/animation/rights approvals; the local review endpoint is not an authentication service. Keep it on a trusted local interface.

`/research/status` exposes completed reports; `/research/predict` invokes registered local adapters in their isolated ML environment with a timeout and single-inference limit. All outputs remain unvalidated candidates. No automatic model downloads occur at request time.

Optional French lemmatization: install spaCy and a French pipeline separately, then set SIGNACULTURE_SPACY_MODEL (default fr_core_news_sm). No automatic downloads occur. The model is cached per process; absent packages/models or pipelines without a French lemmatizer select deterministic regex fallback. X-NLP-Backend identifies the active mode. Original regex token spans stay authoritative: only exactly aligned spaCy tokens supply lemmas, and unknowns retain original text and duplicates. Restart after installing/changing models. Lemma lookup remains an experimental candidate matcher.

Reproduce the optional NLP setup from the repository root (Python 3.12):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r backend/requirements-nlp.txt
.venv/bin/python -m pip install https://github.com/explosion/spacy-models/releases/download/fr_core_news_sm-3.8.0/fr_core_news_sm-3.8.0-py3-none-any.whl
SIGNACULTURE_SPACY_MODEL=fr_core_news_sm .venv/bin/python -m uvicorn backend.app.main:app
```

The model download above is an explicit installation step. The API never downloads models at runtime. The optional integration test skips only when spaCy or the model package is absent; an installed but unusable model fails the test.

Capture registration requires an explicit `sign_language: "fsl"` alongside concept, aliases, source and license. Missing/other signing-language codes return422. Declaration is uploader provenance, not linguistic approval. Other-language captures remain previewable but cannot enter the France-LSF library. Existing legacy corpus entries retain their separate historical source audits.

`GET /research/status` includes a read-only `recent_experiments` summary from the local research catalog. Each recent entry carries an explicit signing language and representation. Measures are exposed only when the recorded result hash, metric payload and language declaration agree. Missing or changed evidence is flagged without scores. Private report/checkpoint paths and source sentences are omitted. These entries remain unavailable to `/research/predict`; registration in the ledger does not enable inference. `SIGNACULTURE_RESEARCH_CATALOG` can select a local catalog for testing/deployment; the API never creates a missing database.

The emotion2vec worker verifies loaded inference tensors against the local checkpoint before returning scores. The known missing reconstruction decoder is unused in the audited classification path and is guarded against execution. See [weight coverage and measured limits](../research/audio/EMOTION2VEC_WEIGHT_AUDIT.md). This verifies checkpoint use, not emotional accuracy or French-domain calibration. Besides the clip-level `emotion_hypothesis`, the worker scores overlapping 2 s windows (1 s hop, silent windows skipped) and returns them as `emotion_timeline`; the Studio maps them to avatar blendshapes via `frontend/src/utils/emotionFace.ts`.
