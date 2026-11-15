Run from repository root: `pip install -r backend/requirements.txt` then `uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --proxy-headers`.
Tests: `python -m pytest backend/tests` after installing requirements-dev.txt.
Configuration: SIGNACULTURE_DATA_ROOT (repository containing data/), SIGNACULTURE_DB_PATH (SQLite outside tracked files), SIGNACULTURE_CORS_ORIGINS (comma separated origins).
NLP uses Unicode token matching, longest expressions first, with optional installed spaCy French lemmatization. Lemmas in fallback are lowercase tokens, not linguistic lemmatization. Unknown words retain original spelling and duplicates. This is controlled vocabulary lookup, not validated translation.
Audio/video is decoded by FFmpeg to mono 16 kHz PCM (60 seconds maximum), then analyzed for RMS, autocorrelation F0 and pauses. `POST /analyze-audio?include_emotion=true` adds actual local emotion2vec inference when the explicitly installed runtime/model exist. Hypotheses are not ground-truth emotion. `/compose-animation` can use voiced audio regions for approximate timing; it does not align words to signs.

`POST /captures` accepts authorized video, native motion JSON or Animics keyframes. The isolated MediaPipe worker reports coverage and missing hands. Register a ready capture via `/captures/{id}/register`; runtime library entries are persisted in `.runtime/` (override with SIGNACULTURE_STORAGE). `/library` exposes captured motions and the attributed Bonjour example. `/captures/{id}/source` serves the original for review.

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
