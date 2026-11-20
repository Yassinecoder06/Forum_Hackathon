# Local French speech transcription

The studio can turn uploaded French audio or video audio into an editable French text draft. Transcription and inference stay on the local PC; only public model weights were fetched through Colab.

## Model and provenance

- Model: [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small)
- Pinned revision: 536b0662742c02347bc0e980a01041f333bce120
- Stated model-card license: MIT.
- Converted Whisper Small multilingual model; French is among its supported languages.
- model.bin: 483,546,902 bytes; SHA-256: 3e305921506d8872816023e4c273e75d2419fb89b24da97b4fe7bce14170d671
- File hashes are stored in configs/asr_model.json.
- Weights live under ignored .models/faster-whisper-small/; they are not copied into the repository.

## Transfer and setup

The Google Drive remount returned HTTP 400 during authorization, so this run used the Colab SSH file-transfer channel directly. No user audio, video, or private dataset was uploaded. The transfer uses 16 MB chunks because the CLI stalled on a single 484 MB transfer. To repeat it with an active Colab session:

    colab sessions
    colab new --session signaculture-asr-model
    bash scripts/fetch-asr-model-from-colab.sh signaculture-asr-model
    uv pip install --python .venv-audio/bin/python -r backend/requirements-audio-asr.txt

The fetch script pins the revision and verifies each downloaded file against the checked-in manifest. scripts/setup-models.sh audio installs the runtime dependency and reports when the separately transferred ASR weights are absent.

## Runtime behavior

POST /analyze-audio?include_transcription=true transcribes to French using faster-whisper 1.2.1, CTranslate2, the local model and the local audio runtime. It returns an editable draft and timestamped segments alongside pitch, energy, pauses and optional emotion2vec results. CUDA FP16 is used when CTranslate2 reports a CUDA device; otherwise it uses CPU INT8. If CUDA model initialization fails, the worker retries on CPU. ASR failures preserve acoustic features and return a warning rather than a fabricated transcript. Requests are limited to 60 seconds and use a bounded subprocess timeout.

The studio offers the action “Utiliser comme brouillon français”; it never replaces typed text automatically. The user can correct the transcript before composing. Whisper output remains machine-generated and is not a validated transcript or LSF translation.

## Minimal local execution

The existing public example data/public_examples/bonjour.webm contains a short French greeting. Local RTX 4060 inference returned “Bonjour.” in one segment (0–2,000 ms), with French forced and 0.53 seconds model decode time. The latest API request completed in 3.22 seconds including worker startup, decoding and acoustic analysis. This is an execution smoke check on one short example, not a French ASR accuracy benchmark. No broad speech evaluation set or human transcript review was run.

Verification after integration: 41 backend schema/API tests passed; the frontend motion/avatar checks passed; the TypeScript/Vite production build passed; and the full Playwright smoke passed, including uploading the greeting video, displaying the draft, retaining the existing typed text until the user explicitly inserts the transcript, and then using “Bonjour.” as the text draft.

## Direct download and word timestamps (feat/voice-emotion-face)

On a machine where the Hugging Face Hub is reachable, the same pinned files can be fetched directly and checked against `configs/asr_model.json`:

```bash
.venv-audio/bin/pip install -r backend/requirements-audio-asr.txt
hf download Systran/faster-whisper-small --revision 536b0662742c02347bc0e980a01041f333bce120 config.json model.bin tokenizer.json vocabulary.txt README.md --local-dir .models/faster-whisper-small
```

The worker now requests `word_timestamps=True`; each transcript segment carries `words` with start/end times and a probability. Word timings drive the prosody analysis (emphasis, pace) used for word-aligned signing. They are model estimates and can drift by tens of milliseconds.
