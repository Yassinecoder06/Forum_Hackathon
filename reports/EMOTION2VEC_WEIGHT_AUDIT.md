# emotion2vec weight coverage and decoder audit

Completed 2026-11-17 using the existing local `emotion2vec/emotion2vec_plus_seed` artifact, revision `7be9a30d52bdc793cc24e6f7d25358d6648927c1`, FunASR 1.4.26 and the existing eight-second French demo narration. No new downloads or training.

## Finding

The earlier warning about ten missing `modality_encoders.AUDIO.decoder` tensors is reproducible. Inspection of the installed FunASR implementation shows utterance classification uses `extract_features(..., features_only=True)` followed by pooling and `proj`, without invoking the modality reconstruction decoder. The checkpoint contains no decoder weights.

The library prints `All keys matched successfully` after copying available checkpoint weights into a state dictionary already containing random initialization. That message alone cannot establish checkpoint coverage. We therefore independently compared actual loaded tensors against the local checkpoint.

**185 tensors / 93,178,133 values match exactly**, including both classification-head tensors. The only missing entries are the ten narrowly enumerated reconstruction-decoder weights/biases. A forward hook observed zero decoder calls during the actual audio inference. Filling its parameters with NaNs produced exactly identical nine-category scores on the same audio. This establishes that the missing decoder did not affect this exercised inference path, not that every possible future configuration or input has been tested.

Checkpoint SHA256: `327e861049c7105a3f8589edfd16543a7cdc0c1926f1b64de975d98e0650cdd3`, matching the cached download metadata. Installed implementation hashes, input hash and complete measured results are recorded in `reports/EMOTION2VEC_WEIGHT_AUDIT.json`.

## Applied change

The isolated production worker now verifies exact checkpoint equality for inference parameters before generating scores. Missing or mismatched inference weights and ambiguous scope mappings fail explicitly. Only the audited decoder entries are exempted; a runtime hook rejects any attempt to execute that untrained decoder. Warnings are not suppressed, and third-party installed source files were not modified.

Score parsing also rejects mismatched label/score counts, duplicate or unexpected categories, nonfinite/out-of-range numbers and invalid probability sums. Worker output includes weight-coverage evidence. The existing service continues to expose categories as model hypotheses and preserve acoustic results if the model subprocess fails.

## Actually executed

- Five counterfeit-checkpoint/score tests passed: exact mapping, missing head, wrongly loaded values, ambiguous mapping and invalid score distributions.
- 24 existing backend API/capture regression tests passed. The pre-existing Starlette/httpx deprecation warning remains.
- Revised worker ran real CPU inference, including the strict check and unused-decoder guard. Nine normalized scores exactly match the prior real-model run, including neutral 0.992552042.
- Actual audio service invocation completed with 8,000 ms duration, 200 acoustic windows and the same nine model scores.
- Decoder-hook and NaN-perturbation experiments passed on the real loaded model. NaN changes were confined to the audit process and were never saved to the checkpoint or production worker.

Private outputs: `reports/private/emotion2vec-verified-worker.json` and `reports/private/emotion-verified-service.json`. Logs: `reports/emotion2vec_weight_audit.log`, `reports/emotion_verified_worker.log`, `reports/emotion_weight_tests.txt`, `reports/emotion_backend_regression.txt`.

## Reproduce

```bash
.venv-audio/bin/python scripts/audit_emotion_weights.py
.venv-audio/bin/python backend/app/workers/emotion.py .runtime/pipeline-recheck-voice.wav .models/emotion2vec-seed reports/private/emotion2vec-verified-worker.json
.runtime/t5-local-venv/bin/python -m pytest backend/tests/test_emotion_weights.py -q
.venv/bin/python -m pytest backend/tests/test_api.py backend/tests/test_capture_pipeline.py -q
```

The weight tests require a PyTorch environment and skip in a lightweight backend environment without Torch. Model inference remains in `.venv-audio`; it does not add Torch to the API environment. The input is the existing demo excerpt described in `PIPELINE_AND_MULTISIGN_RECHECK.md`.

## Limits and next requirement

This resolves the specific weight-loading warning for the tested inference configuration. It does not measure emotional accuracy, calibration or French-domain generalization. A confident neutral output is not an emotion ground truth. Labeled, appropriately licensed French speech and an evaluation protocol are needed for that claim. Category/style controls must not overwrite grammatical signing facial expressions.

The local model card declares `other` / `model-license`; no independent commercial-use permission conclusion follows from the software library's license. No model or source audio is redistributed here. This remains a local experimental feature, not an inference about a person's inner state. Translation adequacy, qualified sign-language review and validated calibrated 3D motion remain outstanding in the overall pipeline.
