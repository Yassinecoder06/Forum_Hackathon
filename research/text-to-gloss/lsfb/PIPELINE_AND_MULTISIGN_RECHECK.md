# Pipeline and multi-sign diagnostic verification

Previous goal turn classified as progress:matched context ablation produced new authoritative developmental results and backed up the model. This continuation verifies additional required pipeline components and tests multi-sign learnability.

## New multi-sign evidence

Train-only memorization gate warm-started current-only T5,on49multi-sign examples,seed91,LR3e-4,no dropout/weight decay. At100updates48/49exact97.96%,WER0.83%,zero unfinished predictions. Then a separately frozen developmental evaluation yielded4/23exact17.39%,WER143.33%,0/6multi-sign exact,107.69%multi-signWER. Generalization failed despite training fit. This diagnostic is not promoted;the12/23current-only developmental checkpoint remains retained. The gate changes optimization and training distribution together;it is not a pure rebalancing ablation. No validation labels opened during gate selection,no official test opened.

## Pipeline execution

- Frontend production build and motion/avatar checks passed (fixed bones,endpoints,pauses,seeking,hand contacts,brows).
-22backend API/capture tests passed,including Animics position import and finger/side preservation checks.
- Browser smoke on real API/UI passed composition,seeking,WebMdownload,actual audio/pause timing,capture JSON/provenance/persistence,lookup,mobile and offline/error cases. Used locally installed Chrome after bundled Playwright binary was absent.
- Export independently ffprobe-checked:341,196bytes,3.654063seconds. Private artifact:research/private/pipeline-browser-recheck/preview.webm.
- Real source video data/public_examples/bonjour.webm reprocessed with localMediaPipeworker:62output frames,61sampledframes,100%pose/face coverage,left-hand100%,right-hand52.5%,no missing body frames. Output passes validate_motion. Partial right-hand detection is retained and disclosed;software execution is not linguistic validation. MediaPipe inferred depth is approximate,not calibrated mocap.
- Real local emotion2vec+seed CPU inference completed on8seconds of existing French demo narration;9finite category scores,neutral0.99255. Decoder-weight warnings are retained in the log;no exact weight-coverage or emotion-accuracy claim. Scores are model hypotheses,not human emotional ground truth.

## Reproduction

```bash
.runtime/t5-local-venv/bin/python -m training.lsfb_cont.multisign_fit_gate
.runtime/t5-local-venv/bin/python -m training.lsfb_cont.evaluate_multisign_fit
npm --prefix frontend run build
npm --prefix frontend test
.venv/bin/python -m pytest backend/tests/test_capture_pipeline.py backend/tests/test_api.py -q
.venv-motion/bin/python backend/app/workers/capture.py data/public_examples/bonjour.webm research/private/bonjour-capture-recheck/motion.json research/private/bonjour-capture-recheck/progress.json
.venv-audio/bin/python backend/app/workers/emotion.py .runtime/pipeline-recheck-voice.wav .models/emotion2vec-seed research/private/emotion2vec-recheck.json
```

Training commands refuse existing outputs. Audio WAV was extracted with ffmpeg16kHzmono,first8seconds from artifacts/SignaCulture_Demo_FR.mp4. Browser test requires API/UI running;APP_URL=http://127.0.0.1:15173,APIproxy target18000,CHROME_PATH=/home/jmal/.local/bin/google-chrome,E2E_OUTPUT=../reports/private/pipeline-browser-recheck.

## Still incomplete in the overall objective

No human-reviewed/generalizing text-to-gloss translator;no approved Belgian-ID-to-France-LSFmotion map;no qualified signer comprehension study;restricted Rosetta/continuousDicta-Sign access unresolved. Current mT5-base transfer remains incomplete/unverified,so no result from that model claimed. SignVerse2Dmotion is not integrated as true3D;its body-score semantics and embedded language labels are unresolved. Existing avatar and exports are a working experimental studio,not a verified general signing translator. The goal remains active.

The next useful research step is an expanded,reviewed multi-sign supervision set with fresh conversations;alternative loss/model complexity alone is not supported by the fit/generalization gap. Preserve source/annotation conventions and do not turn subtitles into gold glosses.

## Current frontend and browser rerun (2026-11-17)

Against the current research-panel API at 127.0.0.1:18004 and UI at :15177, a fresh frontend production build, frontend motion/avatar tests and full browser smoke run passed. The flow exercised WebGL composition/seeking, 340,012-byte WebM export at 3.664058 seconds (ffprobe verified), generated WAV pitch/energy/pause analysis, pause-aware animation timing, JSON fixture capture with explicit fsl declaration, registration/provenance, lookup persistence, unknown input, mobile layout and offline error messaging. The temporary smoke library entry was deleted (zero TestPipeline entries remain). Browser screenshots/preview retained privately at `research/private/full-pipeline-smoke-20261117/`; logs: `docs/verification/logs/full_pipeline_build.log`, `docs/verification/logs/full_pipeline_frontend_test.log`, `docs/verification/logs/full_pipeline_browser_test.log`.

The capture portion uses a procedural synthetic JSON motion fixture; it verifies UI/API behavior, not fresh computer-vision extraction or natural signing. Actual source-video extraction is separately evidenced by `BONJOUR_CAPTURE_RECHECK.json`, with its 52.5% right-hand coverage disclosed. Export integrity and usability do not establish signing intelligibility or linguistic correctness.
