# Corrected CV live review: 15 user phrases

Executed against the actual local API, React viewer, Noa GLB skeleton and real captured source motions. This is a controlled lexical preview, not validated grammatical LSF. The exact dictionary sequences supplied by the user are preserved in `data/source_inventory/cv_phrases15_review.json`; ordinary French input was also checked for corrected source selection. Its articles/prepositions are still reported as unrepresented (for example “de”, “l”, “du”); the controlled dictionary recordings therefore must not be described as complete automatic French translations. The ordinary route chooses TRAVAILLER while the supplied exact sequence intentionally uses TRAVAIL. Both are included in the textual results.

## Activation and CV checks

36 registered source rows are now using corrected motion and matching raw landmarks: 35 clips needed by the exact dictionary sequences, plus TRAVAILLER selected by the ordinary French phrase “Mon père travaille aujourd'hui”. V5 focused outputs are preferred where available, otherwise the completed V4 extraction is used. The other registered motions have not been switched. The UI/API reports the active corrected count.

Activation is reversible with `scripts/activate_cv_phrase_preview.py`. Original registry and overwritten native assets are backed up under `.runtime/cv-live-loop/`; original source videos and original capture raw files remain intact. IDs, attribution and rights are preserved. No human or linguistic approval is granted. All 36 registered corrected raw files were verified byte-for-byte through the actual `/captures/{id}/landmarks` endpoint. That endpoint now follows the active registry instead of serving stale original keypoints.

The full extraction's structural checks and the focused source-video overlay findings remain in `CV_FULL_REEXTRACTION.md`, `CV_FOCUSED_FIXES.md` and their textual manifests. Three additional hand observations were recovered conservatively in PAIN/TRAVAIL; occluded fingers and remaining gaps in LIVRE, CHAT and CHERCHER remain uncertain. Coverage and agreement between detector passes do not measure independently labelled keypoint accuracy.

## Review and repair loop

Recorded all 15 exact sequences in the browser, with screenshots at 20/50/80% and actual attributed WebM downloads. The same corrected CV inputs were held fixed while comparing renderer versions.

The review found hand-versus-opposite-forearm intersections missing from the existing collision checks. Added those capsule pairs while excluding forearm/forearm pairs. The solver first preserves its hand-to-hand solution, then accepts the best measured full collision pose. Fingers/palm rotations and bone lengths are preserved, and wrist displacement remains bounded to 60 mm. The raw CV and sign interpolation algorithm are unchanged.

An initial forearm solver made results worse and was rejected. A subsequent 50 Hz solution improved coarse averages but worsened a dense interframe maximum; it was rejected too. The accepted cached correction runs at 100 Hz. At 14,521 samples spaced 10 ms across all 15 sequences (baseline renderer revision `0e22b395ad9de0ca9348eb0ff5544cd2e343ac8e`, input SHA256s recorded):

| Capsule diagnostic | Before | Accepted |
|---|---:|---:|
| Equal-phrase mean of per-frame maximum overlap | 6.205 mm | 0.703 mm |
| Worst overlap in the set | 47.375 mm | 27.410 mm |

The mean proxy overlap falls 88.7%; no sequence's mean or maximum regressed on this check. These are capsule approximations on the actual skeleton, not mesh penetration measurements or signing quality scores. Finger/palm/opposite-forearm radii are 8/22/28 mm. The measurement supplies no source contact allowance. Full per-case results, sample counts, worst times, activation sources and rejected trials are in `data/source_inventory/cv_live_loop_results.json`.

Video inspection also caught an export reset race: recording could start with the old paused pose before React displayed frame zero. Export now pauses, resets and waits two animation frames before recording and starting playback.

## Reproduction and artifacts

```bash
.venv/bin/python scripts/activate_cv_phrase_preview.py
.venv/bin/python -m pytest backend/tests -q
(cd frontend && npm test && npm run lint && npm run build)
(cd e2e && CHROME_PATH=/path/to/local/chrome E2E_OUTPUT=../artifacts/phrase-review/live-cv-loop4 node cv-loop.mjs)
(cd frontend && CV_SAMPLE_MS=10 AVATAR_MEASURE_ENTRY=scripts/measure-collisions.ts npm run measure:avatar -- /tmp/collision-check.json)
```

Backend: 42 tests pass, including the active corrected-raw endpoint. Frontend motion checks, lint and production build pass; the build retains its existing bundle-size warning. Actual browser paused seeking is pixel-identical and front/side views render successfully. The additional actual-browser `e2e/export-reset.mjs` check passes: the first source canvas drawn to the export equals frame zero exactly after seeking to a distinct paused pose. Media, environments and raw points remain ignored local artifacts.

Local comparison gallery: http://127.0.0.1:8766/ . Final recordings are in `artifacts/phrase-review/live-cv-loop4/`; pre-repair renderer recordings using the same corrected CV are in `live-cv-loop1/`. The raw-source diagnostic galleries remain on ports 8768/8769.

## Remaining limits

Residual hand contact/intersection remains, especially in phrases 10/11/6/3. More aggressive positional separation risks changing intended signing contact; the remaining geometry is not declared fixed. Hidden or blurred fingertips cannot be established as correct without independent reference labels or better views. Eyebrow/face motion remains an inferred retargeting preview. Noa remains the current avatar; no MetaHuman asset was installed. This result is a measured improvement ready for further visual/source review, not linguistic approval or a claim of perfect CV.
