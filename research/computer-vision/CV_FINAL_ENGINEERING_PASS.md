# CV and avatar engineering pass

This pass supersedes the 1,312-active / 134-held snapshot in `CV_COMPLETION_STATUS.md`. Training and its uncommitted files were left untouched.

## All registered captures have an explicit extraction path

**1,446 / 1,446 registered captures now use corrected or reconciled local research previews.** The additional 134 use a conservative whole-frame fallback: retain the complete original detector frame when a new frame loses an existing body/face/hand observation, and use corrected detector frames otherwise. The timestamp union contains actual samples only; no raw points are interpolated or hallucinated.

Across these 134 clips, **308 original detector frames are preserved**. Each fallback frame's original coordinate arrays are checked for exact equality, and every original observed field remains represented at its actual timestamp. Native poses are taken from the corresponding original/corrected extraction at that same timestamp. This resolves observation-loss activation holds by retaining existing evidence; it does **not** establish better keypoint accuracy or independently approve those original detections. Every clip remains pending review. Original captures, raw arrays, attribution, licences, rights and gloss identities are preserved. Runtime backups are under `.runtime/cv-v6-reconcile/`.

The actual API check caught required-joint metadata incorrectly including sparse fingers. Required keys now use the intersection of observed native frame keys; `tracked_joints` retains their union. No fingers were filled, and the application validator was not relaxed. A regression test covers this case.

## Hand contact without extra wrist jumps

The previously rejected unbounded actual-rig search remains rejected. Its replacement tests the actual arm IK while requiring each wrist's distance to the preceding accepted pose and next fallback pose to stay within the original trajectory's adjacent-step lengths. It preserves finger articulation and palm world orientation, keeps existing 60 mm displacement bounds and runs offline so arbitrary seeking remains deterministic.

Using identical 15 phrase inputs and 10 ms samples, the constrained pass improves **73 frames**. Mean capsule overlap falls **0.703 → 0.677 mm**. No phrase regresses in mean/max capsule overlap, maximum world hand/finger rotation step, maximum local finger rotation step, maximum wrist step or wrist steps above 30 mm. This is a small further engineering gain; residual intersections remain. Capsule scores are not mesh-accuracy or linguistic scores. Full paired results are in `data/source_inventory/cv_trajectory_contact_results.json`. These metrics use frozen inputs. In the final live recording, phrase 13 selects TRAVAIL portal 1032 instead of portal 473 after the existing coverage-based alias ranking refresh; both are the exact same source-video SHA256, but their capture derivatives differ. That selection change is recorded separately and is not counted as a renderer improvement. The other 14 sequences retain identical selected motion IDs and body/finger coordinates.

## Head-roll defect corrected

Old facial measurements used image-vertical brow/lip gaps divided by horizontal eye width, so head roll could change the apparent expression. Measurements now project gaps onto the perpendicular of the eye line in source pixels, normalize by interocular distance and retain the legacy level-face scale. Tests show invariance under image-plane rotation in a non-square image; a level face retains the previous units. Degenerate eye geometry returns missing data rather than confident neutral expression.

Applied these derived features to **all 1,446 native capture assets**, using immutable recorded face landmarks. Body/finger coordinates and hand-observation flags are checked unchanged for every updated frame. Endpoint holds without actual raw samples are labelled as playback holds and have no newly claimed observed face. Runtime backups are under `.runtime/cv-face-roll/`. The report counts 65,959 source-referenced face frames across registry entries, including repeated source aliases; this is not a unique-video count. New capture workers use the same feature function, and extractor fingerprints include it.

A real Dicta-Sign Bonjour source was freshly extracted with the updated worker as a canary: 21 actual samples, body/face coverage 100%, left hand 0%, right hand 61.9%. These hand figures are availability, not accuracy; the inactive/hidden hand is not invented. The extracted native clip contains an explicitly marked endpoint hold.

## Verification and reproduction

```bash
.venv/bin/python scripts/reconcile_cv_holds.py             # candidate generation and checks
.venv/bin/python scripts/reconcile_cv_holds.py --apply     # pending-review preview activation
.venv/bin/python scripts/apply_face_roll.py --apply
.venv/bin/python scripts/verify_live_cv.py                  # actual application asset validator
.venv/bin/python -m pytest backend/tests -q
(cd frontend && npm test && npm run lint && npm run build)
(cd e2e && CHROME_PATH=/path/to/chrome E2E_OUTPUT=../artifacts/phrase-review/live-cv-loop6 node cv-loop.mjs)
```

54 backend tests pass, including original-coordinate preservation, actual timestamp unions, matching native poses, sparse mandatory-joint metadata, head-roll invariance and missing-eye geometry. Frontend motion checks/lint/build pass. Final browser recordings use `artifacts/phrase-review/live-cv-loop6/`; earlier accepted comparison inputs remain preserved. Models, environments, raw captures and generated videos remain ignored local artifacts.

## Requirements which cannot be claimed complete

Remaining contacts, occluded fingers and grammatical/nonmanual signing require review; the fixes do not prove correct LSF. Independent manually labelled reference keypoints and local LSF expert approval are absent. None of the engineering checks is relabelled as human approval.

**MetaHuman is delegated to the user's teammate.** The user explicitly stopped this engineer's MetaHuman work. No avatar asset or avatar-selection code was changed; Noa remains unchanged. The earlier asynchronous asset-path question is superseded and does not need an answer. MetaHuman replacement is excluded from this CV pass's completion scope.
