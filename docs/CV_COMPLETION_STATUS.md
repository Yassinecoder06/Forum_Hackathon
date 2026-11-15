# Corpus activation and remaining external requirements

The completed full CV rerun is now activated across **1,312 of 1,446 registered clips**. This includes the 36 previously activated phrase sources, 840 additional no-observation-loss candidates and 436 additional unbaselined research previews. These last cases have no older raw keypoint baseline and are not labelled accurate or approved. All candidates were checked for monotonic raw/native timestamps, finite coordinates, landmark counts, consistent observation flags and absent independent world-hand fields before activation.

**134 registered clips remain on their previous extraction** because the candidate lost observations or lacked comparable timestamps. Their exact per-source decisions and hashes are recorded in `data/source_inventory/cv_corpus_activation.json`. Compared all 1,446 registry rows before/after: gloss records, identifiers, source attribution, rights, licences and synthetic flags are unchanged. Native assets and the registry are backed up under `.runtime/cv-corpus-activation/`. Existing focused phrase fixes are preserved.

```bash
.venv/bin/python scripts/activate_cv_corpus.py --allow-unbaselined-preview        # dry-run
.venv/bin/python scripts/activate_cv_corpus.py --apply --allow-unbaselined-preview
.venv-motion/bin/python scripts/render_cv_holds.py
```

The held-source review packet contains **402 actual source-frame overlay images for all 134 held clips**, comparing previous/candidate points at shared timestamps. Open http://127.0.0.1:8770/ . No missing points were filled for this review, and generating a review packet does not approve a candidate. The active corrected count is visible through `/capabilities` and the library UI.

## Additional contact trial rejected

Tested an actual-rig candidate search to replace a rigid-capsule approximation. At 10 ms samples on the same 15 phrase inputs, mean proxy overlap improved from 0.703 to 0.536 mm. However, wrist steps above 30 mm increased in seven phrases, and phrase 3's maximum wrist step increased from 104.398 to 107.657 mm. Finger local rotation maxima and world hand/finger rotation maxima did not worsen; local wrist rotations changed to compensate for arm IK. This does not justify accepting additional positional jerks.

**The trial is rejected; runtime retains the previously verified renderer from `f3d1397`.** Full paired diagnostics are preserved in `data/source_inventory/cv_contact_rejected_trial.json`. Its 15 browser recordings remain ignored in `artifacts/phrase-review/live-cv-loop5/`; the live comparison gallery continues to show the accepted `live-cv-loop4/` result. Remaining intersections and uncertain occluded fingertips are not declared fixed. The rejected trial suggests a future trajectory-constrained solver, but that solver has not been implemented or claimed complete.

Backend checks: **46 tests pass**, including capture-specific activation gates, holding regressions, explicit unbaselined opt-in and preserving focused fixes. Frontend motion checks, lint and build pass. The actual export first-frame regression test passes. Environments, weights, captures and preview videos remain ignored local runtime data.

## Eyebrows, accuracy and linguistic review

The existing restrained brow rendering and source-specific neutral calibration remain active. Examined all 15 phrase inputs for isolated large brow-ratio reversals (neighbours within 0.025, middle differing by more than 0.12, adjacent span at most 180 ms): zero such events were found. This narrow diagnostic does not establish correct facial expression and was not a reason to invent additional smoothing or neutral labels.

There are no independent manually labelled reference keypoints for these captures, and no local LSF expert approval. Model agreement and structural checks cannot validate hidden fingertips or grammatical signing. The 402-image hold packet and existing source/phrase galleries provide review material; these human-dependent requirements remain open. No external expert has been contacted.

## MetaHuman remains blocked by the actual asset/runtime

No UnrealEditor, Blender executable, MetaHuman DNA/MHB/UAsset or exported MetaHuman character was found in the checked project, Downloads or /opt locations. The browser still renders Noa; it has not been silently relabelled.

Epic's [official download instructions](https://www.metahuman.com/download) place MetaHuman Creator/core data in Unreal Engine. The [official Fab workflow](https://dev.epicgames.com/documentation/en-us/metahuman/buying-metahumans-from-fab) acquires characters into an Unreal project. The [DCC export documentation](https://dev.epicgames.com/documentation/metahuman/metahuman-creator-export-tool-in-unreal-engine) describes exporting character data; this does not supply a ready-to-use web GLB.

Completing replacement requires a legitimately acquired/exported character with body and finger rig, facial shapes, textures and source/licence metadata, followed by conversion and rig verification in this browser runtime. No Epic account was created, credentials requested, paid asset acquired or unofficial ripped character downloaded. An actual exported asset is the concrete missing input; an untested adapter would not complete this task.
