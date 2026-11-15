# Working France-LSF lexical pose pipeline

The requested minimal path is actually running: **real lexical capture → French/gloss lookup → upstream pose concatenation → .pose + MP4 + GIF**. No training was performed. Existing training/, catalog, backend/frontend and lead source files were not edited. This is a lexical prototype and2Dskeleton diagnostic, not grammatical LSF translation, a full Animics browser port or a complete paper reproduction.

## Measured runs

Pinned upstream: [spoken-to-signed-translation fec7ac393989f4b6541ab5f860819262a54f27c7](https://github.com/sign-language-processing/spoken-to-signed-translation/tree/fec7ac393989f4b6541ab5f860819262a54f27c7), MIT, copyright2026ZurichNLP. Exact LICENSE and reused-source hashes are under integration/spoken_pipeline/. Isolated Python3.11 environment; pose-format0.15.0. No default Swiss lexicon, ASL/German model, fingerspelling backup, pose anonymizer or pix2pix model was invoked.

| Run | Actual inputs | Output | Runtime |
|---|---|---|---|
| Cleared Bonjour–Merci gloss sequence | Curated real Bonjour + real Commons Merci |137frames,25fps,512×512,H.264MP4,5.48s; GIF and .pose |2.58s|
| Explicit local-preview French `Bonjour merci` | Dicta-Sign BONJOUR cs380 raw capture + Commons Merci |108frames,25fps,512×512,H.264MP4,4.32s; GIF and .pose |2.23s|
| `Bonjour motinconnu` | Known Bonjour + unsupported token | Explicit unknown `motinconnu`, exit2, no pose/video | Blocked as intended |

Default export:440real cleared entries,440CSVrows, zero export errors,3.75s. An explicit preview snapshot exported638captured entries, including198raw-Animics entries, in12.62s, with zero errors. Portal ingestion is still running; this is NOT a claim that all1046concept pages have been captured. Re-running export refreshes every completed entry in the current registry. Pending Dicta-Sign rights are preserved, and their poses/videos remain ignored local-preview artifacts, not redistributed corpus data.

## Compatibility and observation integrity

Upstream CSVPoseLookup successfully reads the generated index.csv with spoken_language=fr and signed_language=fsl. The executed path uses actual PoseLookup.lookup_sequence, normalize_pose and join_poses, plus pose-format Pose.read/write, normalize_pose_size and PoseVisualizer.draw/save_video/save_gif. A per-selected-entry lookup preserves canonical/source decisions. Custom code only adapts the source geometry/masks, controlled French matching, source/rights gating and nearest-observation rate alignment. No low-confidence gap is converted to tracked motion by interpolation.

Raw Animics image fields PLM33, FLM468 and actual LLM/RLM21 map directly to543standard holistic points. Source visibility is retained; absent arrays get masked coordinates and confidence0. Normalized image coordinates are scaled to640pixels without a Yflip. PWLM is not mixed into image coordinates, and missing independent world-hand arrays are never invented. Old native captures map only stored measured/retargeted upper-body and finger points by a reversible affine projection; face ratios and procedural hip/neck roots do not become tracked face/body landmarks. Hand-root reuse is explicit. Coordinates remain captured estimates, not independent expert motion approval.

Exported clips retain native frame rate and source timestamps. For joining different native rates, custom nearest-frame selection repeats actual observations at25fps and records each source-frame index. Actual upstream join_poses retains NumPy masks/confidences; tests verify no newly tracked sample appears. Source frames are shoulder-normalized by upstream code and rendered through its dependency's visualizer. Default upstream smoothing, gap interpolation, duration caps and idle-hand heuristics are bypassed because they can alter observed motion/support. Thus this conservative join can still be abrupt; the lead's3Dseam work is separate.

Different canonical glosses for the same French phrase stop as ambiguous. Matching duplicate canonicals may prefer an explicitly permitted Dicta-Sign local-preview entry over Commons. All selections remain pending LSF review. Unknowns are explicit and stop generation: there is no synthetic fixture, foreign-language lookup or invented fingerspelling fallback.

Eight tests pass against real source files and outputs: source/France language gate, exact measured coordinates, absent face/hips masking, raw visibility/missing hands, preview permission/preference/ambiguity, unknown rejection, upstream CSV lookup, exact mask preservation across joining and verified video metadata. Full evidence JSON and test log are committed under integration/spoken_pipeline/evidence/. Actual .pose/MP4/GIF outputs, raw pending-rights data, checkout and environment are ignored.

## Runnable handoff

```bash
integration/spoken_pipeline/setup.sh
integration/spoken_pipeline/.venv/bin/python integration/spoken_pipeline/pipeline.py export --output integration/spoken_pipeline/runtime/lexicon
integration/spoken_pipeline/.venv/bin/python integration/spoken_pipeline/pipeline.py render --lexicon integration/spoken_pipeline/runtime/lexicon --glosses BONJOUR MERCI --output integration/spoken_pipeline/runtime/bonjour-merci
```

For captured Dicta-Sign entries add `--allow-local-preview` at export and use that lexicon. See integration/spoken_pipeline/README.md for text input, tests and the explicit preview commands. The first video is available locally at `integration/spoken_pipeline/runtime/bonjour-merci/diagnostic.mp4`; the Dicta preview at `runtime/dicta-bonjour-merci/diagnostic.mp4`. These paths coexist with the lead3Dplayer and do not change it.
