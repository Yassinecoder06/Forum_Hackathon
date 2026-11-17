# Homelab playback and showcase correction

Implemented directly on homelab, based on `9e880dc`, on `feat/showcase-motion-quality`. Local commit only; no push. Eva avatar asset is unchanged.

The homepage referenced five absent WebM files. Generate the actual Studio recordings with `APP_URL=http://127.0.0.1:8320 CHROME_PATH=/path/to/chrome node e2e/build-showcase.mjs` (Playwright, Chromium and ffmpeg required). The builder produces WebM, H.264 MP4 and a genuine frame poster, and publishes metadata only after all recordings succeed. The page uses native controls, posters, an explicit error fallback and respects reduced motion. Generated assets remain ignored because their underlying source rights/review status is preserved.

## Source-preserving library correction

`python scripts/audit_library_motion.py` audits every registered real clip and creates an SVG contact sheet for every sign at captured core start/middle/end. Results: 1,446 clips, 324 with flags, 458 isolated multi-segment finger-length bursts. This is automated engineering triage, not someone watching or approving all 1,446 full videos. No isolated arm reversals met the stringent source-coordinate threshold. A fist is not treated as an error.

`python scripts/apply_motion_playback_quality.py` adds reversible playback masks and core-window metadata, keeps backups and hashes under ignored `.runtime/motion-quality`, and verifies every original joint coordinate and original hand-tracking annotation remains equal. Raw capture exports are untouched. The renderer treats flagged hand samples as gaps, using the existing explicitly interpolated/held/untracked playback states. It does not claim those samples were tracked. Do not reclassify linguistic/source rights approval.

Single-sign library previews now use their cached core window. Text composition applies core windows to single signs too. First/last contextual lead and release are shortened from 400/600 ms to 120/180 ms. This reduces repeated setup/rest without deleting source videos or blanket-removing closed hands. Automatic wrist-based cores can still truncate relevant movement; source-video inspection and LSF review remain necessary. All source observations remain available in the native motion and original video. Restore playback by removing derived `playback_quality`/`playback_window` metadata or restoring the ignored backups.

## Actual renderer measurements

See [metrics](motion-quality/metrics.json). Paired measurements use the actual Eva skeletal graph, the same captured poses/plans, 40 ms sampling, five homepage phrases and bibliothèque. The baseline is frozen from the base commit. The accepted change mainly removes setup/release and isolates bad hand samples; it is not a general arm-jitter solution. Large hand-joint steps fell from 172 to 159 for the fifth phrase, but increased 197 to 198 for the museum phrase. Wrist jump maxima and collision maxima are essentially unchanged. Remaining crossings, depth ambiguity and uncertain fingers are real limitations.

Two continuous torso/arm smoothing prototypes were rejected: they increased wrist jumps/collisions on real cases. Neither is shipped. This prevents a visually smoother-looking heuristic from silently introducing larger mesh penetrations. No claim of collision-free motion or validated LSF is made.

Reproduce renderer metrics from saved actual API plans/assets (each case contains `text`, `plan`, `assets` map entries):

```sh
cd frontend
CV_CASE_LIMIT=6 CV_MEASURE_INPUT=.runtime/motion-quality/cases \
AVATAR_MEASURE_ENTRY=scripts/measure-library-quality.ts \
node scripts/measure-avatar.mjs ../.runtime/motion-quality/repeated.json
```

The per-sign source preview gallery is served at `/media/library-review/`. Full audit, backups, actual clips and generation outputs remain local; the small measured report and reproduction code are committed. Tests cover isolated-vs-sustained hand errors, gap duration, immutable source data, playback mask semantics, fixed bones, core windows and deterministic seeking.

## Live verification

All six public video panels (hero plus five examples) decoded and advanced playback with no browser errors. The public MP4 byte-range request returned HTTP 206 with correct content range. Desktop and mobile screenshots were inspected. Recorded and inspected an additional real bibliothèque preview. Backend: 61 passed, 1 skipped; frontend motion tests, lint and production build passed. Some finger/arm artifacts remain.

Fresh inputs can be saved with `python scripts/prepare_motion_quality_cases.py`; they are written under `.runtime/motion-quality/current-cases` so the frozen measured cases are preserved. Point `CV_MEASURE_INPUT` at that directory for a fresh runtime diagnostic.
