# Simplified diagrams and result tables

16:9 pages (1600×900) on a plain white background, for slides and documents. Each page exists as an editable SVG and a 2× PNG; `index.html` shows them all.

| Page | Content |
|---|---|
| 01-overview | French sentence → lookup → sign sequence → stitching → avatar, with a worked example |
| 02-capture | Source video → landmarks → captured joints → motion file; hand-recovery ladder |
| 03-retargeting | Captured joints → five retargeting steps → rendered avatar |
| 04-timeline | Clip bridges (333 ms quintic), voice-timed mode, hand-tracking states |
| 05-source-pose-avatar | One sign at three moments: source frame, captured joints, avatar |
| 06-data-sources | Dataset table and corpus totals |
| 07-language-results | Text→gloss model results and concept-planning comparison |
| 08-motion-quality | Transition filling, finger jitter, hand overlap, eyebrow fixes |

## Changing the example sign (e.g. from your own dataset)

Every image and skeleton comes from the sign described in `sample.json`: its captured motion file, its source video, three timestamps and a credit line. To use another sign, edit `sample.json`, start the app (`make dev`), then run from the repository root:

```bash
npm --prefix e2e run diagram-examples   # source frames (ffmpeg) + avatar frames (running app)
python scripts/build-diagrams.py        # SVGs + index.html; skeletons drawn from the motion file
npm --prefix e2e run diagram-png        # optional PNGs
```

Set `CHROME_PATH` if Playwright's own Chromium is not installed. Table numbers are written in `scripts/build-diagrams.py` with their source files named in each page footer.
