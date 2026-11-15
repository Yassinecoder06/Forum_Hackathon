# Full pipeline diagrams

The four-page vector PDF is saved at the repository root as `SignaCulture_Full_Pipeline.pdf`.

1. Current application: French text, default concept lookup, optional Qwen drafts, motion composition, audio analysis, avatar and WebM export.
2. Video ingestion: rights, checksums, MediaPipe, tracking review, real motion library and traceability.
3. Research: separate UD/SUD data and T5/mT5 experiments, Qwen evaluation, and limits of paper coverage. No GAN is implemented in SignaCulture.
4. Architecture of the supplied T5–GAN paper, now read in full, and the missing implementation components.

The SVG pages and HTML are editable sources. Counts and metrics describe the implementation snapshot of 15 November 2026.

Rebuild from the repository root:

```bash
python3 scripts/build-pipeline-diagram.py
# Requires npm ci --prefix e2e and a Playwright-compatible Chromium installation.
CHROME_PATH=/path/to/google-chrome node scripts/print-pipeline-pdf.mjs
```

The printer uses local files and fonts; it requires no running API or model server. The PDF is A3 landscape with selectable vector text.
