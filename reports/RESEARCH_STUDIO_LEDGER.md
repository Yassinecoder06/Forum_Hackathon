# Research results in the studio

The Research panel now displays six recent completed runs from the local SQLite ledger. France-LSF and Belgian LSFB are explicitly labeled and filterable. Each card identifies its task, training/development counts, exact-match rate, and either gloss WER or graph F1. Multi-sign results or graph validity are shown where available.

These are developmental measurements with repeated inspection and model selection, not an independent benchmark. The panel states that tasks/splits differ, local linguistic approval is absent, and these recent models are not enabled for the avatar. GeoQuery's machine-translated French is disclosed. There is no global ranking across languages or tasks.

## Evidence handling

The backend opens the catalog read-only. It verifies the result-file SHA256, checks that stored metrics match that file, and checks the language against the recorded run configuration. Missing, changed, mismatched or external evidence yields null measurements and a visible unavailable-evidence state. Private checkpoint/report paths and source sentences are omitted. Unknown/missing catalogs are not created. Existing legacy inference registration remains a separate mechanism; none of these recent ledger entries enters it.

## Executed verification

- Six service-specific cases cover normal summaries, read-only access/privacy, missing catalogs, changed result hashes, external paths, conflicting metrics and conflicting languages (some cases check multiple properties).
- The combined research-summary/API suite passed 22 tests.
- Frontend build and lint passed.
- Real Chrome test read the actual API/catalog, verified all six records, filtered France to two and Belgium to four, checked measured 52.17% exact and 2.49% graph F1 values, confirmed no inference buttons, and checked the 390px mobile layout without page overflow.
- Visual inspection exposed an overly narrow table. It was replaced with cards; the browser test passed again. No API or model results were mocked.

Private screenshots: `reports/private/research-panel-browser/{research,mobile}.png`. Logs: `research_summary_api_tests.txt`, `research_panel_build.log`, `research_panel_lint.log`, `research_panel_browser.log` under reports.

```bash
.venv/bin/python -m pytest backend/tests/test_research_ledger_summary.py backend/tests/test_api.py -q
npm --prefix frontend run build
npm --prefix frontend run lint
CHROME_PATH=/home/jmal/.local/bin/google-chrome APP_URL=http://127.0.0.1:15177 npm --prefix e2e run test:research-ledger
```

The browser command requires the current API on port18004 and the Vite proxy on15177 with `SIGNACULTURE_API_TARGET=http://127.0.0.1:18004`. It requires the registered local research catalog and result artifacts; absent artifacts intentionally display unavailable measurements. No remote deployment or new model promotion occurred.

The overall goal remains incomplete: the functioning experimental studio, ingestion, audio analysis and export do not establish a general, linguistically validated translator. Qualified review, broader aligned supervision and independent multi-sign evaluation remain unresolved.
