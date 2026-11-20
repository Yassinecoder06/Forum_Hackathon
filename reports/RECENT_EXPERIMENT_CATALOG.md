# Recent experiment catalog registration

Six completed experiments were registered in the existing `data/catalog/catalog.sqlite`. The `model_runs` table records model configuration, measured metrics, dataset digest and checkpoint location. A new `research_experiment_index` table records signing-language code (`fsl` for France, `sfb` for Belgium), representation type and result-file SHA256.

Registered runs cover Belgian LSFB extracted, narrow-context and narrow-current models; the Belgian GeoQuery full-graph model; and the France-LSF mT5-small and T5-base comparison. Incomplete runs are not registered as completed. Registration does not enable application inference or imply linguistic approval.

The original training dataset identity is preserved when the current manifest has changed. Current manifest hashes and mismatch flags are additional metadata. Unavailable historic model revisions remain null.

The original 11 sources, 128 pairs, 238 annotations and zero reviews were preserved. Ordered-row SHA256 comparisons against the SQLite backup verified that all four tables' contents were unchanged. Database integrity and foreign-key checks passed. The backup is `.runtime/catalog-before-recent-experiments.sqlite`.

Nine ledger/graph tests and eight legacy catalog regression tests passed. They cover idempotent registration, language-mismatch rejection and preservation of recorded dataset identity. The SQL schema snapshot was updated.

```bash
.runtime/t5-local-venv/bin/python -m training.register_recent_experiments
.runtime/t5-local-venv/bin/python -m pytest tests/test_experiment_catalog.py tests/test_geoquery_graphs.py -q
PYTHONPATH=training:. .runtime/t5-local-venv/bin/python -m pytest training/test_catalog.py -q
```

Evidence: `RECENT_EXPERIMENT_CATALOG.json`, `CATALOG_SOURCE_CONTENT_VERIFICATION.json`, `research_ledger_tests.txt` and `catalog_regression_tests.txt` under `reports/`.

This is local research metadata registration. Translation adequacy, qualified signer review and restricted-corpus access remain unresolved.
