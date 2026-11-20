# Cached ingestion repair and explicit France-LSF library provenance

Completed 2026-11-17. The preceding goal turn made progress by verifying real emotion2vec parameter coverage. This continuation audited the existing motion library against the Paper B requirement to keep France-LSF assets identifiable and separate from Belgian/Swiss resources.

## Bug fixed

The new capture-registration service requires `sign_language="fsl"`, but both legacy ingestion scripts still called it without that argument. Cached ingestion would therefore fail at registration. `scripts/ingest_commons_signs.py` and `scripts/ingest_dictasign.py` now pass the explicit identifier. Before IO, Commons requires its source inventory's `language_variant="LSF_FR"` and motion eligibility; Dicta-Sign requires `language="fsl"`. Missing/Belgian/Swiss declarations are rejected. This does not change the raw capture-preview API's ability to inspect other-language input separately.

Both scripts were exercised on genuine existing captures in an isolated temporary library with network calls forbidden. Commons retained its individually recorded license status and Dicta-Sign rights remained pending. Both remained linguistically and animation-unreviewed. Live sources, capture files and library hashes were unchanged by this replay. This checks cached ingestion and registration; it does not rerun computer vision or re-audit remote licenses.

## Removed the legacy language assumption

All 1,446 historical imported records lacked the newer language field. The registry previously treated absence as France-LSF. Their identities were audited individually against existing cached source inventories:

- 439 Commons records identified through explicit `LSF_FR` inventory entries and source-URL-derived capture IDs.
- 1,007 Dicta-Sign records identified through explicit `fsl` entries and matching per-capture source provenance.
- Cached source videos were checked against recorded capture SHA256 values. Eight early Commons records lacked provenance sidecars; each instead matched the original Commons file SHA1 in the cached inventory exactly.
- No unresolved records, new downloads or external access requests.

The applied migration adds `sign_language="fsl"`, `language_status="source_inventory_identified_pending_linguistic_review"` and per-record evidence containing inventory/source digests. Every non-language field is exactly unchanged: glosses, aliases, motion paths, quality, attribution, rights and review status. This identifies source language; it does not certify that a motion expresses its assigned concept correctly.

The original full library is preserved at `.runtime/library.before-language-migration.json`. The registry now excludes records with missing, unknown or foreign language declarations. Source-inferred legacy defaults are no longer accepted. New uploads still use the explicit uploader declaration and retain pending review.

Before library SHA256: `5dd370cc6c0bbec819f7a728357620878a1c069c47a6ffda71377e15e0a8796e`.
After library SHA256: `022881539c5bd2bd798b522bfedf972ca34b9da2b8726309fa01a67b9f61fc71`.

## Executed verification

41 tests passed: source-migration checksum/identity rejection, both offline cached ingestion paths, language/eligibility rejection, Dicta-Sign parsing, capture registration/foreign-and-missing metadata filtering and existing API tests. The pre-existing Starlette/httpx deprecation warning remains.

After migration, the actual API registry loaded and all **1,452 motion assets** passed its registered-file, endpoints, timing, finite-joint and metadata checks. The six base assets are included in that total; it is not a count of 1,452 approved natural signs. All 1,446 imported records retain their pending linguistic/animation reviews. A repeat dry run found zero records needing migration and left the post-migration library hash unchanged. This is a format/runtime check, not a perceptual or sign-language intelligibility study.

No new model was trained, no active motion trajectories were replaced and no historical human approvals were manufactured. Private backup and evidence copies were retained locally outside the ephemeral runtime as well.

## Reproduce

```bash
.venv/bin/python scripts/recheck_cached_ingestion.py
.venv/bin/python scripts/migrate_capture_languages.py
# Apply only after inspecting the dry-run report; preserves a historical backup:
.venv/bin/python scripts/migrate_capture_languages.py --apply
.venv/bin/python -m pytest backend/tests/test_capture_language_migration.py backend/tests/test_ingestion_language_registration.py backend/tests/test_dictasign_ingestion.py backend/tests/test_capture_pipeline.py backend/tests/test_api.py -q
```

The replay command uses existing local assets and refuses network access. Migration defaults to a dry run; its repeat-check report is separate from the original applied-migration report. The live registry reads the resulting metadata dynamically once the updated API code is loaded; an already running process without reload must be restarted to use new Python code. No user deployment process was interrupted here.

Evidence: `reports/CACHED_INGESTION_RECHECK.json`, `reports/CAPTURE_LANGUAGE_MIGRATION.json`, `reports/CAPTURE_LANGUAGE_MIGRATION_RECHECK.json`, `reports/CAPTURE_LANGUAGE_ASSET_RECHECK.json`, `reports/ingestion_registration_tests.txt`, and the associated replay/migration logs.

## Remaining limits

The existing library is an experimental lexical-motion resource. Many clips have partial hand tracking; MediaPipe depth and retargeting remain approximate. Dicta-Sign rights remain pending and cannot be inferred from public accessibility. Source-language metadata is distinct from qualified France-LSF semantic/animation review. Belgian lexical-projection checkpoints remain separate and are not mapped into this France-LSF library. Successful ingestion does not resolve the demonstrated text-to-gloss generalization failures, missing linguistic supervision, calibrated 3D reconstruction or fluent-signer comprehension assessment.
