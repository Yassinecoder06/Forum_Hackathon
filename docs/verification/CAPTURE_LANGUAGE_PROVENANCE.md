# Capture language provenance change

Observed problem:new video/Animics captures were registered asLSFwithout a signing-language declaration. This could mislabel Belgian or other-language captures in the France-LSFstudio.

Implemented:new POST/captures/{id}/register requires sign_language:"fsl". Missing,sfb,ssr,und,LSF/FSL(noncanonical)values fail422. The source capture and motion preview remain available,while non-France captures cannot enter this library. Registration stores sign_language:fsl and language_status:declared_by_uploader_pending_review;review and rights remain pending. Direct service registration also rejects non-fslmetadata. Registry excludes explicitnon-fslmetadata even if a stored legacygloss claimsLSF. Existing legacyrows without this field retain their historicalhandling;this change does not retroactively certify them.

UI:language selector withFrance,Belgium,Swiss,other/unknown choices;onlyFrance enables registration. Declaration resets for each new extraction. No auto-detection or semantic validation claimed. Other-language research remains separate;captured assets are not relabeled across languages by spelling.

24backendAPI/capturetests passed,including missing/foreign declaration rejection,preview preservation andregistryconflict filtering. Frontendbuild,lint andgitdiffwhitespacecheckpassed. RealChromeAPI/UIbrowsercheck passed composition,seeking,WebMexport,audio/pause analysis,JSONcapture/provenance/persistence/lookup,mobile andofflineerrorcases. Initial selector accessible-name failure was fixed with explicitaria-label and rerun passed.

Evidence:docs/verification/logs/capture_language_tests.txt,capture_language_build.log,capture_language_lint.log,language_guard_browser.log;privatebrowserartifacts research/private/language-guard-browser/. APIcontract/docs updated. This is a provenance safeguard,not a review approval or a new translation model.

## 2026-11-17 follow-up: legacy source audit and strict registry

The historical fallback described above has now been removed. All 1,446 existing corpus records were individually matched to explicit France-LSF source inventories and capture hashes, then migrated with source-language evidence while preserving every non-language field and all pending reviews. The registry excludes undeclared legacy records. Both ingestion-script registration calls were fixed to supply fsl. See INGESTION_LANGUAGE_INTEGRATION.md and CAPTURE_LANGUAGE_MIGRATION.json for evidence; 41 regression tests and all 1,452 registered asset checks passed.
