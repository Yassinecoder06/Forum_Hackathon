# Private offline transfer bundle

`build_bundle.py` snapshots the current app, its prebuilt UI, 1,446 corrected/reconciled capture assets, deduplicated active raw landmarks, source videos, emotion2vec weights, isolated Linux Python runtimes, pinned upstream source and media tools. It omits duplicate histories, credentials, training state and the removed Qwen runtime. Runtime assets remain a private local research transfer; their source rights/review statuses are preserved, not upgraded.

The ready-made binary environments target Linux x86-64 compatible with the tested Ubuntu 24.04 host. On Windows use WSL2, or recreate native dependencies on another platform. Do not execute from MTP; copy/extract onto a local filesystem with at least 12 GB free before copying (8 GB additional after the ZIP is copied).

```bash
python3 integration/portable/build_bundle.py --stage .runtime/portable-build/SignaCulture
python3 .runtime/portable-build/SignaCulture/integration/portable/bootstrap.py --check
# Precompute against the snapshot's corrected raw paths before final packaging:
(cd .runtime/portable-build/SignaCulture && integration/spoken_pipeline/.venv/bin/python integration/spoken_pipeline/pipeline.py export --output .runtime/lexical_pipeline/lexicon --library-root . --allow-local-preview)
python3 integration/portable/build_bundle.py --stage .runtime/portable-build/SignaCulture --archive-only --zip .runtime/SignaCulture_offline_linux_x86_64.zip
```

Copy the ZIP, checksum and companion `BOOTSTRAP.py` to the destination. Run `python3 BOOTSTRAP.py --check`, then the same command with `--start`. The companion repairs ZIP-extraction execution permissions, including the FFprobe ELF loader. Open http://127.0.0.1:8080 . No frontend build, model download, training or CV re-extraction is needed. Bootstrap restores interpreter symlinks, editable import paths and cached-manifest paths. Requirement snapshots record the installed environment; if rebuilding the pose environment, replace its historical editable-file line with `-e integration/spoken_pipeline/runtime/upstream` and keep the recorded upstream revision.

The test transfer ZIP is 3,858,010,143 bytes, contains 71,869 entries, and expands to 7,050,237,602 bytes. SHA256: `403ec58ac78fd1e765467db7c2cb70fdfd99439c3da71c543cbb60e2d2f0ed4a`. Its root `agent_prompt.md` describes exact startup/recovery steps. A checksum sidecar is supplied for verification on the receiving computer after local copy.

Verification: all four relocated runtime imports, real MediaPipe extraction of Bonjour, actual local emotion2vec CPU inference on an explicitly synthetic engineering tone (no emotion-quality claim), actual prebuilt-UI manual composition/browser checks, and Bonjour/Merci pose/video export succeeded. The snapshot contains a precomputed lexicon with 1,447 entries (1,446 from active raw arrays, one curated stored-motion example), with no export errors. Publication/linguistic validation remains pending.
