# France-LSF lexical pose adapter

A real captured-lexicon → controlled French/gloss lookup → upstream pose concatenation → diagnostic2Dvideo path. This is NOT linguistic translation or a full paper reproduction. No training, generative video model, synthetic pose fixture or Swiss/ASL/German lexicon is used.

Run from the project root:

```bash
integration/spoken_pipeline/setup.sh
integration/spoken_pipeline/.venv/bin/python integration/spoken_pipeline/pipeline.py export --output integration/spoken_pipeline/runtime/lexicon
integration/spoken_pipeline/.venv/bin/python integration/spoken_pipeline/pipeline.py render --lexicon integration/spoken_pipeline/runtime/lexicon --glosses BONJOUR MERCI --output integration/spoken_pipeline/runtime/bonjour-merci
integration/spoken_pipeline/.venv/bin/python integration/spoken_pipeline/pipeline.py render --lexicon integration/spoken_pipeline/runtime/lexicon --text 'Bonjour merci' --output integration/spoken_pipeline/runtime/text-demo
```

Default source root: this project checkout (read-only). Override export's `--library-root` when moving machines. Re-run export to refresh the latest completed capture registry snapshot; no source files are modified. Export produces real .pose files, upstream-compatible index.csv and a provenance/masking/timestamp manifest. All runtime outputs and isolated Python environment are ignored. Requires installed ffmpeg/ffprobe. Upstream MIT revision fec7ac393989f4b6541ab5f860819262a54f27c7 is verified by setup.sh. Exact dependency versions and LICENSE are retained.

Pending-rights Dicta-Sign assets require an explicit **local research preview** gate:

```bash
integration/spoken_pipeline/.venv/bin/python integration/spoken_pipeline/pipeline.py export --allow-local-preview --output integration/spoken_pipeline/runtime/lexicon-preview
integration/spoken_pipeline/.venv/bin/python integration/spoken_pipeline/pipeline.py render --lexicon integration/spoken_pipeline/runtime/lexicon-preview --text 'Bonjour merci' --output integration/spoken_pipeline/runtime/dicta-bonjour-merci
```

This includes only captured assets already in the lead registry. It does not scrape/download portal data. Best source priority is applied before canonical ambiguity: Dicta-Sign overrides lower-priority Commons variants (including ÂGE vs AGE); different canonicals within the winning priority remain ambiguous. Pending rights stay explicit and preview artifacts must not be redistributed. Default export excludes them and all synthetic fixtures. Language is fr→fsl (France), never ssr/LSF-CH.

Raw Animics PLM/FLM/LLM/RLM arrays are preferred when supplied by a real capture. Original normalized image coordinates and visibility are exported; missing fields stay masked/zero-confidence. PWLM uses a different axis convention; independent world hands are absent and never fabricated. For older captures, actual stored body/finger coordinates are mapped by a documented reversible affine projection; absent face/hips/legs are masked rather than reconstructed from ratios/rig roots. Presence flags are not calibrated detector confidence. Source timestamps remain in the manifest.

The adapter calls actual upstream lookup, shoulder normalization and mask-preserving join functions. Custom nearest-observation resampling aligns native rates to25fps without interpolating a missing point. Each repeated frame references its source-frame index. Default upstream smoothing/trimming/speed caps are deliberately bypassed to retain all captured frames and their masks. The join can remain abrupt; this is a diagnostic concatenation, not validated coarticulation. Actual pose-format visualizer methods encode MP4/GIF; no pix2pix model is invoked. See evidence/upstream.json for exact reused versus custom methods.

Unknown words or true canonical ambiguity stop generation with explicit JSON and exit2; they are never silently dropped or fingerspelled. Successful render writes sequence.pose, diagnostic.mp4, diagnostic.gif and result.json; JSON on the last stdout line reports actual sources, rights, unknowns, masks and video metadata. `--pose-only` skips video; `--fps` controls fixed-rate output. French text matching uses simplemma's offline French lemmatizer and controlled longest-phrase lookup, not a signed-language grammar model.

```bash
PYTHONDONTWRITEBYTECODE=1 integration/spoken_pipeline/.venv/bin/python -m pytest integration/spoken_pipeline/test_adapter.py -q -o cache_dir=integration/spoken_pipeline/.pytest_cache
```

Tests use the actual local exports and source measurements. The cleared export above is required. Preview tests default to the app's completed `.runtime/lexical_pipeline/lexicon`; when using the standalone export above, set `SIGNACULTURE_PREVIEW_LEXICON=integration/spoken_pipeline/runtime/lexicon-preview`. Tests render fresh temporary Bonjour/Merci videos and an unknown-word rejection rather than depending on old ignored demo files. Recorded successful runs and snapshot counts are in evidence/; historical snapshots are distinct from the completed corpus. Read research/SPOKEN_PIPELINE_INTEGRATION.md for the measured handoff.

Manifest integration key: `fingerprint` is a stable SHA256 over canonical JSON for the export's upstream revision, languages, preview gate, source registry hash, records and errors; elapsed time is excluded. `source_library_sha256` identifies the exact read-only registry snapshot. Render JSON carries `lexicon_fingerprint`. The priority/fingerprint follow-up changes no captured poses and does not re-export a moving partial corpus.

Completed bulk capture: the final local lexicon contains **1,447 real entries: 1,007 raw Dicta-Sign captures + 440 earlier motions**, with zero export errors. The final export took 78 seconds on this machine; subsequent text runs reuse it. Both BONJOUR cs380 and MERCI cs905 are now Dicta-Sign sources: 80 frames / 25 fps / 3.2 s. Raw normalized image coordinates are converted using original video width for x/z and height for y, preserving aspect ratio. Earlier 638-entry evidence is a historical intermediate snapshot. See [the completed pipeline](../../docs/DICTASIGN_PIPELINE.md).
