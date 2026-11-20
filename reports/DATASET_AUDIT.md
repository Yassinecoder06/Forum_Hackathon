# SignVerse France-LSF audit

Pinned HF revision: `0c7e956a8471acad3a9cebaeb66bb23bf5deaafa`. Public, ungated, derived data CC BY-NC 4.0. Original video rights are not conveyed. Sources: https://huggingface.co/datasets/SignerX/SignVerse-2M and https://github.com/SignerX/SignVerse-2M . Downloads were performed in Colab CPU session `signaculture-fsl-audit`.

Both metadata tables contain39,197 records (the README says39,196). The original CSV is headerless: its first row is data, not a header. Exact language joins yield900 `fsl` videos:810train/45val/45test. Belgian `sfb`, Swiss `ssr` and all other languages are excluded.860FSL videos map to52published shards;40have no available mapping. Full release inventory lists724TARs totaling598,541,107,200bytes. No bulk shard download occurred.

Durations, caption languages and subtitle counts are absent from the relevant metadata rows; corpus-wide usable duration and segment counts remain unknown. Upload progress includes video-folder lists and processed metadata includes archive names; these permit bounded lookup. Range requests read500headers (256,000bytes) in the initial bounded scan. Inspection found actual paths `<video>/npz/poses.npz` and `<video>/captions/<video>.fr.vtt`, not the README's flat layout. Seven small FSL records were then fetched from known offsets (about22MB poses). All seven have French VTT tracks; transcription provenance remains unknown. No `caption.json` was found in those records. Metadata/video ID joins verify `fsl` in both CSVs; individual pose payloads have IDs but no language field. An embedded-language validation therefore cannot pass and is explicitly unverified.

Observed NPZ schema: video_id,fps,total_frames,frame_widths,frame_heights,frame_indices,frame_payloads. Person payloads have split XY arrays and score arrays. Body scores are indices/-1, not probabilities. Face/hand scores sometimes exceed1; those values are invalidated rather than silently clipped. XY values appear normalized despite the README describing pixels. Relative shoulder normalization does not require choosing pixel units, but absolute coordinate interpretation remains unresolved. Original files and scores remain preserved privately.

Gold gloss labels are absent. VTT text is weak subtitle supervision, never converted into glosses. Source language/permissions, original split, unknown signer ID, record ID and source revision are preserved in the private FSL manifest. No source-video availability/takedown crawl or underlying consent verification was performed; these remain limitations. No raw RGB downloaded or published.

The HF repository provides processing scripts. The GitHub tree contains website/docs rather than executable SignDW training code; no compatible official checkpoint/baseline was reproduced. Rosetta/Dicta-Sign continuous corpus download restrictions remain applicable: existing local metadata limits use to eligible public research/university members; eligibility is not established. Fresh handle requests also returned TLS verification failures. Restricted annotation content was not downloaded. AZee remains separate from gloss targets.

Machine-readable records: `data/manifests/signverse_fsl.jsonl` (private), `data/audit/signverse_summary.json`, `data/audit/sample_quality.json`. Raw audit metadata, source CSVs, index and upstream processing scripts are under ignored `data/audit/private/signverse/`. Names in private manifests support attribution and removal, not re-identification. Deletion requests must remove matching derived windows/checkpoints and trigger retraining; no live automated deletion feed is implemented.

## Local weak caption-to-pose manifest

Processed every French WebVTT cue in the seven available recordings. Of 95
cues, 87 had a complete 24 FPS frame range, a duration of 0.5–15 seconds, and
at least 50% valid body and hand landmark coverage. Eight failed the coverage
rule. The retained counts follow the original recording splits: 21 training,
47 validation and 19 test. The private manifest references exact normalized
pose slices and hashes. It carries the original source revision, license,
permission status and unknown signer identifiers.

The timestamps establish overlap in the release. French transcription
provenance and alignment to the signed content are still unverified.
This is a small, weak caption-to-pose pilot with zero gloss targets.
The limited training split does not justify scaling a text-to-pose model
on these seven recordings.

```bash
.runtime/t5-local-venv/bin/python -m src.data.prepare_signverse_caption_pose \
  --output data/manifests/signverse_fsl_caption_pose_reproduction.jsonl
```

[Aggregate output](../data/audit/signverse_caption_pose.json)
