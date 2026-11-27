# LSFB source alignment audit and synchronized video review

Completed 2026-11-17 using existing local LSFB-CONT assets; no downloads, training or changes to source data/labels.

## Measured source consistency

All 2,531 extracted research/code/training/development records were joined to their original frozen preparation candidates by ID. Every original candidate content hash was checked. For all 2,531 records, recomputing the lexical projection from original aligned events reproduced the training target exactly, and recording IDs and translation intervals matched. No retained event had less than the prescribed 80% overlap with its French segment. No retained row had source events excluded at segment boundaries. No annotation frame/time discrepancy exceeded 40 ms under the source 50 FPS convention.

These results establish internal preprocessing consistency only. They do not independently verify that video clocks, French translations and linguistic meaning are aligned. The audit did not reopen the official test shard.

1,577 / 2,531 records (62.31%) contain aligned nonlexical events outside the lexical target. Those events remain in the sidecar source annotations; they are omitted from the model's prediction objective. Pointing and other omitted event classes must not be mistaken for an absence of signing. Nonmanual grammar completeness is also unestablished. This is one concrete reason a lexical projection cannot be presented as a full signing translation, even if its token metrics improve.

## Private review artifact

`research/private/lsfb-alignment-audit/video-review/index.html` contains **55 actual source-video excerpts**: all 49 narrow multi-sign training examples plus all six previously inspected multi-sign development examples. All 55 excerpts were successfully extracted locally using the existing TAR index, checked against TAR headers; no full archive extraction was performed. They include eight seconds before and three seconds after the annotation where available, at 640-pixel width and 25 FPS for compact review. They are review proxies, not the original native-rate recordings.

Each page row shows original French/context, source annotations and the experimental lexical target separately. Clickable annotation events seek to their source times. Playback highlights active events in the left-hand, right-hand and both-hand tiers and indicates whether the playhead is inside the translation interval. Original candidate hashes and task identifiers are retained for provenance-bound review exports. The experimental projection shown alongside them does not replace source annotations.

Every review begins pending. No decisions or qualified approvals were created. Acceptance requires an available video excerpt and all existing source/meaning/context/representation checks. The browser can collect a named review; the software cannot establish a reviewer's linguistic qualifications. Existing release validation remains separate and will reject stale hashes or excluded rows. Any later correction still requires preparation/tokenization/split checks before training admission.

## Executed verification

- 15 tests passed: original preparation/release/TAR protections and new source-hash, projection-mismatch and nonlexical-omission checks.
- Actual installed Chrome: all 55 pending rows/videos rendered; first video decoded; source-event seeking and highlighting matched its timestamp; missing-check and missing-video acceptance attempts were rejected; zero decisions were created.
- All 55 clips passed ffprobe checks for positive duration and H.264 video.
- Original sources and derived training labels were not changed. No new model was promoted.

## Reproduction

Use a fresh audit/review output directory if these already exist:

```bash
python3 -m training.lsfb_cont.alignment_audit
python3 -m training.lsfb_cont.review --root /home/jmal/Projects/Active/Forum_Hackathon/datasets/lsfb-cont --candidates research/private/lsfb-alignment-audit/review_candidates.jsonl --output research/private/lsfb-alignment-audit/video-review --limit 55
.runtime/t5-local-venv/bin/python -m pytest research/code/training/lsfb_cont/test_alignment_audit.py research/code/training/lsfb_cont/test_prepare.py -q
cd e2e
node lsfb-review.mjs
```

The browser check defaults to the actual local review pack and installed Chrome; `REVIEW_HTML` and `CHROME_PATH` can override paths. Logs: `research/text-to-gloss/lsfb/logs/lsfb_alignment_tests.txt`, `research/text-to-gloss/lsfb/logs/lsfb_alignment_video_review.log`, `research/text-to-gloss/lsfb/logs/lsfb_alignment_browser.log`. Machine-readable aggregate: `research/text-to-gloss/lsfb/LSFB_SOURCE_ALIGNMENT_AUDIT.json`. Per-example traces and clips are private local materials and should not be published or mixed with France-LSF assets.

## Remaining requirement

A qualified Belgian LSFB reviewer must inspect meaning, timing and target completeness, particularly pointing and other omitted events. None of these automated checks can supply that approval. After review, repair and freeze an adequately supported conversation-held-out study before more broad translation training. Existing tiny-development seed results remain diagnostic, not a working sentence translator.
