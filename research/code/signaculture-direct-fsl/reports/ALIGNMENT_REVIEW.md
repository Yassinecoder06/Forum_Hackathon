# France-LSF caption alignment review

## Material ready for private review

The worksheet at `outputs/alignment-review/index.html` contains 34 observed skeleton clips, two cue samples per recording across 17 recordings. The samples come from the first and third quarters of each recording's ordered usable cues. The accompanying `review.csv` includes the caption, source recording reference and time interval. It is blank for all reviewer judgments. Keep the worksheet and source references private.

A fluent France-LSF reviewer should first watch each trajectory, then compare its meaning with the French caption. Score tracking readability separately from meaning alignment. Use **indeterminate** when occlusion, missing handshape or facial detail, or absent context prevents a judgment. A skeleton-only result can flag problems but may not resolve them. For any timing correction, use an authorized source video with surrounding context and record signed offsets for both boundaries. Perform this check independently with a second reviewer where possible and report disagreements by recording.

## Automated checks completed

| Set | Retained cues | Median subtitle span | Mean visible body landmarks | Mean visible hand landmarks |
| --- | ---: | ---: | ---: | ---: |
| Training | 102 | 3.29 s | 75.3% | 84.2% |
| Validation | 47 | 2.20 s | 73.4% | 81.1% |
| Original test | 19 | 3.18 s | 76.9% | 87.5% |
| Additional test | 26 | 3.87 s | 77.2% | 82.1% |

All retained cue windows had every expected indexed frame at 24 FPS and passed the existing minimum body and hand visibility threshold. One of the 27 new candidates failed the visibility threshold. These checks establish pose coverage within the subtitle timestamps. Meaning alignment, actual signing boundaries and translation provenance await review.

## Return the worksheet

Keep `reference.csv` private and unchanged. Fill the rating columns in a copy of `review.csv`. Use the worksheet's French rating labels and enter `squelette` or `vidéo source` under `evidence`. A positive offset moves a subtitle boundary later. Enter both offsets, including zero where appropriate, only when authorized source footage supports them.

```bash
python scripts/collect_alignment_review.py outputs/alignment-review/reference.csv reviewer-return.csv --output outputs/alignment-review/reviewed-intervals.jsonl
```

The intake checks cue identity, applies signed timing offsets, and reports counts by original split. It exports interval candidates only for full meaning matches with source-video timing evidence. Those intervals still require pose re-extraction and coverage checks before training. Running the intake on the current worksheet found 34 unreviewed items and zero interval candidates.

## Prepare reviewed motion

After collecting a returned worksheet, run the corrected interval preparation from the experiment directory:

```bash
python scripts/prepare_reviewed_intervals.py outputs/alignment-review/reviewed-intervals.jsonl \
  --original ../.runtime/direct-fsl-expanded/cues.jsonl \
  --additional ../.runtime/direct-fsl-holdout/cues.jsonl \
  --output outputs/alignment-review/reviewed-poses.jsonl
```

The script preserves original splits and timestamps, recalculates frame slices, and checks pose checksums, continuous frame coverage, and body and hand visibility at the corrected boundaries. It preserves the separate additional test cohort. The empty worksheet produced zero accepted rows. A simulated judgment on one cached test cue verified correct re-slicing and rejection of an out-of-range interval. The simulated judgment was used only to verify the software and was excluded from the output.

## Review in the browser

Open the private `index.html` in the same folder as the 34 MP4s. The page keeps the subtitle collapsed until requested. Save each response and select **Exporter les réponses CSV** when finished. Answers are stored in the current browser when local storage is available. Export before closing if the page reports session-only storage. The page works locally with no network connection.

A headless Chromium check opened all 34 item options, rejected a one-sided timing correction, persisted a simulated complete response across a reload, and exported valid CSV containing quotes and a multiline note. The Python intake accepted the exported example, and the pose preparer re-sliced its cached test cue while preserving its split. The simulated response stayed in a temporary browser profile and temporary files. The actual worksheet remains blank.

## Measure the effect of corrected boundaries

```bash
python evaluate_reviewed_boundaries.py outputs/alignment-review/reviewed-poses.jsonl \
  --original ../.runtime/direct-fsl-expanded/cues.jsonl \
  --additional ../.runtime/direct-fsl-holdout/cues.jsonl \
  --checkpoint checkpoints/expanded/best.pt \
  --french-features ../.runtime/holdout-text/features.pt
```

The command keeps the checkpoint, input captions and training template fixed. It compares the original and corrected targets on the intersection of their visible landmarks at each normalized motion phase. Results are grouped by held-out cohort. This measures the effect of changing target boundaries on the same set of reviewed cues. It does not measure the effect of training on reviewed data.
