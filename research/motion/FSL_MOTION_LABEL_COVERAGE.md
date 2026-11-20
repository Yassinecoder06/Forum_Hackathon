# France-LSF annotation and motion coverage

Matched plain bracketed UD FORM tokens to exact canonical gloss labels in
the local studio registry. Matching uses Unicode NFC, trims surrounding
whitespace and ignores case. It uses no aliases, fuzzy matches or semantic
substitutions. The scan excluded synthetic motions and checked all 110
candidate motion files with the normal playback asset validator. It read
the frozen training and development annotations only.

| Measure | Train | Development |
|---|---:|---:|
| Annotated sequences | 39 | 28 |
| Plain bracketed FORM occurrences | 168 | 150 |
| Exact label with a valid motion file | 70 | 74 |
| Exact label with a rights-cleared motion file | 37 | 49 |
| Exact label with rights and both review statuses cleared | 0 | 0 |
| Examples with every plain FORM matched | 2 | 6 |
| Other FORM occurrences outside this comparison | 47 | 44 |

An exact label only identifies a candidate asset. Its variant, handshape,
meaning and contextual use require signer review. The complete-plain-FORM
count allows unsupported annotation types elsewhere in the sequence and
does not indicate a renderable sentence. The check also leaves spatial,
simultaneous and nonmanual information outside the exact matching rule.

The registry count was 1,447 non-synthetic motions. Despite that inventory,
fewer than half of the inspected plain FORM occurrences had a valid exact
label match. Translating the source into the reference FORM sequence would
still leave a substantial animation coverage gap.

```bash
.venv/bin/python -m training.audit_fsl_motion_coverage \
  --output /tmp/fsl-motion-coverage-reproduction.json
```

[Aggregate result](FSL_MOTION_LABEL_COVERAGE.json)

## Follow-up on missing labels

Of 126 distinct labels without valid motion, only two have an exact-match
candidate in the cached inventories. Both source clips were available locally.
Re-extraction with the current worker produced the following results.

| Candidate | Pose | Left hand | Right hand | Asset terms |
|---|---:|---:|---:|---|
| APPORTER (Commons concept “apporter”) | 100% | 26.5% | 35.3% | CC BY 3.0 |
| COUPER (Dicta-Sign canonical label) | 100% | 26.6% | 35.8% | Pending |

The current worker processed 34 and 109 frames respectively. Earlier
coverage on COUPER was recorded at a different sample count, so this is
not a matched frame-by-frame extractor comparison. Both clips still leave
at least 64% of frames without a detection on either individual hand.
These are private review candidates. Neither result was registered for
automatic playback, and the exact inventory labels still require a
variant and meaning check before linking to a sentence annotation.

The audit also found 161 labels without any rights-cleared motion. The
private queue prioritizes missing training occurrences and includes
inventory references for review. The aggregate file contains no row text.

```bash
.venv/bin/python -m training.audit_fsl_motion_coverage \
  --output /tmp/fsl-motion-inventory-followup.json \
  --queue research/private/fsl-motion-inventory-reproduction.json
```

### Tracking inside the estimated signing interval

The studio's wrist-motion heuristic selected 1.214–2.143 s for APPORTER
and 2.640–4.040 s for COUPER. APPORTER had 8/14 left-hand and 11/14
right-hand detections. COUPER had 29/36 left-hand and 36/36 right-hand
detections. The whole-recording fractions were therefore strongly
affected by frames outside these windows. The intervals still require
signer review and tracking presence does not establish accurate
handshapes. The local review page at
`.runtime/fsl-missing-asset-recheck/index.html` opens the source videos
at the estimated intervals. No new registration or rights clearance
was applied. [Counts and artifact hashes](FSL_MOTION_CORE_RECHECK.json).
