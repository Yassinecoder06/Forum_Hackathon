# France-LSF training source review packet

The private packet at `.runtime/france-lsf-review-v002/` contains all 61
training pairs in the frozen v002 export. There are 39 UD pairs and 22 SUD
pairs, each labeled with its original convention. The generator verified the
training export and raw annotation checksums and reconstructed each target
from the original FORM rows in order. Existing validation and test exports
were excluded from the packet.

`source_evidence.jsonl` contains each French translation, lexical target,
recording reference, source segment IDs, native annotation details and token
timings. `decisions.tsv` has one blank review row per pair, keyed by the
pair ID and the SHA-256 of the original training JSONL line. Reproduction
generated both files byte for byte.

Sixty pairs have complete token timing. The remaining recipe example has a
partial interval, which must be checked against the original recording before
approval. A filename search in the local project, Downloads and Dataset
directories found no matching copies of the three training recordings.
No media was fetched and zero review decisions exist. The annotation license
and any permission to obtain or share the source videos are separate.

## Review procedure

1. With authorized access to the source recording, compare each French
   translation with the signed interval and inspect neighboring context where
   necessary. Use the token intervals as navigation hints.
2. Check whether the lexical FORM sequence is a suitable target for this
   specific task and note any discrepancy involving timing, meaning, or
   annotation convention. Use `accept`, `reject`, `needs_context`,
   `needs_alignment`, or `video_unavailable` in the worksheet. Add reviewer
   role, review date and notes.
3. Keep the worksheet and source evidence together. Review decisions need a
   separate checked import before they can affect training eligibility. This
   generator does not change frozen data, rights state, or catalog approvals.

```bash
python3 -m training.prepare_fsl_review --output .runtime/france-lsf-review-reproduction
```

## Validate a returned worksheet

```bash
python3 -m training.validate_fsl_review .runtime/france-lsf-review-v002/decisions.tsv \
  --output .runtime/france-lsf-review-v002/validated-decisions.json
```

Use a new output path for each submission. Keep the full worksheet and edit
only the four decision fields. Dates use YYYY-MM-DD. For each decision,
include a role and enough notes to explain the outcome. The validator checks
every pair ID, hash, annotation convention and interval against a freshly
verified copy of the dataset. It creates a private decision record without
changing the catalog. Reviewer qualifications and video rights still require
confirmation before any eligibility change.

The original worksheet validated with all 61 decisions pending. A worksheet
with a changed pair hash was rejected.

## Recording access audit

| Training recording | Pairs | What is verified |
|---|---:|---|
| LS-Colin CUC022 | 39 | The exact CoCoON deposit lists the annotated MP4 as a component and declares CC BY-NC-ND 2.5. An HTTPS HEAD request to the MP4 returned HTTP 403 from this environment. |
| CAP Petite Enfance | 11 | The pinned SUD annotations give a relative filename. No matching publisher media record or media license was verified. |
| Primaire Populaire en LSF | 11 | The pinned SUD annotations give a relative filename. No matching publisher media record or media license was verified. |

The [exact CUC022 metadata record](https://api.datacite.org/dois/10.34847/cocoon.cd3efaf4-bb76-38dc-af8c-6a2c65aa4c78)
describes the recording as freely accessible. The observed 403 establishes
the result of this request only. The DOI cited in the treebank README
actually links CUC020, a different recording. A separate
[CUC022 annotation deposit](https://api.datacite.org/dois/10.34847/cocoon.778ff90b-cb03-3613-8db0-06cf7ffdbaad)
lists EAF and XML components under CC BY-NC-SA 2.5. Both component URLs
also returned 403 to HTTPS HEAD. Their schema and alignment remain unverified.

The decision worksheet still has 61 pending rows. No media or additional
annotations were retrieved. For CUC022, the next access route is the
[CoCoON deposit](https://doi.org/10.34847/cocoon.cd3efaf4-bb76-38dc-af8c-6a2c65aa4c78)
or an authorized copy from the archive. For the other two recordings,
request the source references and permission terms from the treebank
maintainers before using the videos in review.

## Repetition across training recordings

The 61 training pairs contain 231 distinct plain bracketed FORM labels.
Only 15 appear in more than one recording. Of 242 distinct adjacent
plain-label pairs, only one appears in more than one recording.

| Annotation convention | Rows | Recordings | Shared plain labels | Shared adjacent pairs |
|---|---:|---:|---:|---:|
| UD | 39 | 1 | 0 | 0 |
| SUD | 22 | 2 | 7 | 1 |

The combined counts are an upper bound on label overlap if the conventions
are treated alike for this diagnostic. No merge was applied to the training
data. The scan accepts only whole bracketed FORM tokens, normalizes Unicode
NFC and case, and counts adjacency only when two original neighboring tokens
are both plain labels. It reads training rows only. Repeated label strings
do not establish repeated grammatical constructions.

Qualified review of these pairs would help assess alignment, but broader
supervision still requires new recordings with repeated constructions.
[Machine-readable counts](FSL_REVIEW_RECORDING_SUPPORT.json).
