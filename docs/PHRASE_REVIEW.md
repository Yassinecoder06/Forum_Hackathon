# Review of the 14 requested French phrases

Executed all 14 user-supplied sentences in the real browser application before and after the fixes. Each run saves the actual 3D avatar WebM, three paused screenshots, selected glosses, unrepresented words, motion inputs and rig diagnostics. All **28 WebM files** were independently inspected with ffprobe for decoded frames and video metadata. Sampled frames were visually reviewed, including all 14 original midpoint views and corrected verb sequences. This is software review, not linguistic ground truth or a claim that all videos were linguistically validated.

The local comparison gallery is `artifacts/phrase-review/index.html`, with original and regenerated videos, downloads, glosses, missing-word labels and half-speed playback. Source-derived videos and screenshots remain ignored local research previews. [Committed results](PHRASE_REVIEW_RESULTS.json) retain textual scenarios and diagnostics.

## Concrete fixes

The French contextual lemmatizer returned `habit`, `dort` and `cherche` rather than matching the existing HABITER, DORMIR and CHERCHER recordings. A pinned local French dictionary fallback now runs only when the original lookup found no candidate. Exact matches and canonical ambiguity remain authoritative. No new sentence/sign pairs or signs were invented.

| Phrase | Previously omitted | Regenerated lexical sequence |
|---|---|---|
| 5. Mon ami habite à Paris. | habite | MON → AMI → HABITER → PARIS |
| 8. Le chat dort. | dort | CHAT → DORMIR |
| 11. Je cherche mon téléphone portable. | cherche | JE → CHERCHER → MON → TELEPHONE PORTABLE |

The original export did not identify missing words in its video caption. Exported incomplete previews now name those words explicitly. The other 11 gloss sequences remain unchanged. Articles, prepositions and other unrepresented tokens remain reported; they are not silently declared unnecessary in LSF.

## Observed remaining problems

Sampled views show incomplete hand tracking and hand overlaps, including hands around the torso during rest/bridges. The previous stabilization and contact-aware solver remain active, but **no additional collision improvement is claimed for this run**. The system cannot reliably reconstruct every occluded hand from monocular footage. Stronger automatic repulsion could break intentional contacts.

Lexical concatenation still includes source rest intervals and repeated returns toward rest, and grammatical correctness is pending. Phrase 10's LANGUE → SIGNE sequence especially needs lexical-sense and linguistic review before it can be called an appropriate rendering of “langue des signes”. No automatic semantic substitution was made. Matching every token in phrases 11 and 13 is not proof of correct signing.

## Verification

The unified suite passed **76 backend/schema/geometry tests**, frontend motion/avatar checks, lint and production build. New tests cover recovery of conjugated verbs while preserving exact-match priority and ambiguity rejection. The actual application recorded every regenerated video; all artifacts were decoded by ffprobe. The API is running with the installed pinned `simplemma==2.0.0` dependency.

Reproduce the browser recordings from `e2e/` with the app running:

```bash
CHROME_PATH=/path/to/chrome E2E_OUTPUT=../artifacts/phrase-review/after node phrases.mjs
```

The runner resumes completed cases from `results.json`; choose a new output directory for a new complete run.
