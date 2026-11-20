# France-LSF SUD recording holdout baseline

Evaluated all 130 locally available SUD segments with both a French free
translation and source FORM annotations. All targets use the SUD convention.
Seven documents map to six recordings. Both LS-Colin extracts use the same
video, so their 40 examples are held out together. Each recording is evaluated
once using the other five as the training pool. The diagnostic reuses source
material inspected in earlier experiments, and signer identities are not
verified across recordings.

| Method | Exact | FORM-token error rate |
|---|---:|---:|
| Nearest French unigram/bigram TF-IDF retrieval | 0/130 | 134.75% |
| Most frequent training sequence, per fold | 0/130 | 128.19% |

In total, 593 of 823 reference FORM occurrences (72.05%) are absent from
the corresponding training fold. Only 3 of 130 references have full FORM
coverage. Unicode NFKC normalization and case folding leave 586 unseen
occurrences and 4 fully covered references, with the same baseline scores.
This diagnostic keeps neutral position and other source FORM tokens, so the
coverage statistic refers to this annotation convention. It does not measure
the percentage of signs a signer or avatar can produce.

This expands the coverage audit from one split to all available recordings.
The result makes simple retrieval and any decoder limited to observed FORM
labels poor candidates for this corpus. An open vocabulary model could emit
unseen spellings, but the experiment provides no evidence that it could learn
the missing annotation mappings. No new T5 training was launched. The next
supervision acquisition should prioritize repeated annotated constructions
across independent recordings and qualified review of the French alignment.

Source annotations are locally cached from the pinned SUD revision recorded
in the v002 manifest and licensed CC BY-SA 4.0. The evaluation reports only
aggregate counts and source hashes. The corpus supplies free translations and
published annotations with local linguistic review still pending.

```bash
python3 -m training.fsl_sud_recording_baseline --output /tmp/fsl-sud-recordings.json
cmp /tmp/fsl-sud-recordings.json research/text-to-gloss/fsl/FSL_SUD_RECORDING_BASELINE.json
```

The command was rerun to a new destination and the result matched byte for byte.
