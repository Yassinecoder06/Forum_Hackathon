# Frozen graph reranking diagnostic

This experiment used the selected epoch-1 T5 adapter without further training. For each of the 37 development questions, character TF-IDF retrieved eight valid training graphs. T5 selected the candidate with the lowest mean teacher-forced token loss. A second pass used cyclically shuffled French inputs with the same candidate sets.

| Method | Exact graphs | Structurally valid | Triple micro F1 | Articulation micro F1 |
|---|---:|---:|---:|---:|
| Character retrieval | 0/37 | 37/37 | 22.96% | 36.53% |
| Frozen T5 reranker | 0/37 | 37/37 | 17.67% | 27.37% |
| Shuffled French, same candidates | 0/37 | 37/37 | 18.16% | 27.68% |

Reranking was worse than retrieval. It also showed no benefit over this shuffled-input control on the measured overlap metrics. Restricting candidates guarantees structural validity, but does not establish that the graph expresses the input meaning.

The candidate bank cannot produce novel graph structures. This diagnostic therefore cannot replace compositional translation. The mean per-example best-candidate F1 oracle was 34.41%; references were used only after selection to compute that ceiling. This macro score is not directly comparable with the corpus micro F1 values above.

The run took 14.26 seconds on the local GPU. It retained all 37 predictions and used no additional training, official test labels, synthetic expansion or human approval. The small development set has already been inspected. French inputs remain DeepL translations; source graphs remain authored LSFB transcriptions.

Evidence: `research/text-to-gloss/geoquery-lsfb/GEOQUERY_RERANK_RESULT.json` and `.runtime/geoquery-rerank-v1/{RESULT,predictions}.json`. A local backup with hashes is at `/home/jmal/Projects/Active/Forum_Hackathon/datasets/geoquery-lsfb-graph-feasibility-v1/rerank-v1/`.

```bash
.runtime/t5-local-venv/bin/python -m training.geoquery_lsfb.rerank_graphs
```

The runner refuses to overwrite its completed directory; choose a new output path in the script for reproduction. Retrieval remains the stronger measured baseline here, but its zero complete matches also preclude a usable-translator claim.
