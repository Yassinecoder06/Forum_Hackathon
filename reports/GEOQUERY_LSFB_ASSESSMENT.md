# GeoQuery-LSFB source audit and baseline

Source downloaded in an earlier Colab-backed resource audit;reused locally in this continuation without new downloads. Pinned commit879731568a5cd93177d11bf742740b187176c99a,https://gitlab.unamur.be/beehaif/GeoQuery-LSFB . Original source hashes and GPL2license retained under /home/jmal/Dataset/geoquery-lsfb-source-v1. No raw video download. Scope:BelgianLSFB only,separate from CONT andFrance-LSF.

## Provenance correction

Source README explicitly says French sentences were generated with DeepL from original English questions. The original250signed examples derive from a consenting native signer,recorded and transcribed withHamNoSys per authors. Thus human signing annotation and machine-translated French input are distinct provenance axes. Previously staged files'`synthetic:false`means original signed example,not human-authored French. Corrected derived records explicitly carry French/input and signed/target provenance. No local linguistic approval claimed. The expanded4,519release has synthetic entity replacement and is not used in this experiment.

## Actual structural audit

Original staged split:203train/47development,disjoint source template types. Entity-normalized FunQLstructures have zero train/validation overlap. Graph format retains right/left/two-hand articulations,HamNoSys strings and adjacent/during/start-coincides/end-coincides relations. It is not a single gloss sequence. Only one signer per source description,so no signer-generalization claim.

223/250graphs pass referential integrity checks.27(17train/10development)reference undefined articulation IDs. Example originalID34uses temporal references to`taille.1-1`while its articulation is named`taille.1-2`. These are source inconsistencies under the observed graph schema,not automatically repaired.14training and3development normalized-French input groups have differing target graphs;these may be legitimate alternatives. No contradictory examples were relabeled automatically.

There are255distinct training HamNoSys strings and48/564development articulation occurrences use unseen strings. This differs from the CONT lemma vocabulary. No ID or motion equivalence between corpora is asserted.

## Executed baselines on original47development graphs

|Method|Full graph exact|Triple microF1|Articulation microF1|Structurally invalid predicted graphs|
|---|---:|---:|---:|---:|
|Training-only character3–5gramTFIDF|0/47|22.50%|36.66%|3/47|
|Training-only word/bigramTFIDF|0/47|22.67%|35.35%|2/47|

Metrics retain every original reference,including10with structural problems;those are explicitly counted. Exact serialized graph and order-independent triple-set exact are both zero. F1uses complete authored triples includingHNSand occurrence IDs;it is not semantic adequacy or intelligibility. Retrieval runs on training-only fitted vocabularies/IDF. No synthetic expansion used,no model fine-tuned,no benchmark promotion.

## Prepared separate graph feasibility package

/home/jmal/Dataset/geoquery-lsfb-graph-feasibility-v1 contains186structurally valid training records,37valid development records and27quarantined originals. It preserves original split/ELANreference/hand roles/HamNoSys/temporal edges and explicit machine-translated input provenance. Normalized French overlap across clean train/val is zero;original disjoint query structures retained. No excluded examples were silently repaired,no temporal graph flattened into sequential glosses. SHA256manifest retained. These are feasibility data,not approved translations. Full record IDs and graph predictions remain private.

Local review tool:reports/private/geoquery_graph_review.html. Shows all250original graphs,marks dangling relation rows,and exports notes. Exported status is notes_only_not_approved;tool does not manufacture reviewer identity or acceptance. Reviewer must assess source alignments and any corrections against original recordings/annotations before training a linguistically validated system.

## Executed verification and reproduction

Four tests passed:temporal integrity/invalid reference rejection,invalid references retained in scores,serialized versus set graph scoring,and train-only retrieval including unknown inputs.

```bash
.runtime/t5-local-venv/bin/python -m training.geoquery_lsfb.audit_baseline
.runtime/t5-local-venv/bin/python -m training.geoquery_lsfb.prepare_graphs
.runtime/t5-local-venv/bin/python -m pytest tests/test_geoquery_graphs.py -q
```

Completed output directories refuse rerunning without a fresh name;source snapshots remain unchanged. Evidence:reports/GEOQUERY_LSFB_BASELINE.json,reports/private/geoquery-lsfb-baseline-v1/,reports/geoquery_graph_tests.txt. The47row baseline must not be compared directly with later37rowclean-subset runs.

## Next useful step and scope limit

GeoQuery offers clearer domain-limited signed annotation than generic conversation projection,but input translation,source graph inconsistencies,small original corpus,one-signer scope and compositional holdout difficulty remain. Validate source graphs and their French alignment first. Then test a lossless graph representation with HNSround-trip preservation and length/tokenizer checks before neural generation;standard T5tokenizers do not automatically represent private-use HamNoSysUnicode. Any synthetic training extension requires grouped structure/entity leakage checks. Do not attach these Belgian graphs to France-LSFavatar assets by spelling.
