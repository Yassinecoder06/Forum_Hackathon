# Lossless LSFB graph representation and train-only fit gate

This continues the separate Belgian GeoQuery investigation. It does not create a France-LSFtranslation benchmark or attach Belgian signs to France-LSFmotion assets. Source signed annotations come from original human signing;French inputs are machine-translated by DeepL per authors;local review remains pending.

## Implemented representation

Every author triple remains recoverable with its relation/hand role,original articulation occurrence name,HamNoSys,and original triple order. Relation codesR/L/B/A/S/E/Dshorten repeated labels without merging meanings. Temporal edge references use articulation indices;decoder restores original names and checks graph integrity. HNSdictionary fitted on186training graphs only;unseen HNSvalues use an explicit hexUnicode literal fallback. No validation HNSstrings enter the dictionary;no opaque value is silently dropped. This is a symbolic serialization,not a statement that its code indices carry language semantics.

## Executed audits

All223valid source graphs decode exactly after encoding;T5tokenizer encode/decode also preserves every target string. Zero unknown subtokens in186train/37development. Dictionary241entries. Training target length mean244.12,max536tokens. Development mean331.46,max1,196;oneexceeds1,024. No examples were truncated or removed for length. Runtime artifacts:.runtime/geoquery-graph-codec-v1/{train,val}.jsonl,hns_dictionary.json,AUDIT.json. Six tests passed including both-hands/Unicode literal/edge-order roundtrip and malformed reference/index rejection.

## Train-only gate

Runner:training/geoquery_lsfb/fit_gate.py. Fresh pinnedT5-base,rank16LoRAattention/feedforward,BF16,LR5e-4,no adapter dropout,seed92,batch4viaaccumulation. Eight deterministic training rows,up to300updates. Their maximum target373tokens;generation cap405. Gate passes only when at least7/8graphs exactly reproduce source triples,all8graphs are structurally valid,andno generation caps occur. No development rows read by the gate;no full-data training justified solely by lower loss.

At50updates,0/8exact,0/8valid,8caps. This is a measured intermediate failure;final outcome recorded after completion. Sources/targets remain unchanged. Model loss alone is not treated as correctness.

## Reproduction

```bash
.runtime/t5-local-venv/bin/python -m training.geoquery_lsfb.codec_audit
.runtime/t5-local-venv/bin/python -m training.geoquery_lsfb.fit_gate
.runtime/t5-local-venv/bin/python -m pytest tests/test_geoquery_graphs.py -q
```

Audits and training refuse existing output directories. A research candidate CLI(training/geoquery_lsfb/predict_graph.py)expects a completed adapter/tokenizer/dictionary;it returns invalid candidates explicitly and never repairs or approves them. It has not been demonstrated on a heldout graph. Saved artifacts are private and require the source license/provenance to remain attached.

## Completed first gate and bounded extension

At300updates:2/8exact,6/8valid,0caps,277.68seconds. Gatefailed against7/8exact+8/8valid threshold. All8Frenchinputs distinct. No heldout graphs evaluated. The original runningv1did not retain failed weights;its manifest/history/predictions/results are backed up. Future runner version saves failed adapters and accepts explicit --output/--steps. Candidate CLI remains compile-checked only;no deployable gate adapter claimed.

Version2now runs at `.runtime/geoquery-graph-fit-v2`,same seed/configuration/training rows,only maximum updates extended to600. It reruns from identical initialization becausev1failedweights were not saved;this is an optimization-budget diagnostic,not a new independent experiment. Log:reports/geoquery_graph_fit_v2.log. No full186rowtraining launched before the declared fit criterion passes.

Lossless codec/dictionary/paired files and hashes copied into /home/jmal/Dataset/geoquery-lsfb-graph-feasibility-v1/codec;v1fitresult backed up under fit-v1. Source graphs and invalid records remain unchanged.

## Extended gate passed

Version2passed at400updates:8/8exact author graphs,8/8structurally valid,0caps,370.91seconds. Config/seed/rows identical to first300updates;extension allowed further optimization. No development example opened by this fit stage. Adapter/tokenizer/train-onlyHNSdictionary saved in .runtime/geoquery-graph-fit-v2. This is memorization evidence,not heldout translation quality.

Full-data runner now implemented in training/geoquery_lsfb/train_graph.py. It requires a passed gate,verifies source/codebook hashes,uses all186valid train/37development records,retains the1,196token case under1,280target/generationbudget,and saves resumable optimizer/model/RNGstate each epoch. Selection by development token-weighted loss,patience3,max8epochs;no officialtest. Warmstart is the passed train-only8example adapter. Broken generated graphs remain scored as empty triple sets with invalid counts;none dropped. Clean-split retrieval baselines recomputed on the same186/37. Seven graph/metric tests pass;runner compile-checked before launch. Full-data outcome remains pending.

## Full-data completion

Full186/37run completed4epochs/188updates,selectedepoch1,0/37exact,1/37validgraphs,2.49%tripleF1versus22.96%characterretrieval,0capfailures.15repeatedIDand21danglingedgeerrors. No promotion. See reports/GEOQUERY_FULL_GRAPH_RESULT.md forconfiguration,backup,evaluation andlimitations.
