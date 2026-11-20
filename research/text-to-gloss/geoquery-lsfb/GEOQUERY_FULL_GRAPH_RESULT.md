# Full GeoQuery LSFB graph experiment result

Completed17November2026;BelgianLSFB only. InputFrench is DeepLtranslated;targets are original author transcriptions of human signing,with local review pending. Domain isUSgeography;no France-LSFmotion mapping or naturalness claim.

## Gate and execution

Lossless codec passed223/223graph/T5tokenizer roundtrips withoutUNK. Train-onlyHNSdictionary241entries,literalUnicodefallback forunseenvalues. Eight-training-example gate passed at400updates:8/8exact,8/8valid,0caps. A fresh inference process exactly reproduced one selected source training graph. Gate andartifacts are train-set learnability evidence only.

Warmstarted full-data run from that8TRAINexampleadapter:186train/37development,seed93,rank16LoRAattention/feedforward,BF16,effectivebatch4,LR2e-4warmup20updates,max8epochs,developmenttoken-loss selection/patience3. Source files anddictionary hashes checked. No synthetic expansion. Entire1,196tokenreference retained under1,280tokenbudget. No officialtest labels opened. Development is not signer-independent;one original signer per source.

Fourepochs188updates executed;selectedepoch1. Validationtokenloss2.1156→2.1409→2.1525→2.2346,soearlystoptriggered. Selectedadapter freshly reloaded before all37development graphs decoded. Total236.48seconds,peakTorchCUDAallocation772.46MiB(runtimeoverhead separate). NoOOM/nonfinite failure.

## Matched37rowresults

|Method|Exact graphs|Structurally valid|Triple microF1|
|---|---:|---:|---:|
|CharacterTFIDFretrieval|0/37|37/37|22.96%|
|Word/bigramTFIDFretrieval|0/37|37/37|22.04%|
|T5fullgraph|0/37|1/37|2.49%|

Malformed predictions are retained as empty predicted triple sets in strict graphF1;all37references remain in the denominator. Full raw outputs retained privately. Separate serializedT5subtokenerror75.20%(9,222edits/12,264reference subtokens)also includes malformed text;this is not glossWERor semantic accuracy. Errorcategories:15repeatedarticulationID,21danglingedgeindex,1validgraph. No generationcap hits. No model superiority or promotion.

Training fit does not transfer to this structure-heldout dataset under these settings. Arbitrary instance/HNSindex predictions,graph referential consistency,sparse structure supervision andmachine-translated inputs remain possible contributors. Their causal impact is not isolated here. A passed8examplegate does not prove the186rowtraining is wellfit or the annotation convention is predictable fromFrench. This run does not rule out other structured decoders or training schemes.

## Artifacts and verification

Runtime:.runtime/geoquery-full-graph-v1/{MANIFEST,BEST,history,RESULT,ERROR_ANALYSIS}.json;adapter/,tokenizer/,hns_dictionary.json;development_predictions.json;last-state.pt. Model/optimizer/Python/Torch/CUDA RNGstate saved everyepoch. Epoch4state loaded successfully withweights_only=TrueonCPU;full resumed-training equivalence untested. Completed runs now refuse resume rather than silently re-evaluating. Seven codec/scorer tests passed;compilecheckpassed. All source IDs remain private for attribution/removal.

Backup:/home/jmal/Dataset/geoquery-lsfb-graph-feasibility-v1/full-graph-v1/with file hashes. All originalsourcegraph inconsistencies remain quarantined;no guessed repairs. NoDriveupload claim.

## Reproduction

```bash
.runtime/t5-local-venv/bin/python -m training.geoquery_lsfb.fit_gate --output .runtime/geoquery-graph-fit-reproduction --steps 600
.runtime/t5-local-venv/bin/python -m training.geoquery_lsfb.train_graph --gate .runtime/geoquery-graph-fit-reproduction --output .runtime/geoquery-full-graph-reproduction
.runtime/t5-local-venv/bin/python -m training.geoquery_lsfb.analyze_graph_errors
.runtime/t5-local-venv/bin/python -m pytest research/code/tests/test_geoquery_graphs.py -q
```

Erroranalyzer currently targets the namedv1run;adapt its path for a reproduction. Resume unfinished training withsame--gate/--output/configand--resume;do not use completedrun. Thelosslesscodec requires originaldatasetprovenance/license. NoheldoutgraphinferenceorT5weights should be described as approved signing.

## Next research direction

Prioritize source alignment/reviewer inspection and a structured prediction approach that can enforce unique vertices and validedge references. Structural constraints must be assessed separately from lexical/semantic correctness;they cannot be advertised as an adequacy fix. Existing retrieval is the measured baseline and alsogets0completegraphs. Any further neuralexperiment should preserve this development history,protect a new evaluation split where possible,and keep original human-signed versus synthetic-input provenance explicit.
