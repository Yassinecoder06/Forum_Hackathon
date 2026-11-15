# Model card — research only

First completed run: `t5-small-cpu-smoke`, base `google-t5/t5-small` revision `df1b051c49625cf57a3d0d8d3863ed4d13564fe4` (Apache 2.0). LoRA q/v rank8 alpha16 dropout.05, AdamW lr3e-4, batch1 accumulation4, seed42, two epochs FP32 CPU. No custom gloss vocabulary. Checkpoint chosen by validation loss, test decoded once afterward. Run metadata includes checkpoint SHA256 and dataset manifest SHA256. Weights/environments ignored and never committed.

Training data: CC BY-SA 4.0 STK source annotations, free French translation and FORM target, pending local linguistic review. Published source annotation does not equal local approval. No nonmanual output channels, spatial grammar synthesis or motion generation. Derived data retains attribution/share-alike terms; trained weights stay local pending downstream review of distribution obligations.

Training8.2657s; total27.0761s. Validation loss4.5381→4.3329. Test: exact match0, tokenWER2.0728155, corpusBLEU0 (0–100 scale), predicted token OOV vs training1.0. Retrieval tokenWER1.1116505. This model is worse than retrieval and mostly copies French input. It is an engineering smoke test, unsuitable for translation deployment. CUDA/VRAM not measured for this CPU run; values null. Full report: runs/t5-small-cpu-smoke/metrics.json.

Use `training/.venv-smoke/bin/python training/infer.py --text 'Bonjour à tous' --model training/checkpoints/t5-small-cpu-smoke --device cpu`. CLI returns `candidate:true`, `linguistic_review:PENDING_LOCAL_REVIEW`, `translation_validated:false`, `render_allowed_for_publication:false`. API integration is lead-owned. CPU/CUDA are configurable; CPU actually tested.
