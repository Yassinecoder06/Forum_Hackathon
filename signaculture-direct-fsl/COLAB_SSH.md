# Colab execution

Use a private user-owned runtime approved for the derived data. The pilot transferred 22 MiB of normalized pose archives, cue metadata, and code. It used a 30 MiB pose transfer budget and targeted at most 8 GiB of PyTorch memory. Public model weights for the frozen-encoder variant are separate.

From the repository root, package only the referenced private pose files, the exact cue manifest, and project source.

```bash
python3 - <<'PY'
from pathlib import Path
import json, tarfile
manifest = Path('data/manifests/signverse_fsl_caption_pose.jsonl')
records = {Path(json.loads(line)['pose_path']) for line in manifest.read_text().splitlines()}
project = Path('signaculture-direct-fsl')
code = {p for p in project.rglob('*') if p.is_file() and p.suffix in {'.py','.yaml','.txt'}}
with tarfile.open('.runtime/direct-fsl-input.tar.gz', 'w:gz') as output:
    for path in sorted(records | code | {manifest}):
        output.add(path)
PY
colab new -s signaculture-direct-fsl --gpu T4
colab ssh -s signaculture-direct-fsl -- nvidia-smi
colab upload -s signaculture-direct-fsl .runtime/direct-fsl-input.tar.gz /content/direct-fsl-input.tar.gz
colab ssh -s signaculture-direct-fsl -- bash -lc 'cd /content && tar -xzf direct-fsl-input.tar.gz && cd signaculture-direct-fsl && python scripts/check_gpu.py && python train.py'
colab download -s signaculture-direct-fsl /content/signaculture-direct-fsl/checkpoints/best.pt signaculture-direct-fsl/checkpoints/best.pt
```

Use `configs/frozen.yaml` with `--out checkpoints/frozen` for the frozen French model. Save both best and last checkpoints before losing the remote runtime. Resume with the same configuration, manifest and output directory using `--resume`. Set a larger `steps` target only for a new experiment. The bundled notebook uses the same archive and does not require SSH.

## Rebuild the expanded training set

The first and expanded experiments use the same 66 held-out cues, including matching source intervals and pose checksums. The expanded run adds eight official train recordings from one shard. With `fsl_only.csv` and `scripts/stage_shard.py` copied to the private runtime, stage the pinned shard there:

```bash
hf download SignerX/SignVerse-2M dataset/Sign_DWPose_NPZ_000637.tar --repo-type dataset --revision 0c7e956a8471acad3a9cebaeb66bb23bf5deaafa --local-dir /content/fsl-one-shard
python stage_shard.py /content/fsl-one-shard/dataset/Sign_DWPose_NPZ_000637.tar fsl_only.csv /content/direct-fsl-expansion
tar -C /content -czf /content/direct-fsl-expansion.tar.gz direct-fsl-expansion
```

Retrieve the resulting private archive with `colab download`. From the local repository root:

```bash
tar -xzf .runtime/direct-fsl-expansion.tar.gz -C .runtime
mkdir -p .runtime/direct-fsl-expanded
cp -a --update=none data/poses/private/raw/* .runtime/direct-fsl-expansion/
python3 -m src.data.prepare_signverse_samples --raw .runtime/direct-fsl-expansion --output .runtime/direct-fsl-expanded/poses --audit .runtime/direct-fsl-expanded/quality.json --preview .runtime/direct-fsl-expanded/preview.html
python3 -m src.data.prepare_signverse_caption_pose --raw .runtime/direct-fsl-expansion --poses .runtime/direct-fsl-expanded/poses --output .runtime/direct-fsl-expanded/cues.jsonl
```

Bundle the new manifest and only its referenced poses using the packaging recipe above. The actual expanded input bundle was 48,055,123 bytes. In the Colab project directory:

```bash
python train.py --config configs/frozen.yaml --manifest .runtime/direct-fsl-expanded/cues.jsonl --out checkpoints/expanded
python evaluate.py --manifest .runtime/direct-fsl-expanded/cues.jsonl --checkpoint checkpoints/expanded/best.pt --out outputs/expanded
python predict.py --manifest .runtime/direct-fsl-expanded/cues.jsonl --checkpoint checkpoints/expanded/best.pt --text "Bonjour, bienvenue au musée." --output outputs/expanded/demo_001
```

## Evaluate additional held-out recordings

Repeat the bounded download and staging recipe for `Sign_DWPose_NPZ_000644.tar` and `Sign_DWPose_NPZ_000667.tar`. Use `--split test --max-archive-bytes 576716800` with the same staging output directory, and prepare an independent cue manifest under `.runtime/direct-fsl-holdout/cues.jsonl`. The two shards total 1,099,141,120 bytes. Only the derived poses and caption files for eligible test recordings are staged.

Package the unchanged expanded training manifest, additional test manifest, all referenced poses, the existing best checkpoint and project source. From the project directory:

```bash
python evaluate.py --checkpoint checkpoints/expanded/best.pt --manifest .runtime/direct-fsl-expanded/cues.jsonl --evaluation-manifest .runtime/direct-fsl-holdout/cues.jsonl --out outputs/holdout
python scripts/prepare_alignment_review.py --manifest .runtime/direct-fsl-expanded/cues.jsonl --additional .runtime/direct-fsl-holdout/cues.jsonl --output outputs/alignment-review
```

If the pose bundle exceeds the transfer allowance, send only a JSON mapping from cue ID to French caption for the original validation, original test, and additional test sets. Run `python scripts/encode_evaluation_captions.py captions.json features.pt` in Colab with the same project encoder. Bring the features back privately and add `--french-features PATH` to the local evaluation command. The evaluator checks that the cached cue IDs and captions match the manifests.

## Reproduce the local data-scale study

Run from the experiment directory with the private 102- and 179-cue manifests and their cached frozen encoder features. Set `seed` to 17, 29 or 43, and `training_manifest` to `.runtime/direct-fsl-expanded/cues.jsonl` or `.runtime/training-demo/expanded-cues.jsonl`. Use a separate output directory per pair.

```bash
sed "s/^seed: 17$/seed: $seed/" configs/frozen_500.yaml > "../.runtime/training-demo/frozen-500-seed-$seed.yaml"
python train.py --config "../.runtime/training-demo/frozen-500-seed-$seed.yaml" \
  --manifest "$training_manifest" --french-features ../.runtime/training-demo/features-expanded.pt \
  --out "$checkpoint_dir"
python evaluate.py --metrics-only --checkpoint "$checkpoint_dir/best.pt" \
  --manifest "$training_manifest" --support-manifest .runtime/direct-fsl-expanded/cues.jsonl \
  --evaluation-manifest .runtime/direct-fsl-holdout/cues.jsonl \
  --french-features ../.runtime/holdout-text/features.pt --out "$evaluation_dir"
```

Both dataset sizes use the same landmark support and held-out recordings. Select the checkpoint using validation only.

For the template-residual experiment, use the 102-cue manifest and substitute `configs/residual_500.yaml` for `configs/frozen_500.yaml` in the seed-configuration command. Choose new checkpoint and evaluation directories. The pose template is calculated solely from the training split and is saved inside the checkpoint.
