# Motion Lab demo

Port 8104. Code and generated assets: `~/Projects/Active/Forum_Hackathon/apps/forum-tests/direct_fsl_demo/`. It reads experiment outputs from `research/code/signaculture-direct-fsl/outputs/` in this repo. Regenerate the static assets and serve them (normally the `signaculture-direct-fsl-demo` unit does this):

```bash
cd ~/Projects/Active/Forum_Hackathon/apps/forum-tests
python3 -m direct_fsl_demo.prepare_demo
python3 -m http.server 8104 --bind 127.0.0.1 --directory direct_fsl_demo
```

Open http://127.0.0.1:8104/. The page uses the saved expanded checkpoint’s exploratory prediction and both held-out comparison videos from the retrieval evaluation. FFmpeg is required to generate poster frames. Derived motion assets and subtitle excerpts stay local and are excluded from Git.

Training examples are generated separately from exact captions and pose slices in the original 102-cue checkpoint manifest. From the experiment directory `research/code/signaculture-direct-fsl/` in this repo:

```bash
FT=~/Projects/Active/Forum_Hackathon/apps/forum-tests
PYTHONPATH=$FT python $FT/direct_fsl_demo/render_training_examples.py --manifest ../.runtime/direct-fsl-expanded/cues.jsonl --checkpoint checkpoints/expanded/best.pt --features ../.runtime/training-demo/features.pt
(cd $FT && python3 -m direct_fsl_demo.prepare_demo)
```

The render command checks the manifest hash and exact caption text before showing the training label. Each comparison uses the observed playback duration and displays the checkpoint's independently predicted duration separately.
