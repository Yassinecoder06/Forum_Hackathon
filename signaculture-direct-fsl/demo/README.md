# Motion Lab demo

From this directory, regenerate the local static assets and start the demo:

```bash
python3 prepare_demo.py
python3 -m http.server 8104 --bind 127.0.0.1
```

Open http://127.0.0.1:8104/. The page uses the saved expanded checkpoint’s exploratory prediction and both held-out comparison videos from the retrieval evaluation. FFmpeg is required to generate poster frames. Derived motion assets and subtitle excerpts stay local and are excluded from Git.

Training examples are generated separately from exact captions and pose slices in the original 102-cue checkpoint manifest. From the experiment directory:

```bash
python demo/render_training_examples.py --manifest ../.runtime/direct-fsl-expanded/cues.jsonl --checkpoint checkpoints/expanded/best.pt --features ../.runtime/training-demo/features.pt
python demo/prepare_demo.py
```

The render command checks the manifest hash and exact caption text before showing the training label. Each comparison uses the observed playback duration and displays the checkpoint's independently predicted duration separately.
