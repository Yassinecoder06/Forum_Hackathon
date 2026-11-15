# Optional experiments
No optional dependency is imported until execution. Nothing runs or downloads at application startup.

Use a separate CPython 3.10/3.11 environment for the legacy API (newer MediaPipe runtimes may omit `mp.solutions`):

```sh
python3.11 -m venv .venv-motion
.venv-motion/bin/pip install -r experiments/motion_extraction/requirements-optional.txt
.venv-motion/bin/python experiments/motion_extraction/extract.py authorized.mp4 landmarks.json
.venv-motion/bin/python experiments/motion_extraction/convert.py landmarks.json motion.json
python -m pytest experiments/motion_extraction/test_motion.py
```

Pinned optional versions document the intended runtime; real video extraction has not been exercised here. Use only authorized video. Extraction v2 applies the same hip origin and shoulder scale to pose and hands, without mirroring or exchanging anatomical sides. Y and Z signs are inverted to the player convention. Shoulder width is assumed 0.4 m. MediaPipe hand depth is wrist-relative while pose depth is hip-relative: applying the shared arithmetic transform does not calibrate these depth origins, so hand Z and wrist alignment remain approximate and require review. No calibrated metric reconstruction is claimed.

The converter rejects legacy v1 files, nonfinite coordinates, invalid FPS/duration, unordered timestamps and empty sequences. Extraction records missing/degenerate frames and fails when none are usable. Missing frames are omitted; playback interpolation across gaps can be misleading. The output contains the nine supported player joints; hand wrist landmarks replace pose wrists when available. Finger landmarks remain in the intermediate artifact pending an agreed finger rig. Register the resulting file in the motion manifest with `synthetic:false`, pending animation/linguistic review and independently verified rights before local preview; the converter does not register or approve assets automatically.

Animics compatibility: the intermediate JSON and signaculture-v1 position fixtures are project formats, not native Animics Script Mode or Keyframe Mode files. Import requires an explicit mapping to the chosen avatar skeleton, bone rotations, finger articulation and exporter, followed by geometric and linguistic review. No Animics import has been validated.

`python3 experiments/emotion2vec/infer.py voice.wav --model-dir /existing/local/model --output scores.json`
requires optional FunASR and its model dependencies. Supply an already provisioned complete local emotion2vec FunASR directory; no remote model ID is accepted and no weights are downloaded by this adapter. For strictly offline execution also disable networking: dependency behavior is outside this adapter. Raw real model output is preserved separately from the API AudioFeatures contract, whose emotion_hypothesis remains null. No inference has been executed in this environment and no scores are fabricated.
