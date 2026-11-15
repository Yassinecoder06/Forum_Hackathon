# Optional experiments
No optional dependency is imported until execution. Nothing runs or downloads at application startup.

`python3 experiments/motion_extraction/extract.py authorized.mp4 landmarks.json`
requires OpenCV and a MediaPipe release exposing `mp.solutions.holistic` (legacy API). Use only videos you have permission to process. MediaPipe may fetch its bundled model resources on first use; provision resources offline where required. Missing pose frames and hand occlusions remain explicit. Hand points remain in MediaPipe image coordinates; pose normalization assumes a 40 cm shoulder width and cannot recover calibrated depth. This is an intermediate artifact, not a rig motion asset. No automatic retargeting, finger reconstruction or LSF approval is claimed.

Animics compatibility: the intermediate JSON and signaculture-v1 position fixtures are project formats, not native Animics Script Mode or Keyframe Mode files. Import requires an explicit mapping to the chosen avatar skeleton, bone rotations, finger articulation and exporter, followed by geometric and linguistic review. No Animics import has been validated.

`python3 experiments/emotion2vec/infer.py voice.wav --model-dir /existing/local/model --output scores.json`
requires optional FunASR and its model dependencies. Supply an already provisioned complete local emotion2vec FunASR directory; no remote model ID is accepted and no weights are downloaded by this adapter. For strictly offline execution also disable networking: dependency behavior is outside this adapter. Raw real model output is preserved separately from the API AudioFeatures contract, whose emotion_hypothesis remains null. No inference has been executed in this environment and no scores are fabricated.
