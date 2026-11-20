# Limitations

- The held-out test contains just two source recordings. Signer identity is unavailable. Source-video separation does not establish signer separation.
- Released French captions provide coarse timing and unclear transcription or translation provenance. There is no independent caption–sign alignment review. These pairs are unsuitable as gold text-to-gloss supervision.
- Released poses are 2D image observations. Normalization uses the shoulders and retains visibility masks. Source body scores encode landmark availability rather than calibrated detector confidence. The pipeline does not infer physical depth.
- All cues are resampled to 32 relative phases. Duration is predicted separately. Velocity and acceleration measurements therefore describe normalized phase. They do not establish physical timing.
- Both pilots use small cross-attention decoders inspired by SignDW. They do not reproduce the paper's autoregressive 40M model. An executable official training release and checkpoint were not verified.
- No qualified France-LSF comprehension review or browser-based 3D avatar playback was completed. The optional adapter provides experimental planar input to the existing landmark import schema.
- Next, audit caption alignment and transcription provenance on the training and held-out sets, add source-disjoint recordings through a second bounded download, and check correctly paired versus mismatched captions against the caption-independent template before investing in model scale. Report results by source recording.
