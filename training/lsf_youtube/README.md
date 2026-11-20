# French subtitle and LSF pose collection

The acquisition worker enumerates videos from the public Messe LSF uploads
listing. It checks each video's caption metadata, prefers an available
manual French JSON3 track and otherwise accepts the original French
automatic track. It downloads video only after obtaining a usable French
subtitle event stream. It writes one status file per checked upload and
reserves 12 GiB of free disk for the running workspace. Incomplete media
downloads are resumable.

The pose worker consumes completed downloads with MediaPipe Holistic and
the 157-point layout from the Matignon notebook. FFmpeg decodes H.264 and
AV1, resamples at 12.5 Hz and can apply a manually reviewed interpreter ROI.
Messe LSF uses full-frame extraction by default. For the three meeting
broadcasts, the pilot crops were taken from the visible interpreter windows
and checked on sample screenshots. Full broadcasts still need crop review
around layout changes, introductions, and interpreter handoffs.

Each extraction writes:

- An NPZ with `pts[T,157,3]`, `has[T,4]` for body, left hand, right hand
  and face, and `t_ms[T]` on the source timeline. XYZ is image-normalized
  within the recorded ROI. The Z coordinates are monocular.
- JSONL subtitle windows with source start/end, French caption text,
  half-open frame indices, coverage and a candidate flag. The flag
  requires a complete window, at least five pose samples, 85% detected
  body coverage, at least one hand on 50% of samples, and three words.
- A summary with source identity, available video digest, crop, caption
  provenance and aggregate coverage.

These are weakly paired French subtitle windows and LSF motion, with the
nominal subtitle clock as reference. Interpreter lag and French/LSF
reordering can vary across a recording. Candidate status does not establish
translation quality, individual gloss boundaries, or dataset redistribution
rights. Retain the source and signer/recording provenance, review time
alignment and split by signer or recording before model evaluation.

The local ingest stages use `training.lsf_youtube.collect acquire` and
`training.lsf_youtube.collect extract`. The download environment needs
current yt-dlp with Node for YouTube support. The pose environment requires
FFmpeg, NumPy, MediaPipe Tasks 1.2.3, `HOLISTIC_MODEL`, and notebook 07's
`extract_poses.py` at `.runtime/matignon-cv-review/extract_poses.py`.
The per-video `training.lsf_youtube.align` entry point also accepts a
time range and normalized `--roi x0 y0 x1 y1`.

Runtime artifacts are under `.runtime/youtube-lsf/`. `validated-pilot`
contains the six checked extraction samples. `channel-dataset` contains
completed channel outputs. `download-status` and `progress.json` track
the ongoing collection. `broadcast-dataset` holds full-length broadcast
outputs as they complete.

## Broadcast viewer

[Relecture LSF on port 8100](http://127.0.0.1:8100/) pairs French subtitle
windows with video playback and the corresponding MediaPipe frames. It
supports searchable subtitles, single-window playback, looping, playback
speed, and a full-frame or interpreter-only view. Coverage filtering uses
the same candidate flag as the extraction output.

```bash
python -m training.lsf_youtube.viewer_server --wait-for-results --port 8100
```

The viewer listens on loopback. It waits for all three full extractions,
then validates the timestamp and frame ranges before opening the port. To
check layout with the short pilot data, use
`SIGNACULTURE_BROADCAST_VIEWER_DATA=validated-pilot` with a different port.
