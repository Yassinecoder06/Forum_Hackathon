# Broadcast viewer (port 8100)

[Relecture LSF](http://127.0.0.1:8100/) pairs French subtitle windows from
three public LSF-interpreted broadcasts with video playback and the
corresponding MediaPipe frames. It supports searchable subtitles,
single-window playback, looping, playback speed, and a full-frame or
interpreter-only view. Coverage filtering uses the same candidate flag as
the extraction output. Collection and alignment are documented in
[research/code/training/lsf_youtube/README.md](../../research/code/training/lsf_youtube/README.md).

Code: `~/Projects/Active/Forum_Hackathon/apps/forum-tests/broadcast_viewer/`. Data:
`.runtime/youtube-lsf/` in this repository.

```bash
cd ~/Projects/Active/Forum_Hackathon/apps/forum-tests
python3 -m broadcast_viewer.server --wait-for-results --port 8100
```

The viewer listens on loopback. It waits for all three full extractions,
then validates the timestamp and frame ranges before opening the port. To
check layout with the short pilot data, use
`SIGNACULTURE_BROADCAST_VIEWER_DATA=validated-pilot` with a different port.
