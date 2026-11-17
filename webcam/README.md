# Standalone webcam capture

Open **http://127.0.0.1:5180/** and select **Activer la caméra**. This is a
separate app; the dashboard and its API are unchanged.

- Live body, fingers and face overlays, anatomical hand indicators and 52 face
  blendshape coefficients. These coefficients are measurements, not emotion labels.
- CPU processing by default; optional GPU mode with CPU fallback.
- Extra face/wrist crops recover small detections missed in the full frame;
  crop coordinates are mapped back to the camera image and marked in raw data.
- Mirrored preview only: exported coordinates retain anatomical sides.
- Record up to 30 seconds, scrub/replay the landmarks, and download source video.
- Convert to `webcam-motion.json`, importable through SignaCulture's native JSON
  motion import. Missing hands remain missing; internal tracking gaps are retained.
- Export the full raw capture, or Animics-style landmarks with **millisecond**
  `dt` and converted world axes. Use native motion export for the existing
  SignaCulture importer, whose older Animics adapter assumes seconds.

Camera permission is requested on activation. No microphone is requested.
Frames are processed in a browser worker, never uploaded or automatically saved.
Models and WASM are served locally after explicit setup. Closing/reloading the
page discards the take unless downloaded. Webcam access requires localhost or HTTPS.

```bash
npm --prefix webcam ci
npm --prefix webcam run setup
npm --prefix webcam run dev       # port 5180, loopback only
npm --prefix webcam run build
npm --prefix webcam run preview   # production build, same port
npm --prefix webcam test
```

The setup command downloads the three pinned Google MediaPipe models and copies
the matching `@mediapipe/tasks-vision@0.10.32` WASM files. Download origins and
SHA-256 hashes are saved in `public/models/manifest.json`. Large generated assets
are ignored by Git. The classic worker bundle is generated before dev/build
because MediaPipe's WASM loader uses `importScripts`.

The implementation follows the capture approach and raw field conventions of
[Animics](https://github.com/upf-gti/animics/blob/master/src/Mediapipe.js), using
Google's Pose, Hand and Face Landmarker APIs. It does not include Animics' full
animation editor, rig retargeter or BVH/GLB export. The converted JSON contains
approximate monocular joint positions; signing accuracy requires review.

Verified with a prerecorded fake camera, never the physical webcam: live pose/
hands/face and 52 blendshapes, recording, all four downloads, replay, camera track
shutdown, mobile width, and zero page errors/external requests. The CPU test
captured 22 frames in about four seconds, with 173 ms median inference time in
headless Chromium. Real hardware performance varies. Evidence is in
`.runtime/webcam-check/`. Native output also passed the backend motion validator.

Browser regression (requires the root `e2e` Playwright dependencies):

```bash
CHROME_PATH=/path/to/chrome WEBCAM_TEST_VIDEO=/path/to/prerecorded.y4m \
  node webcam/scripts/browser-test.mjs
```

The local `signaculture-webcam.service` serves the production build independently
of the dashboard. Check/restart it with `systemctl --user status/restart
signaculture-webcam.service`. Rebuild before restarting after source edits.
