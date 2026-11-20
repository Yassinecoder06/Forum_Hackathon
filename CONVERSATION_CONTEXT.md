# SignaCulture conversation context

## Project

SignaCulture is an experimental signing-avatar pipeline. The original proposal was to extract body, hand, and face keypoints from Dicta-Sign videos, build a lexical gloss-to-motion lookup, and animate a 3D avatar. French text-to-gloss, motion transitions, and audio-driven rhythm and emotion were separate research directions. France-LSF (`fsl`) and Belgian LSFB (`sfb`) are separate datasets and evaluation tracks.

The recent work focused on computer vision and the review interface. Keep the existing research artifacts and workspace changes in place. The user preferred direct implementation after earlier agentic work and asked to avoid retesting sign-to-gloss research during the web and Docker work.

## Current review site

- Public HTTPS [comparison](https://cv.yassinejemmali.me/) uses the existing Cloudflare tunnel and opens without a sign-in prompt, per the user’s request. The tunnel routes to a loopback origin on port 8095. The public browser loads source recordings directly from Dicta-Sign and receives derived landmarks from SignaCulture.
- The [local review](http://127.0.0.1:8094/) uses cached source MP4 files and can fall back to local WebM copies.
- The selected dataset is the first **100 France-LSF clips in the cached Dicta-Sign inventory**. A searchable, horizontally scrolling word bar sits above the model panels. Playback, frame steps, and scrubbing use a single source timeline. Body, left hand, right hand, and face overlays can be toggled.
- Each panel now has a synchronized Noa avatar beneath the source overlay. The four whole-body previews use their own keypoints and an assumed depth plane. They do not retarget facial expressions. The HaMeR preview uses recovered wrist-relative finger depth and MediaPipe upper-body tracking.
- Local webcam preview is separate at [port 5180](http://127.0.0.1:5180/webcam/). It has live landmark layers, a rolling detection-coverage readout, recording and exports. Camera preview starts as soon as video arrives while tracking models load. Chromium tests used a prerecorded simulated camera. The physical webcam and one-time permission prompt were not verified during those checks.

## Model comparison

| Panel | Inference and scope |
| --- | --- |
| MediaPipe Holistic | Cached body, hands and face, with separate full-frame and wrist-crop hand recovery in the offline extractor |
| MMPose DWPose-t | OpenMMLab 133-point ONNX model on CPU through rtmlib. Uses the full source frame as the single-person region |
| Sapiens 0.3B | Meta's 308-point Goliath TorchScript checkpoint on the RTX 4060. Uses BF16 autocast and the full source frame as the single-person region |
| SDPose-Wholebody | Authors' 133-point checkpoint on CUDA BF16. Uses the full frame, a fixed VAE noise seed, one UNet step, and the release's instance-normalized decoder |
| HaMeR | Official hand-only model on the RTX 4060. Uses MediaPipe-guided crops and handedness, then projects its reconstructed hand joints back into the source frame |

All five have results for the 100 clips. The four added models align on **2,361 sampled source frames**, with matching MediaPipe frames available. This does not mean every frame has a valid hand detection. HaMeR recovered at least one hand in 1,850 frames, both in 1,523, and neither in 511. Its dependency on MediaPipe crops limits any independent comparison of hand-detection recall.

The user supplied the licensed MANO file at `.runtime/pose-comparison/models/hamer/MANO_RIGHT.pkl`. We verified it and the public HaMeR checkpoint against the release hashes before running inference. The managed Colab SSH connection was unavailable during this work, and local inference succeeded.

Public Chromium checks covered the selector, synchronized overlays and avatars, including the fifth HaMeR avatar. These are engineering and visual checks without annotated ground truth. Landmark counts and confidence scores use different conventions across models. They cannot establish which model best preserves sign-language handshape or intelligibility.

## Latest MediaPipe change

We inspected [sign/translate at 928d49f](https://github.com/sign/translate/tree/928d49ffe223fccedca64162851d4951b9152ecc). It uses browser MediaPipe Holistic for tracking and a TensorFlow.js signing-activity classifier based on motion of body landmarks. Its text-to-sign output comes from a separate cloud pose endpoint. The frontend does not expose the backend training or generation method. The activity classifier does not recover missing fingers.

We adapted its temporal shoulder normalization to stabilize our motion exports. Both the offline video worker and webcam exporter now use a rolling median of up to seven visible shoulder measurements. Samples expire after 300 ms. This changes the pixel-to-rig scale without changing the raw MediaPipe detections. On six before/after Dicta-Sign clips and 345 adjacent frame pairs, median absolute scale change fell from **0.150% to 0.132%**. The 95th percentile moved from 1.141% to 1.138%, and raw body, hand, and face results matched in the paired runs. Eleven backend capture tests and seven webcam tests passed. The local webcam build was restarted. The public comparison still uses the cached landmark tracks.

## Earlier project decisions and research

- The initial workflow requested worktrees and multiple agents. The user later preferred direct implementation and briefly allowed only the second Codex pane with Sol 6.1 medium. Those were task-specific instructions, not a requirement for ordinary follow-up work.
- The French-to-gloss experiments investigated different T5 sizes, Colab and local GPU training, transfer from Text-to-Gloss NMT and related research, and data compatibility. The user identified the quality and scarcity of aligned supervision as concerns. The conversation mentioned a 39-pair France-LSF subset. Such a small set is suitable for feasibility checks. Use project reports and their experiment artifacts for exact training counts and scores.
- The agreed Belgian follow-up was a narrow LSFB experiment with consistent labels, repeated instances, context, conversation-disjoint evaluation, and a retrieval baseline. Use tagged Belgian data for that task. The later France-only brief is a separate experiment.
- The France-LSF research brief called for SignVerse-2M metadata-first auditing, genuine French-to-gloss targets for mT5, a separate subtitle-to-pose track, interpolation baselines before a conditional GAN, and held-out evaluation. Subtitle text is weak supervision. AZee annotations and verified glosses have distinct representations. Continuous-motion inbetweening and lexical coarticulation also require separate evaluation.
- Earlier work included Docker packaging, deployment to the homelab, research retrospectives, a demonstration video and prototype link, and branch review. The user reported a Docker studio and webcam push to `origin/feat/docker-studio` at `f956461` from another session. Treat that as historical context and check Git for the current state.

## Where to resume

- [Review setup and inference commands](docs/test-sites/cv-comparison.md) and the [test-site index](docs/test-sites/README.md)
- Viewer code moved to the sibling folder `~/Projects/Active/Forum_Tests/` (`cv_comparison/server.py`, `review.js`, `index.html`, `extract_pose.py`, `run_hamer.py`, `sdpose.py`)
- Shared pose and HaMeR loaders: [motion_worker/pose_models.py](motion_worker/pose_models.py) and [motion_worker/hamer.py](motion_worker/hamer.py)
- [Offline MediaPipe worker](backend/app/workers/capture.py) and [webcam motion exporter](webcam/src/capture.mjs)
- `data/source_inventory/dictasign_lsf.json` selects the clips. Cached source media and MediaPipe data live in `.runtime/captures/`. The other model outputs live in `.runtime/pose-comparison/results/`.
- User services `signaculture-dictasign-review`, `signaculture-cv-origin`, and `signaculture-webcam` serve the two review origins and local webcam. The system `cloudflared` service handles the public hostname.
- At the time of this handoff, the comparison directory is untracked and the two MediaPipe exporters have local modifications. The handoff does not imply a commit or push.

The most direct follow-up for MediaPipe is to annotate a small set of difficult source frames for missing hands, wrong-side assignments, and bad wrist anchors. Compare the current fallback and crop choices on those same frames before changing the default detector or making accuracy claims.
