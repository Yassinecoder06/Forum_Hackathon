# SignaCulture project progress

Snapshot checked November 19, 2026.

## Main Studio

- Docker Compose serves the [Studio](http://127.0.0.1:8088/studio/) and [standalone webcam](http://127.0.0.1:5188/). Both containers were healthy when checked.
- The Studio integrates webcam capture under **Capturer → Capturer avec la webcam**. A browser recording can be handed to the capture form, which uploads it after rights confirmation. Landmark JSON remains available when video recording is unavailable. The decoder handles webcam WebM files without frame-count metadata.
- Video captures use **Sapiens 0.3B + HaMeR** through a private host CUDA worker. MediaPipe supplies facial measurements, the raw Animics export and fallback observations. The worker runs Sapiens and HaMeR in separate passes to fit the local machine. A Chromium test using prerecorded webcam input completed the full round trip with **69 Sapiens frames and 138 HaMeR hand reconstructions**.
- The Studio currently uses **mathematical interpolation** for joins between recorded signs. Chromium verified playback of a two-sign sequence after the Docker rebuild. Docker and worker setup are documented in [deployment/DOCKER.md](deployment/DOCKER.md).

## Computer-vision previews

| Port | Preview |
|---|---|
| [8094](http://127.0.0.1:8094/) and [8095](http://127.0.0.1:8095/) | Full 100-clip pose comparison with MediaPipe, MMPose DWPose-t, Sapiens, SDPose and HaMeR |
| [8096](http://127.0.0.1:8096/) | MediaPipe 1.2.3 + HaMeR with Noa, hand meshes and facial expressions |
| [8097](http://127.0.0.1:8097/) | Sapiens 0.3B + HaMeR with Noa and hand meshes, 100 clips |
| [8098](http://127.0.0.1:8098/) | Mathematical versus learned transitions on three example sign sequences |
| [8099](http://127.0.0.1:8099/) | Matignon notebook 07, full-frame MediaPipe Holistic with Noa, 100 clips |

All six preview endpoints responded during this snapshot. The review viewer selects the same extraction timestamp for each overlay and its avatar. The combined previews can estimate head rotation from facial landmarks and bridge short wrist tracking gaps. The tracking readout distinguishes estimates from hand observations. DeepSeek's shared renderer changes add a forearm-aligned neutral fallback for missing hands. Noa can still show an estimated hand during occlusion.

### Completed tracker comparison

The two previews were matched on **100 clips and 2,361 source timestamps**. Notebook 07 was run with its interpreter crop disabled.

| Coverage | Sapiens + HaMeR | Notebook 07 |
|---|---:|---:|
| Both hands available | **2,141 / 2,361 (90.7%)** | 1,386 / 2,361 (58.7%) |
| Both shoulders, elbows and wrists available | 2,263 / 2,361 (95.8%) | **2,361 / 2,361 (100%)** |

Sapiens + HaMeR had higher two-hand coverage on 88 clips, tied on nine, and had lower coverage on three. All 2,141 two-hand results included two HaMeR meshes. This comparison measures coverage. Finger-pose accuracy against ground truth remains unmeasured. [Per-clip CSV](.runtime/full-cv-comparison/per-clip.csv)

### Full local benchmark in progress

The benchmark inventories **1,490 capture files**, including 1,025 Dicta-Sign LSF files and 465 other local captures. These represent **1,358 unique videos** after byte-identical deduplication. Both pipelines are scored on every second source frame. The report retains one row per file and records decode failures and sample mismatches.

At this snapshot, notebook 07 had written **949** unique-video results and Sapiens had written **243**. The HaMeR pass follows Sapiens. The systemd user service `signaculture-cv-benchmark-all` was active. It writes `summary.json` and `per-video.csv` to [.runtime/all-video-cv-benchmark](.runtime/all-video-cv-benchmark/) when the stages finish.

## Anatomy-aware motion normalization

The opt-in [PyTorch normalization package](normalization/README.md) adds invertible torso transforms, robust shoulder scaling, confidence-aware temporal filtering, fixed-length skeletal reconstruction, contact candidates, and optional IK against Noa's measured skeleton. Its native adapter connects to the transition training manifest and the existing avatar motion format. It requires explicit shared XYZ for 3D fitting.

- **17 normalization tests passed**, including CPU/CUDA gradient checks and synthetic hand-to-hand, chin, forehead and chest contacts.
- **440 native clips and 17,181 frames** passed through the loader across the existing transition splits. The splits group source videos, and signer-disjoint recognition quality remains unverified.
- In the 32-frame Bonjour example, IK reduced mean wrist target error from **0.432 to 0.091 shoulder widths** compared with direction-only reconstruction. The exported clip preserved its 2.44-second duration and facial data. Chromium rendered the export with Noa and kept a missing hand marked unobserved. This measures fit to the extracted targets. Real hand-to-face contact still needs calibrated 3D anchors. [Animation](.runtime/anatomy-normalization/bonjour/comparison.gif) · [Noa render](.runtime/anatomy-normalization/bonjour/noa-avatar.png) · [Motion export](.runtime/anatomy-normalization/bonjour/noa-motion.json)

## Learned transition experiment

The local [exp15 checkpoint](.models/lsfb-transition/transition_model_exp15.pt) loaded and passed an inference smoke test. Among the logged LSFB runs, exp15 had the lowest validation MSE, **0.006742**. Its reported test MSE, **0.006105**, was **30.9% below** the smoothstep baseline. A fresh Colab ranking remains pending because the private Drive mount required interactive authorization and the connector could not stream the 802 MB dataset.

The port 8098 comparison keeps the source clips, endpoints and 480 ms joins identical across panels. It compares the Studio stitcher with the adapted model. The checkpoint was trained on LSFB, and performance on France-LSF transitions is unverified.

## Git and deployment

The Docker baseline was pushed to `feat/docker-studio` in `f956461`. The homelab snapshot merge was pushed to `main` in `e894c7c`. The live local Docker build also contains later workspace changes. This snapshot describes the local running stack and does not imply all later work is on `main`.
