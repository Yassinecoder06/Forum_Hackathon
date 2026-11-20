# Dicta-Sign pose comparison

The viewer compares MediaPipe Holistic, MMPose DWPose-t, Meta Sapiens, and SDPose-Wholebody on the same source frame. It uses the first 100 France-LSF clips in the cached project inventory.

The fifth panel uses [HaMeR](https://geopavlakos.github.io/hamer/) for hand reconstruction. Its [official setup](https://github.com/geopavlakos/hamer) requires `MANO_RIGHT.pkl` from the [registered MANO download](https://mano.is.tue.mpg.de/). The local inference runner publishes projected joints, wrist-relative depth and the projected MANO surface on the same source timestamps as the other runs. The **HaMeR hand surface** toggle switches between the shaded mesh and joint overlay.

- **MediaPipe Holistic** reads the existing Animics-format detection cache, including the project's hand fallback.
- **MMPose DWPose-t** uses the [OpenMMLab ONNX export](https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-t_simcc-ucoco_dw-ucoco_270e-256x192-dcf277bf_20230728.zip), COCO WholeBody's 133-landmark layout, and rtmlib/ONNX Runtime. Code and model provenance are in the [MMPose checkout](https://github.com/open-mmlab/mmpose) at `759b39c13fea6ba094afc1fa932f51dc1b11cbf9`.
- **Sapiens 0.3B** uses [Meta's official 308-point Goliath TorchScript checkpoint](https://huggingface.co/facebook/sapiens-pose-0.3b-torchscript) at `796d82b26775444bbeb38a8202f8f9783370603f`, licensed CC BY-NC 4.0. Its UDP decoder comes from the [Sapiens Lite checkout](https://github.com/facebookresearch/sapiens/tree/main/lite) at `2cb07227a740cf09896309ea3a3b8fa44429865c`. The checkpoint is 1.32 GB. Use is for local research.
- **SDPose-Wholebody** uses the authors' [133-keypoint release](https://huggingface.co/teemosliang/SDPose-Wholebody) at `0c8153e39458bc8ae87fb68790c141bf0f03c11b`, marked MIT by its model card. The [reference inference](https://github.com/t-s-liang/SDPose-OOD) uses a VAE sample, an empty CLIP prompt, one UNet pass at timestep 999 and an instance-normalized heatmap head. The preview uses a fixed VAE noise seed across frames and the complete source image as the single-person region. The downloaded pose components total about 5.2 GB.

The top-down models use the full frame as a single-person region. They run at a stride of two source frames by default. The four display layers preserve each model's native facial and hand keypoints. The body uses a common subset so lines connect corresponding joints. The review uses no human-annotated ground truth, and score thresholds are not calibrated across models.

## Run locally

The code lives in the sibling folder `~/Projects/Active/Forum_Tests/cv_comparison/`. The shared Sapiens, MMPose and HaMeR loaders it uses live in this repo under `motion_worker/` (the Studio capture worker uses them too). Run the commands below from `Forum_Tests`, with `R=../Forum_Hackathon` pointing at this repo. Normally the servers run as systemd units (`make -C ../Forum_Tests status`).

```bash
cd ~/Projects/Active/Forum_Tests && R=../Forum_Hackathon
$R/.venv/bin/python -m cv_comparison.server
```

Open http://127.0.0.1:8094/. A model panel displays its checkpoint and device when a result exists. Pending clips refresh automatically while open. The viewer uses one video timeline for all three canvases.

The Cloudflare origin runs with `DICTASIGN_PUBLIC_REVIEW=1` and `--port 8095`. In this mode, browsers load videos directly from Dicta-Sign, and the review server delivers the 2D overlays. The local viewer can use its cached MP4 and WebM copies.

All panels share the Noa avatar and SignaCulture retargeter. The four full-body previews use their own model's body and finger tracks, shoulder-width normalization, and fixed depth. HaMeR supplies hand poses and wrist-relative finger depth, with the body driven by MediaPipe. Each avatar uses the exact extraction timestamp selected by its own video overlay during playback and after seeking. If that sample lacks usable upper-body landmarks, the preview returns to its neutral pose. Facial expressions are outside the comparison preview. Build the viewer bundle from this repository's root after installing the frontend dependencies.

```bash
frontend/node_modules/.bin/esbuild frontend/src/dictasignAvatar.tsx \
  --bundle --minify --format=esm --jsx=automatic \
  '--define:process.env.NODE_ENV="production"' \
  --outfile=.runtime/dictasign-review/avatar.js
```

The server loads Noa from `.runtime/avatar-candidates/Noa.glb`.

Use a Python 3.11 environment with NumPy, OpenCV, rtmlib, and ONNX Runtime for OpenMMLab. For Sapiens, use a CUDA-enabled PyTorch environment with NumPy and OpenCV. The example commands assume the paths below and an environment with those libraries installed.

```bash
python -m cv_comparison.extract_pose mmpose \
  --checkpoint $R/.runtime/pose-comparison/models/dwpose-t.onnx \
  --label 'DWPose-t · 133 landmarks' --device cpu --limit 100 --stride 2

python -m cv_comparison.extract_pose sapiens \
  --checkpoint $R/.runtime/pose-comparison/models/sapiens_0.3b_goliath_best_goliath_AP_573_torchscript.pt2 \
  --sapiens-demo $R/.runtime/pose-comparison/sapiens/lite/demo \
  --label 'Goliath 0.3B · 308 landmarks' --device cuda --precision bf16 --limit 100 --stride 2

python -m cv_comparison.extract_pose sdpose \
  --checkpoint $R/.runtime/pose-comparison/models/sdpose \
  --label 'Wholebody · 133 landmarks' --device cuda --precision bf16 --limit 100 --stride 2
```

Extract the ONNX file from the OpenMMLab archive as `dwpose-t.onnx`. For Sapiens, SHA-256 is `59ea239bfb217caf76cbeac1836460ef8f9ab5f39aeb22f2577373b62fa25edc`. Use `--precision fp32` to run without autocast. Predictions are written to `.runtime/pose-comparison/results/` as each clip completes. Reruns skip completed clips with the same checkpoint hash, sampling stride, and precision.

SDPose also requires Diffusers 0.35.1, Transformers, Pillow and Safetensors. Download the decoder, text encoder, tokenizer, UNet and VAE directories from the pinned model revision.

### HaMeR hand extraction

With the locally authorized MANO file in place, the runner uses HaMeR's official source in `.runtime/pose-comparison/hamer-source` and its checkpoint, release config, mean parameters and MANO file in `.runtime/pose-comparison/models/hamer`. It checks the checkpoint and MANO hashes before loading either file. It samples on the same frame indices as DWPose and uses MediaPipe for hand crops and left/right assignment. A frame with no usable crop has no HaMeR overlay for that hand. The avatar combines HaMeR's wrist-relative finger depth with MediaPipe's body and an assumed arm-depth plane.

The local Python 3.11 GPU runtime uses Torch 2.7 with torchvision 0.22, Lightning, timm, smplx, yacs, einops, the HaMeR renderer import dependencies, and a Chumpy 0.70 source copy patched for Python 3.11 and NumPy 1.26. The runtime paths are kept outside the app.

```bash
pose_work="$R/.runtime/pose-comparison"
PYTHONPATH="$pose_work/hamer-deps:$pose_work/runtime-deps:$pose_work/chumpy-source/chumpy-0.70" \
  PYOPENGL_PLATFORM=egl $R/.runtime/t5-local-venv/bin/python -m cv_comparison.run_hamer --limit 100
```

### MediaPipe 1.2.3 and HaMeR preview

The separate preview uses MediaPipe Holistic for both body joints and hand
crops, then runs HaMeR on those crops. Results are stored in
`.runtime/pose-comparison/results/hamer-latest/`. The preview lists clips as
their results become available and uses the same sample for the mesh and avatar.
Missing Holistic hand crops can use same-frame cached landmarks when they
agree with the body wrist. The preview shows when this fallback was used.
Holistic face landmarks and supported blendshapes drive the face overlay and
Noa’s expression. The **Face** checkbox toggles both.

Use the HaMeR environment above with MediaPipe 1.2.3 installed and the
[Holistic Landmarker task bundle](https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/1/holistic_landmarker.task).

```bash
# In the HaMeR inference environment
python -m cv_comparison.run_hamer --holistic-model $R/.runtime/pose-comparison/models/holistic-landmarker.task --limit 10

# In the web server environment
DICTASIGN_HAMER_PREVIEW=1 python -m cv_comparison.server --port 8096
```

### Sapiens 0.3B and HaMeR preview

This preview uses cached Goliath 308-landmark results for the body, face
overlay, and hand crops. HaMeR reconstructs the hands on the same source
frames. The Sapiens face points are displayed over the source video.
The avatar uses body and hand poses from this combination.

Both HaMeR previews fit a generic 3D face template to six tracked facial
landmarks to estimate head rotation. This drives the avatar’s head on the
same sample as the overlay. The fit uses approximate camera intrinsics.

Both HaMeR previews use a detected body wrist when a hand mesh is missing.
If the wrist also disappears, the avatar can bridge a short gap using
nearby wrist observations. The tracking readout distinguishes body-only
placement, an estimated wrist, and uncertain placement. These estimates
affect the avatar arm and do not add a hand mesh to the source video.

```bash
# In the HaMeR inference environment
python -m cv_comparison.run_hamer --sapiens --limit 10

# In the web server environment
DICTASIGN_HAMER_PREVIEW=sapiens python -m cv_comparison.server --port 8097
```

### Matignon notebook 07 full-frame preview

The preview on port 8099 uses the `extract_poses.py` cell from notebook 07.
It runs MediaPipe Holistic on full frames, with the notebook's resize rule
and the same clip IDs and sample timestamps as the main comparison.
Noa uses the body and hand landmarks. The source video overlay also
shows the facial landmarks.

```bash
python -m cv_comparison.matignon $R/.runtime/matignon-cv-review/extract_poses.py
DICTASIGN_HAMER_PREVIEW=matignon python -m cv_comparison.server --port 8099
```

Run extraction with the notebook's MediaPipe Tasks environment and
`HOLISTIC_MODEL` pointing at its Holistic Landmarker task. Result caches
are under `.runtime/pose-comparison/results/matignon-full-frame/`.

### Full local capture benchmark

`python -m cv_comparison.benchmark manifest` inventories
`.runtime/captures/*/source.mp4`. The benchmark preserves a row for every
source file and reuses computation for byte-identical files. It compares
notebook 07 (full frame) with Sapiens BF16 and HaMeR on every second source
frame. The notebook processes intervening frames for tracking state.
Run `notebook` in the MediaPipe Tasks environment and `sapiens`, then
`hamer`, in the local CUDA pose environment. Each stage can resume from
its completed per-video results.

The `report` stage checks exact frame and timestamp matches and writes
`.runtime/all-video-cv-benchmark/summary.json` and `per-video.csv`.
Counts measure hand availability. HaMeR mesh counts are reported
separately from 2D fallback coverage. Decode failures and sample
mismatches remain visible in the per-video report.
