# Run SignaCulture with Docker

Use Docker Engine with the Compose plugin. Run these commands in the
repository root. The default image includes the production Studio, showcase,
API, FFmpeg, MediaPipe, and CPU runtimes for emotion2vec and French ASR.
The browser renders the avatar with WebGL.

```bash
mkdir -p .runtime/docker-state .runtime/avatar-candidates .runtime/lsf-models .models
docker compose up --build --detach --wait
```

Open [the Studio](http://127.0.0.1:8088/studio/) or [the showcase](http://127.0.0.1:8088/).
The container listens on port 8080 and Compose publishes it on localhost only.

## Local assets and persistence

The image contains the attributed Bonjour example and its motion assets.
It uses read-only mounts for avatar files and audio model weights.

- Put the existing checksum-pinned `Noa.glb` or `ReadyEva.glb` in
  `.runtime/avatar-candidates/`. See [avatar provenance](../docs/avatar/AVATAR_PROVENANCE.md)
  for the asset setup and reuse terms.
- Put existing model files in `.models/emotion2vec-seed/` and
  `.models/faster-whisper-small/`. See [audio model setup](../research/audio/ASR_MODEL_SETUP.md).
  The service checks for local files. It does not require a GPU or download the
  weights at startup. Missing weights leave the corresponding capability disabled.
- Captures, the library, and the plan database persist in `.runtime/docker-state/`.
  The host development library is separate by default. The image excludes
  private recordings, training checkpoints and workspace files.
- Use `SIGNACULTURE_DOCKER_TARGET=runtime` on the Compose command to build
  the smaller variant without the emotion and ASR dependencies.
- Use `SIGNACULTURE_DOCKER_TARGET=engines` to add CPU PyTorch, transformers and
  scipy for the experimental LSF engines (C0, B2; `POST /api/lsf/generate`). Put the
  private weights in `.runtime/lsf-models/` (mounted read-only at `/lsf-models`) and
  the public `intfloat/multilingual-e5-base` files in `.models/e5-base/`. Override
  with `SIGNACULTURE_DOCKER_LSF_MODELS` and `SIGNACULTURE_DOCKER_LSF_TEXT_ENCODER`.
  See [LSF engines](../docs/pipeline/LSF_ENGINES.md). Without this target the
  endpoint answers 503 and the capability endpoint reports the engines as absent.
- Override `SIGNACULTURE_DOCKER_PORT` if 8088 is occupied. On Linux, set
  `SIGNACULTURE_DOCKER_UID=$(id -u)` and `SIGNACULTURE_DOCKER_GID=$(id -g)`
  if your writable state directory uses IDs other than 1000:1000.

Older library entries contain absolute paths. To reuse an existing library,
stop the native service and mount the state at its original absolute path
inside the container. Set both storage variables for that project.

```bash
SIGNACULTURE_DOCKER_STATE="$PWD/.runtime" \
SIGNACULTURE_DOCKER_STORAGE="$PWD/.runtime" \
docker compose up --detach --wait
```

## Verify the container

Run the API smoke check against a dedicated state directory. It creates
two local capture jobs from the bundled public example.

```bash
python3 scripts/smoke_docker.py --full
APP_URL=http://127.0.0.1:8088/studio/ CHROME_PATH=/path/to/chromium E2E_REQUIRE_ASR=1 npm --prefix e2e test
docker compose down
```

Use `npm --prefix e2e ci` first if the browser test dependencies are missing.
The browser suite checks composition, seeking, audio timing, French
transcription, JSON import and a real WebM download. Omit `--full` for the
smaller image or if audio weights have not been mounted. Set `E2E_REQUIRE_ASR=1` to require the ASR check. Without it, the browser suite skips transcription only when the server reports it unavailable.

## Webcam capture

Put the existing MediaPipe `.task` files under `webcam/public/models/`.
The webcam image builds the worker and copies the SDK's WASM at build time.
The service mounts the model files read-only and processes camera frames
in the browser. In the Studio, open **Capturer → Capturer avec la webcam**,
record a take, then choose **Utiliser dans le Studio**. Confirm that you have
permission and submit the movement using the usual capture form.

To run the optional standalone webcam page as well:

```bash
docker compose --profile webcam up --build --detach --wait webcam
```

Open [the webcam app](http://127.0.0.1:5188/). It uses a separate host port
so the native dev server can continue to use 5180. See the
[webcam readme](../webcam/README.md) for the model setup and browser check.
Stop both web services with `docker compose --profile webcam down`.

The upstream pose diagnostic runner and training experiments retain their
separate environments.

## Local verification

On Linux/amd64, both image variants passed a real video capture and
compressed Animics round trip (61 sampled landmark frames, 62 motion
frames). The full image also ran local emotion2vec and CPU French
transcription with mounted weights. Chromium passed the production Studio
smoke suite and downloaded a decodable WebM. The webcam check used a
prerecorded camera feed and exercised recording, playback, and downloads
with zero browser errors or external requests. Physical webcam performance
was outside this check.

## GPU capture and transition comparison

The Studio uses mathematical interpolation for multi-sign joins. LSFB exp15
is available in the separate experimental transition comparison on port 8098.

Uploaded videos and recordings sent from the embedded webcam use Sapiens
0.3B + HaMeR when the GPU worker is available. The existing MediaPipe pass
supplies facial measurements, the raw Animics export, and observations the
GPU models miss. A webcam browser without video recording can still hand off
its landmark JSON.

On this host, the worker uses the CUDA environment and checkpoints from
the local comparison previews. The Studio container connects over the
shared Unix socket. Start the worker from the repository root before
starting Compose. For a custom state directory, use the same
`SIGNACULTURE_DOCKER_STATE` for both commands.

```bash
bash scripts/start-motion-inference.sh
# In a second terminal
docker compose --profile webcam up --build --detach --wait
```

The worker expects FastAPI and Uvicorn in the local pose inference environment
or its `inference-deps` directory. It also needs the local exp15 checkpoint
under `.models/lsfb-transition/` and the pinned Sapiens and HaMeR assets under
`.runtime/pose-comparison/`. The optional `SIGNACULTURE_AUDIO_DEPS_IMAGE`
build argument accepts a previously built compatible Studio image as the
source of `/opt/audio`. The default build installs audio dependencies
from the repository requirements.
