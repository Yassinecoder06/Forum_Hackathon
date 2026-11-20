# Test and comparison sites

These local sites compare computer-vision trackers, transition methods and
research outputs. They are review tools, not part of the Studio.

**Code** lives outside this repository in the sibling folder
`~/Projects/Active/Forum_Tests/`. **Documentation** (this folder), data,
caches (`.runtime/`) and models (`.models/`) stay here. The viewers find this
repository through `Forum_Tests/forum_paths.py` (override with
`SIGNACULTURE_ROOT`). The shared Sapiens / MMPose / HaMeR / transition-model
code they reuse is in [`motion_worker/`](../../motion_worker), because the
Studio's GPU capture worker needs it too.

| Port | Site | Code in `Forum_Tests/` | systemd unit | Docs |
|---|---|---|---|---|
| 8094 | CV comparison, all five models | `cv_comparison` | `signaculture-review-8094` | [cv-comparison.md](cv-comparison.md) |
| 8095 | Same, **public** via Cloudflare (`cv.yassinejemmali.me`) | `cv_comparison` (`DICTASIGN_PUBLIC_REVIEW=1`) | `signaculture-cv-origin` | [cv-comparison.md](cv-comparison.md) |
| 8096 | MediaPipe 1.2.3 + HaMeR | `cv_comparison` (`DICTASIGN_HAMER_PREVIEW=1`) | `signaculture-review-8096` | [cv-comparison.md](cv-comparison.md#mediapipe-123-and-hamer-preview) |
| 8097 | Sapiens 0.3B + HaMeR | `cv_comparison` (`=sapiens`) | `signaculture-review-8097` | [cv-comparison.md](cv-comparison.md#sapiens-03b-and-hamer-preview) |
| 8098 | Math vs learned (exp15) transitions | `transition_preview` | `signaculture-transition-preview` | [transition-preview.md](transition-preview.md) |
| 8099 | Matignon notebook 07, full frame | `cv_comparison` (`=matignon`) | `signaculture-review-8099` | [cv-comparison.md](cv-comparison.md#matignon-notebook-07-full-frame-preview) |
| 8100 | Broadcast subtitle/pose viewer | `broadcast_viewer` | `signaculture-broadcast-viewer` | [broadcast-viewer.md](broadcast-viewer.md) |
| 8104 | Direct France-LSF pose demo | `direct_fsl_demo` | `signaculture-direct-fsl-demo` | [direct-fsl-demo.md](direct-fsl-demo.md) |
| 8105 | LSFB corpus explorer | `lsfb_explorer` | `signaculture-lsfb-corpus-explorer` | [lsfb-explorer.md](lsfb-explorer.md) |

## Operating them

```bash
cd ~/Projects/Active/Forum_Tests
make status    # state + HTTP check of every port
make up        # start all
make down      # stop all (takes the public 8095 site offline)
make install   # (re)install unit files from Forum_Tests/systemd/
```

The units are user services. They start at login; to start them at boot
without logging in, enable lingering: `loginctl enable-linger $USER`.
