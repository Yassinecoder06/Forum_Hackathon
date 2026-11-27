# Repository structure

SignaCulture / **SignePourMoi**: live French → LSF (French Sign Language) interpretation by a MetaHuman avatar
(speech → Whisper → C0 + hand/face refiners → avatar), with a film mode, an AR mode and a research track.

This folder is both the Git repository **and** the single home of everything the project uses on the main PC.
Only code and documentation are tracked. Everything else lives in git-ignored folders described below, so a fresh
clone contains no private data.

> **Never commit or upload** model weights, datasets, poses, interpreter / demo videos, the MetaHuman avatar,
> `curated.npz`, or anything under the local folders below. They are ignored by `.gitignore` on purpose.

## Top level

| Path | Tracked | What it is |
|---|---|---|
| `backend/` | yes | FastAPI service: Whisper worker, `/lsf/generate` (C0 / B2 engines), voice analysis, brain |
| `frontend/` | yes | Vite + React app (Studio, interpreter, film, AR window) |
| `motion_worker/` | yes | GPU pose / motion worker (Sapiens, HaMeR, learned transitions) |
| `research/` | yes | training and evaluation code, results (`research/RESULTS.md`); `research/private/` is local |
| `docs/` | yes | project documentation (test sites, project log, planning) |
| `scripts/`, `configs/`, `deployment/`, `e2e/`, `integration/`, `site/`, `webcam/` | yes | tooling, configs, deployment, end-to-end tests, public site, webcam studio |
| `data/`, `artifacts/` | partly | small tracked inventories and review outputs; private parts ignored (see `.gitignore`) |
| `start.sh` / `stop.sh` | yes | start / stop the presentation stack (below) |
| `STRUCTURE.md` | yes | this file |
| `.runtime/` | no | research runs, caches, captures, the motion worker socket; read by the test sites |
| `.models/` | no | research model weights (`lsfb-transition`, `faster-whisper-small`, `emotion2vec-seed`, …) |
| `.worktrees/` | no | the feature worktrees (below) |
| `workspace/` | no | the live workspace (below) |
| `datasets/` | no | LSF datasets and pose collections (below) |
| `apps/` | no | local web apps (below) |
| `assets/` | no | private media and documents (below) |
| `CLAUDE.md`, `.claude/` | no | instructions for local coding agents |

## Local folders (git-ignored)

### `workspace/` — the live workspace
Everything the running product uses on this PC. Start with `workspace/CLAUDE.md` (rules for agents, both PCs),
`workspace/CONTEXT_INDEX.md` (where every file is) and the newest `workspace/HANDOFF_*.md` (latest state, open items).

| Path | What |
|---|---|
| `models/` | live LSF weights: C0 generator + tokenizer + hand / face refiners, B2 lexicon and transitions |
| `hf/` | Hugging Face models: Whisper large-v3-turbo, emotion2vec+ large, multilingual-e5-base |
| `data/` | runtime data (MetaHuman avatar, `curated.npz`), sign-to-text data, MediaPipe files, a git bundle of all branches |
| `show/` | the API (`show/run/api.py`, `env.sh`, warm-up) and the comparison renderer |
| `ar/` | AR mode: tracking (OpenVINS build `vio/`), depth, scene models, TensorRT engines, calibration recordings, EuRoC |
| `live/` | microphone hub, context layer, AR program output and its recordings |
| `showcase/`, `demo-media/` | model showcase / comparison videos and metrics; demo clips and subtitles |
| `bench/`, `tools/`, `lan/`, `tunnels/`, `phone-app/`, `docs/` | measurements, landolsi tools, LAN hub, tunnel config, Android sensor app, notes |
| `team-repo/` | the team repository clone (all branches, push disabled); parent of the `.worktrees/FH-*` worktrees |
| `.venv/` | Python environment of the API (torch CUDA, backend dependencies) |

### `.worktrees/` — feature worktrees
`FH-show` (integration), `FH-presentation` (presentation + interpreter), `FH-arpc` (AR window), `FH-signtext`
(sign → text), `FH-ar`, `FH-demo`, `FH-prosody`, `FH-pointing`, `FH-c0opt`, `FH-avatarperf`. They are worktrees of
`workspace/team-repo` (`git -C workspace/team-repo worktree list`). Several hold uncommitted work: commit before deleting.
Their tracked files are branch snapshots and still contain the old absolute paths; `main` has the current ones.

### `datasets/`
| Path | What |
|---|---|
| `pose-collection/` | our pose collection runs (`first/`, `second/`, `OUT/`), crops, collection code |
| `lsfb-*`, `geoquery-lsfb-*` | LSFB (Belgian SL corpus) derived datasets and GeoQuery text-to-gloss sets |
| `signaculture-research-evidence/` | evidence bundle for the research results |
| `dictasign-lexicon/` | DictaSign LSF lexicon (videos, poses, motions) |

### `apps/`
| Path | What | Port |
|---|---|---|
| `presentation/` | the forum deck: slides + live interpreter blended in the slide, big subtitles | 5215 |
| `forum-tests/` | test / comparison sites (code only; data comes from this repo), run as `signaculture-*` user services | 8094–8109 |
| `lsf-azee-explorer/` | LSF + AZee corpus explorer (40 brèves), user service | 8112 |

### `assets/`
`presentation/` (deck PDFs, hackathon brief, Business Model Canvas), `brand/` (SignePourMoi logo), `videos/` (demo and
screen recordings: live interpreter, film, emotions, Studio, AR room), `source-media/` (third-party clips used as demo
inputs), `documents/` (studies, reports, third-party papers), `benchmarks/` (arm-limit before/after), `research/`
(research outputs kept only here).

## Running
| What | How |
|---|---|
| Presentation stack: API 8040 (Whisper, C0 + refiners, emotion2vec), interpreter 5207, deck 5215 | `./start.sh`, then open http://localhost:5215/ · stop: `./stop.sh` (`--keep-api` keeps models loaded) |
| Test sites + motion worker + webcam studio | `systemctl --user status 'signaculture-*'` (unit files in `apps/forum-tests/systemd/`) |
| Full show stack (API + Studio 5200) | `workspace/show/run/start.sh` (do not run right after a reboot: stale pid files) |

## Notes
- The landolsi PC (Windows, training data) keeps its own copy; its helper scripts are maintained in `workspace/tools/landolsi/`.
- This layout dates from 2026-11-27 (everything moved here from `~/Forum`, `~/Dataset`, `~/Data_Collection`,
  `~/Downloads/LSF` and sibling project folders). Old paths in history (branch snapshots, `data/catalog/catalog.sqlite`)
  refer to those earlier locations.
