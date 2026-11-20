# SignaCulture documentation

Product and engineering documentation. Research questions, experiments and
their results live in [`research/`](../research/README.md).

| Folder | Contents |
|---|---|
| [architecture/](architecture/) | System architecture, integration contract, methodology, semantic approach |
| [architecture/voice-expression/](architecture/voice-expression/) | Voice emotion, stress and pace driving the avatar: architecture and benchmarks |
| [avatar/](avatar/) | Avatar playback, provenance, hand collisions, motion quality and phrase reviews |
| [data-pipeline/](data-pipeline/) | Dicta-Sign ingestion and the gloss → pose export pipeline |
| [verification/](verification/) | Studio engineering rechecks: capture language, ingestion, playback, Research panel (raw logs in `verification/logs/`) |
| [governance/](governance/) | Ethics, data governance, validation gating, licences and provenance |
| [test-sites/](test-sites/README.md) | The comparison sites on ports 8094–8105 (code in the sibling `Forum_Tests` folder) |
| [deployment/](deployment/) | Windows restore notes (Docker guide: [../deployment/DOCKER.md](../deployment/DOCKER.md)) |
| [presentation/](presentation/) | Slides, demo narration, pipeline PDF |
| [planning/](planning/) | Original development plan, orchestration, team worktree organization |
| [project-log/](project-log/) | Progress snapshots and conversation handoff context |
| [diagrams/](diagrams/), [pipeline/](pipeline/) | Generated diagrams (built by `scripts/build-diagrams.py` and `scripts/build-pipeline-diagram.py`) |
