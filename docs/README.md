# SignaCulture documentation

Product and engineering documentation. Research questions, experiments and
their results live in [`research/`](../research/README.md).

| Folder | Contents |
|---|---|
| [architecture/](architecture/) | System architecture, integration contract, methodology, semantic approach |
| [avatar/](avatar/) | Avatar playback, provenance, hand collisions, motion quality and phrase reviews; [MetaHuman avatar](avatar/METAHUMAN.md) |
| [data-pipeline/](data-pipeline/) | Dicta-Sign ingestion and the gloss → pose export pipeline |
| [verification/](verification/) | Studio engineering rechecks: capture language, ingestion, playback, Research panel (raw logs in `verification/logs/`) |
| [governance/](governance/) | Ethics, data governance, validation gating, licences and provenance |
| [test-sites/](test-sites/README.md) | The comparison sites on ports 8094–8105 (code in the sibling `Forum_Tests` folder) |
| [deployment/](deployment/) | Windows restore notes (Docker guide: [../deployment/DOCKER.md](../deployment/DOCKER.md)) |
| [presentation/](presentation/) | Slides, demo narration, pipeline PDF |
| [planning/](planning/) | Original development plan, orchestration, team worktree organization |
| [project-log/](project-log/) | Progress snapshots and conversation handoff context |
| [pipeline/LSF_ENGINES.md](pipeline/LSF_ENGINES.md) | Experimental LSF engines C0/B2 (`POST /lsf/generate`): model paths, pose conversion, limits |
| [diagrams/](diagrams/), [pipeline/](pipeline/) | Generated diagrams (built by `scripts/build-diagrams.py` and `scripts/build-pipeline-diagram.py`) |
