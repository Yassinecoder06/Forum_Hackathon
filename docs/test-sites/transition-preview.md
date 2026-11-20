# Transition preview (port 8098)

Compares the Studio's mathematical joins with the learned LSFB exp15
transition model on three example sign sequences. Source clips, endpoints
and 480 ms joins are identical across panels. exp15 was trained on Belgian
LSFB; its quality on France-LSF transitions is unverified.

Code: `~/Projects/Active/Forum_Tests/transition_preview/`. The model code it
uses is [`motion_worker/transition_model.py`](../../motion_worker/transition_model.py)
and the checkpoint is `.models/lsfb-transition/transition_model_exp15.pt`.

Regenerate the examples (bundles the real Studio stitcher from
`frontend/src/utils/planStitcher.ts` with esbuild), then serve:

```bash
cd ~/Projects/Active/Forum_Tests && R=../Forum_Hackathon
$R/.venv/bin/python -m transition_preview.build_examples   # needs torch; writes $R/.runtime/transition-preview/examples.json
$R/.venv/bin/python -m uvicorn transition_preview.server:app --host 127.0.0.1 --port 8098
```

The page also needs `.runtime/dictasign-review/avatar.js` and
`.runtime/avatar-candidates/Noa.glb` (see [cv-comparison.md](cv-comparison.md)).
