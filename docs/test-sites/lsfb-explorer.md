# LSFB Corpus Explorer

Port 8105. Code: `~/Projects/Active/Forum_Hackathon/apps/forum-tests/lsfb_explorer/`. Build the viewer from the existing private LSFB review package, then serve it on loopback (normally the `signaculture-lsfb-corpus-explorer` unit does this).

```bash
cd ~/Projects/Active/Forum_Hackathon/apps/forum-tests && R=../Forum_Hackathon
python3 lsfb_explorer/prepare.py $R/reports/private/lsfb-alignment-audit/video-review $R/.runtime/lsfb-browser
$R/.venv/bin/python -m uvicorn lsfb_explorer.serve:app --host 127.0.0.1 --port 8105
```

The sampled excerpts come from annotated Belgian LSFB-CONT recordings and are provided for local noncommercial research. This convenience sample has 49 training excerpts and six validation excerpts. It uses corpus French translations and original hand annotation timestamps. The generated site and copies of the clips live under the ignored `.runtime` directory.
