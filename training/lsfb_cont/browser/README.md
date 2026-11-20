# LSFB Corpus Explorer

From the repository root, build the viewer from the existing private LSFB review package and serve it on loopback.

```bash
python3 training/lsfb_cont/browser/prepare.py reports/private/lsfb-alignment-audit/video-review .runtime/lsfb-browser
python3 -m uvicorn training.lsfb_cont.browser.serve:app --host 127.0.0.1 --port 8105
```

The sampled excerpts come from annotated Belgian LSFB-CONT recordings and are provided for local noncommercial research. This convenience sample has 49 training excerpts and six validation excerpts. It uses corpus French translations and original hand annotation timestamps. The generated site and copies of the clips live under the ignored `.runtime` directory.
