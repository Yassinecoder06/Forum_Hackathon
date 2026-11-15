#!/usr/bin/env python3
"""Serve the whole local demo from one process: showcase at /, built Studio at /studio/,
API at /api, diagrams at /diagrams. Build the Studio first: npm --prefix frontend run build."""
from pathlib import Path
import os
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('SIGNACULTURE_STORAGE', str(ROOT / '.runtime'))
os.environ.setdefault('SIGNACULTURE_DB_PATH', str(ROOT / '.runtime/plans.sqlite3'))
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
if not os.getenv('SIGNACULTURE_FFMPEG') and shutil.which('ffmpeg'):
    os.environ['SIGNACULTURE_FFMPEG'] = shutil.which('ffmpeg')

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from backend.app.main import app as api

app = FastAPI(title='SignaCulture local demo')
app.mount('/api', api)
studio = ROOT / 'frontend/dist'
if not (studio / 'index.html').is_file():
    raise SystemExit('Studio not built: run `npm --prefix frontend run build` first.')
app.mount('/studio', StaticFiles(directory=studio, html=True), name='studio')
if (ROOT / 'docs/diagrams').is_dir():
    app.mount('/diagrams', StaticFiles(directory=ROOT / 'docs/diagrams'), name='diagrams')
if (ROOT / 'site/index.html').is_file():
    app.mount('/', StaticFiles(directory=ROOT / 'site', html=True), name='site')
else:
    @app.get('/')
    def root():
        return RedirectResponse('/studio/')

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host=os.getenv('SIGNACULTURE_HOST', '127.0.0.1'), port=int(os.getenv('PORT', '8080')))
