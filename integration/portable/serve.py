"""One-process offline app: prebuilt UI plus API, no Node or frontend build."""
from pathlib import Path
import os
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
os.chdir(ROOT)
os.environ.setdefault('SIGNACULTURE_STORAGE',str(ROOT/'.runtime'))
os.environ.setdefault('SIGNACULTURE_DB_PATH',str(ROOT/'.runtime/plans.sqlite3'))
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('TRANSFORMERS_OFFLINE','1')
os.environ['PATH']=str(ROOT/'portable/bin')+os.pathsep+os.environ.get('PATH','')
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from backend.app.main import app as api
app=FastAPI(title='SignaCulture portable local app')
app.mount('/api',api)
app.mount('/',StaticFiles(directory=ROOT/'frontend/dist',html=True),name='ui')
if __name__=='__main__':
    import uvicorn
    uvicorn.run(app,host=os.getenv('SIGNACULTURE_HOST','127.0.0.1'),port=int(os.getenv('PORT','8080')))
