"""Local-only comparison preview for the Studio and learned transition paths."""
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

PAGE = Path(__file__).resolve().parent
RUNTIME = PAGE.parent / ".runtime"
app = FastAPI()


@app.get("/")
def index():
    return FileResponse(PAGE / "index.html")


@app.get("/examples.json")
def examples():
    return FileResponse(RUNTIME / "transition-preview/examples.json", media_type="application/json")


@app.get("/avatar.js")
def avatar():
    return FileResponse(RUNTIME / "dictasign-review/avatar.js", media_type="text/javascript")


@app.get("/api/avatar/model")
def model():
    return FileResponse(RUNTIME / "avatar-candidates/Noa.glb", media_type="model/gltf-binary")
