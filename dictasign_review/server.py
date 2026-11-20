"""Serve a local review of the first 100 clips in the cached Dicta-Sign inventory."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
import os
from pathlib import Path
import subprocess
from uuid import NAMESPACE_URL, uuid5

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, Response
import uvicorn

PAGE = Path(__file__).resolve().parent
ROOT = PAGE.parent
RUNTIME = ROOT / ".runtime"
VIDEO_CACHE = RUNTIME / "dictasign-review" / "video"
POSE_CACHE = RUNTIME / "pose-comparison" / "results"
HAMER_PREVIEW = os.getenv("DICTASIGN_HAMER_PREVIEW")
HAMER_RESULT = {"1": "hamer-latest", "sapiens": "hamer-sapiens", "matignon": "matignon-full-frame"}.get(HAMER_PREVIEW)
POSE_MODELS = ("mmpose", "sapiens", "sdpose", "hamer")
PUBLIC_REVIEW = os.getenv("DICTASIGN_PUBLIC_REVIEW") == "1"

inventory = ROOT / "data/source_inventory/dictasign_lsf.json"
entries = json.loads(inventory.read_text())["entries"]
status_path = RUNTIME / "dictasign_status.json"
status = json.loads(status_path.read_text()) if status_path.is_file() else {}
quality = {
    item["video_url"]: item["quality"]
    for key in ("results", "recovered_failures")
    for item in status.get(key, [])
    if item.get("video_url") and item.get("quality")
}
clips = []
folders = {}
for entry in [entry for entry in entries if entry["language"] == "fsl"][:100]:
    url = entry["video_url"]
    capture_id = str(uuid5(NAMESPACE_URL, url))
    folder = RUNTIME / "captures" / capture_id
    folders[capture_id] = folder
    clips.append({
        "id": capture_id,
        "concept": entry["concept_fr"],
        "gloss": entry["canonical_gloss"],
        "source": entry["page_url"],
        "sourceVideo": url if PUBLIC_REVIEW else None,
        "hasVideo": (folder / "source.mp4").is_file(),
        "hasWebm": not PUBLIC_REVIEW and (VIDEO_CACHE / f"{capture_id}.webm").is_file(),
        "hasLandmarks": (folder / "animics-landmarks.json.gz").is_file(),
        "hasMotion": (folder / "motion.json").is_file(),
        "quality": quality.get(url, {}),
    })

app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/")
def index():
    if HAMER_RESULT:
        html = (PAGE / "index.html").read_text()
        source = {"hamer-sapiens": "sapiens", "matignon-full-frame": "matignon"}.get(HAMER_RESULT, "mediapipe")
        return HTMLResponse(html.replace("<body>", f'<body data-preview="hamer" data-source="{source}">'))
    return FileResponse(PAGE / "index.html")


@app.get("/review.js")
def javascript():
    return FileResponse(
        PAGE / "review.js", media_type="text/javascript", headers={"Cache-Control": "no-cache"}
    )


@app.get("/avatar.js")
def avatar_bundle():
    path = RUNTIME / "dictasign-review" / "avatar.js"
    if not path.is_file():
        raise HTTPException(503, "Avatar viewer has not been built")
    return FileResponse(path, media_type="text/javascript", headers={"Cache-Control": "no-cache"})


@app.get("/api/avatar/model")
def avatar_model(request: Request):
    path = RUNTIME / "avatar-candidates" / "Noa.glb"
    if not path.is_file():
        raise HTTPException(503, "Noa avatar is unavailable")
    headers = {"Cache-Control": "public, max-age=86400", "Vary": "Accept-Encoding"}
    compressed = path.with_suffix(".glb.gz")
    if "gzip" in request.headers.get("accept-encoding", "") and compressed.is_file() and compressed.stat().st_mtime >= path.stat().st_mtime:
        path = compressed
        headers["Content-Encoding"] = "gzip"
    return FileResponse(
        path, media_type="model/gltf-binary", headers=headers
    )


@app.get("/api/clips")
def list_clips():
    if HAMER_RESULT:
        return [
            {**clip, "poseModels": ["hamer"]}
            for clip in clips if (POSE_CACHE / HAMER_RESULT / f"{clip['id']}.json").is_file()
        ]
    return [
        {**clip, "poseModels": [
            model for model in POSE_MODELS if (POSE_CACHE / model / f"{clip['id']}.json").is_file()
        ]}
        for clip in clips
    ]


def asset(capture_id, filename):
    folder = folders.get(capture_id)
    if folder is None or not (folder / filename).is_file():
        raise HTTPException(404, "Asset is unavailable in the review set")
    return folder / filename


@app.get("/api/clips/{capture_id}/video")
def video(capture_id: str):
    if PUBLIC_REVIEW:
        raise HTTPException(404, "Open the original Dicta-Sign video")
    return FileResponse(asset(capture_id, "source.mp4"), media_type="video/mp4")


@app.get("/api/clips/{capture_id}/video.webm")
def webm(capture_id: str):
    path = VIDEO_CACHE / f"{capture_id}.webm"
    if PUBLIC_REVIEW or capture_id not in folders or not path.is_file():
        raise HTTPException(404, "WebM preview is unavailable in the review set")
    return FileResponse(path, media_type="video/webm")


@app.get("/api/clips/{capture_id}/pose/{model}")
def pose(capture_id: str, model: str):
    if capture_id not in folders or model not in POSE_MODELS:
        raise HTTPException(404, "Unknown clip or model")
    result_dir = HAMER_RESULT if HAMER_RESULT and model == "hamer" else model
    path = POSE_CACHE / result_dir / f"{capture_id}.json"
    if not path.is_file():
        if model == "hamer" and not (RUNTIME / "pose-comparison" / "models" / "hamer" / "MANO_RIGHT.pkl").is_file():
            return {"status": "blocked", "reason": "Hand extraction requires an authorized MANO model."}
        return {"status": "pending"}
    return FileResponse(path, media_type="application/json", headers={"Cache-Control": "no-cache"})


@app.get("/api/clips/{capture_id}/landmarks")
def landmarks(capture_id: str):
    path = asset(capture_id, "animics-landmarks.json.gz")
    if PUBLIC_REVIEW:
        with gzip.open(path, "rt") as source:
            frames = json.load(source)
        overlay = [
            {
                "time_ms": frame["time_ms"],
                **{
                    key: [
                        {axis: point[axis] for axis in ("x", "y", "visibility") if axis in point}
                        if point is not None else None
                        for point in frame[key]
                    ] if frame.get(key) is not None else None
                    for key in ("PLM", "LLM", "RLM", "FLM")
                },
            }
            for frame in frames
        ]
        return Response(
            gzip.compress(json.dumps(overlay, separators=(",", ":")).encode(), compresslevel=3),
            media_type="application/json",
            headers={"Content-Encoding": "gzip"},
        )
    return FileResponse(
        path,
        media_type="application/json",
        headers={"Content-Encoding": "gzip"},
    )


def prepare_webm():
    VIDEO_CACHE.mkdir(parents=True, exist_ok=True)

    def convert(item):
        capture_id, folder = item
        source = folder / "source.mp4"
        target = VIDEO_CACHE / f"{capture_id}.webm"
        if not source.is_file() or (target.is_file() and target.stat().st_mtime >= source.stat().st_mtime):
            return
        temporary = target.with_suffix(".tmp.webm")
        try:
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-threads", "1",
                "-i", str(source), "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libvpx",
                "-threads", "1", "-deadline", "realtime", "-cpu-used", "5",
                "-crf", "14", "-b:v", "1M", "-c:a", "libopus", str(temporary),
            ], check=True)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(convert, folders.items()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--prepare-webm", action="store_true", help="Build local browser-compatible video previews and exit")
    args = parser.parse_args()
    if args.prepare_webm:
        prepare_webm()
        raise SystemExit
    uvicorn.run(app, host="127.0.0.1", port=args.port)
