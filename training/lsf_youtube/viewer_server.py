"""Local review API for the three full-length subtitle/pose extractions."""
import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
from time import sleep

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import numpy as np
import uvicorn

ROOT = Path(__file__).resolve().parents[2] / ".runtime/youtube-lsf"
PAGE = Path(__file__).with_name("viewer")
DATA = ROOT / os.getenv("SIGNACULTURE_BROADCAST_VIEWER_DATA", "broadcast-dataset")
SOURCES = {
    "XsCjW__Yp1U": ("Le Bourget", "XsCjW__Yp1U.mp4"),
    "Z5YIa1QD178": ("Saint-Denis", "Z5YIa1QD178.mp4"),
    "alTS84wDvu0": ("Clermont-Ferrand", "alTS84wDvu0.h264.mp4"),
}
app = FastAPI()


def paths(video_id):
    if video_id not in SOURCES:
        raise HTTPException(404, "Diffusion inconnue")
    results = [DATA / f"{video_id}.{suffix}" for suffix in ("summary.json", "pairs.jsonl", "npz")]
    results.append(ROOT / "pilot" / SOURCES[video_id][1])
    if not all(p.is_file() for p in results):
        raise HTTPException(404, "Extraction en cours")
    return results


@lru_cache(maxsize=3)
def manifest(video_id):
    summary, captions, _, _ = paths(video_id)
    meta = json.loads(summary.read_text())
    records = [json.loads(line) for line in captions.read_text().splitlines() if line]
    return meta, records


@lru_cache(maxsize=1)
def archive(video_id):
    _, _, pose_file, _ = paths(video_id)
    with np.load(pose_file, allow_pickle=False) as dataset:
        return dataset["pts"], dataset["t_ms"], dataset["has"]


@app.get("/")
def index():
    return FileResponse(PAGE / "index.html")


@app.get("/viewer.js")
def javascript():
    return FileResponse(PAGE / "viewer.js", media_type="text/javascript")


@app.get("/api/broadcasts")
def broadcasts():
    results = []
    for video_id, (name, _) in SOURCES.items():
        try:
            meta, records = manifest(video_id)
        except HTTPException:
            continue
        results.append({"id": video_id, "name": name, "title": meta["title"],
                        "windows": len(records), "candidates": meta["candidate_windows"], "frames": meta["frames"]})
    return results


@app.get("/api/broadcasts/{video_id}")
def broadcast(video_id: str):
    meta, records = manifest(video_id)
    return {"meta": meta, "windows": records}


@app.get("/api/broadcasts/{video_id}/video")
def video(video_id: str):
    return FileResponse(paths(video_id)[3], media_type="video/mp4")


@app.get("/api/broadcasts/{video_id}/windows/{index:int}")
def window(video_id: str, index: int):
    _, records = manifest(video_id)
    if not 0 <= index < len(records):
        raise HTTPException(404, "Segment inconnu")
    entry = records[index]
    points, times, has = archive(video_id)
    lo, hi = entry["frame_start"], entry["frame_stop"]
    xyz = points[lo:hi, :, :2].astype(float)
    values = [[[round(float(x), 5), round(float(y), 5)] if np.isfinite([x, y]).all() else None
               for x, y in frame] for frame in xyz]
    return {"times": times[lo:hi].tolist(), "points": values, "has": has[lo:hi].tolist()}


def verify():
    for video_id in SOURCES:
        meta, records = manifest(video_id)
        _, _, pose, _ = paths(video_id)
        with np.load(pose, allow_pickle=False) as data:
            times = data["t_ms"]
            detected = data["has"]
        if len(times) != meta["frames"] or detected.shape != (len(times), 4) or (np.diff(times) <= 0).any() or not records:
            raise ValueError(f"Invalid extraction for {video_id}")
        for row in records:
            first, last = row["frame_start"], row["frame_stop"]
            if not 0 <= first < last <= len(times):
                raise ValueError(f"Invalid pose range for {video_id}")
            if times[first] < row["start_ms"] or times[last - 1] >= row["end_ms"]:
                raise ValueError(f"Subtitle timestamps do not match the pose data for {video_id}")
        print(f"{video_id}: {len(times)} frames and {len(records)} verified windows", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8100)
    parser.add_argument("--verify", action="store_true")
    parser.add_argument("--wait-for-results", action="store_true")
    args = parser.parse_args()
    if args.wait_for_results:
        previous = None
        while True:
            ready = sum((DATA / f"{video_id}.summary.json").is_file() for video_id in SOURCES)
            if ready != previous:
                print(f"Waiting for full extractions: {ready}/{len(SOURCES)} ready", flush=True)
                previous = ready
            if ready == len(SOURCES):
                break
            sleep(15)
        verify()
    if args.verify:
        verify()
    else:
        uvicorn.run(app, host="127.0.0.1", port=args.port)
