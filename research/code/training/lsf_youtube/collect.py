"""Collect the public Messe LSF uploads and build local subtitle/pose candidates."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from time import sleep
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from .align import cue_segments, extract, load_extractor, write_json

ROOT = Path(".runtime/youtube-lsf")
STATUS = ROOT / "download-status"
POSES = ROOT / "channel-dataset"
MEDIA = ROOT / "source-videos"
STOP_FREE_BYTES = 12 * (1 << 30)
CHANNEL_ID = "UCgPxLUNSaP6GKsbntEAL6xg"
YTDLP = [sys.executable, "-m", "yt_dlp", "--js-runtimes", "node", "--no-playlist",
         "--socket-timeout", "20", "--retries", "2", "--no-progress", "--no-warnings",
         "--downloader", "aria2c", "--downloader-args",
         "aria2c:-x 4 -s 4 -k 1M --file-allocation=none"]


def get_track(info):
    for field, label, codes in (
        ("subtitles", "youtube_french_manual", ("fr", "fr-FR", "fr-CA")),
        ("automatic_captions", "youtube_french_auto", ("fr-orig",)),
    ):
        for code in codes:
            item = next((f for f in info.get(field, {}).get(code, []) if f.get("ext") == "json3"), None)
            if item:
                return item, label
    return None, None


def metadata(video_id):
    for ending in (".channel.json", ".bulk.json"):
        path = ROOT / "metadata" / (video_id + ending)
        if path.is_file():
            try:
                result = json.loads(path.read_text())
            except ValueError:
                continue
            if result.get("id") == video_id:
                return result
    link = f"https://www.youtube.com/watch?v={video_id}"
    result = subprocess.run([*YTDLP, "--skip-download", "--dump-single-json", link],
                            check=True, capture_output=True, text=True, timeout=100)
    info = json.loads(result.stdout)
    write_json(ROOT / "metadata" / (video_id + ".bulk.json"), info)
    return info


def download_captions(track):
    for attempt in range(5):
        try:
            sleep(3)
            with urlopen(Request(track["url"], headers=track.get("http_headers", {})), timeout=25) as reply:
                return json.load(reply)
        except HTTPError as exc:
            if exc.code != 429 or attempt == 4:
                raise
            retry_after = exc.headers.get("Retry-After", "")
            delay = max(int(retry_after) if retry_after.isdecimal() else 0, 90 * (attempt + 1))
            print(f"YouTube caption rate limit. Waiting {delay}s before retry.", flush=True)
            sleep(delay)


def download(entry):
    video_id = entry["id"]
    status_path = STATUS / f"{video_id}.json"
    if status_path.exists():
        previous = json.loads(status_path.read_text())
        if previous["status"] in {"downloaded", "no_french_track", "empty_french_track"}:
            return
    if shutil.disk_usage(ROOT).free < STOP_FREE_BYTES:
        return
    base = {"video_id": video_id, "url": f"https://www.youtube.com/watch?v={video_id}",
            "title": entry.get("title"), "channel_id": CHANNEL_ID}
    try:
        info = metadata(video_id)
        if info.get("channel_id") != CHANNEL_ID:
            raise ValueError("Unexpected source channel")
        track, caption_kind = get_track(info)
        if track is None:
            write_json(status_path, {**base, "status": "no_french_track"})
            return
        captions = MEDIA / f"{video_id}.fr.json3"
        if not captions.is_file():
            data = download_captions(track)
            if not cue_segments(data["events"]):
                write_json(status_path, {**base, "status": "empty_french_track"})
                return
            write_json(captions, data)
        video = MEDIA / f"{video_id}.mp4"
        if not video.is_file():
            subtype = "bestvideo[height<=720][ext=mp4][vcodec^=avc1]/bestvideo[height<=720][ext=mp4]/best[height<=720][ext=mp4]"
            subprocess.run([*YTDLP, "--continue", "-f", subtype, "-o", str(video), base["url"]],
                           check=True, capture_output=True, text=True, timeout=330)
        if not video.is_file() or not video.stat().st_size:
            raise ValueError("Video file absent after download")
        with video.open("rb") as file:
            digest = hashlib.file_digest(file, "sha256").hexdigest()
        write_json(status_path, {**base, "status": "downloaded", "caption_kind": caption_kind,
                                 "video": str(video), "captions": str(captions), "sha256": digest,
                                 "duration_s": info.get("duration"), "upload_date": info.get("upload_date")})
        print(f"{video_id}: {caption_kind}, video downloaded", flush=True)
    except (ValueError, OSError, KeyError, subprocess.SubprocessError) as exc:
        # Subprocess command lines and signed subtitle URLs are excluded from the manifest.
        message = exc.stderr.splitlines()[-1][:300] if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else type(exc).__name__
        if isinstance(exc, HTTPError):
            message = f"Caption HTTP {exc.code}"
        write_json(status_path, {**base, "status": "failed", "error": message})
        print(f"{video_id}: failed ({type(exc).__name__})", flush=True)


def acquisition():
    listing = json.loads((ROOT / "channel.json").read_text())
    entries = {e["id"]: e for e in listing["entries"] if e.get("id")}
    def priority(entry):
        status_path = STATUS / f"{entry['id']}.json"
        if not status_path.exists():
            return 0
        try:
            status = json.loads(status_path.read_text()).get("status")
        except (OSError, ValueError):
            return 0
        return 1 if status == "failed" else 2

    # Reach untouched uploads before spending time retrying access failures.
    for entry in sorted(entries.values(), key=priority):
        if shutil.disk_usage(ROOT).free < STOP_FREE_BYTES:
            print("Disk reserve reached. Remaining uploads are pending.", flush=True)
            break
        download(entry)
    write_json(ROOT / "acquisition-finished.json", {"enumerated": len(entries), "counts": counts()})


def counts():
    return dict(Counter(json.loads(path.read_text())["status"] for path in STATUS.glob("*.json")))


def processing():
    module = load_extractor(Path(".runtime/matignon-cv-review/extract_poses.py"))
    failed = POSES / "failures"
    failed.mkdir(parents=True, exist_ok=True)
    while True:
        for path in sorted(STATUS.glob("*.json")):
            row = json.loads(path.read_text())
            if row["status"] != "downloaded":
                continue
            video_id = row["video_id"]
            if (POSES / f"{video_id}.summary.json").exists() or (failed / f"{video_id}.json").exists():
                continue
            try:
                report = extract(Path(row["video"]), Path(row["captions"]), POSES, module,
                                 source_id=video_id, title=row["title"], caption_kind=row["caption_kind"],
                                 channel_id=row["channel_id"], video_sha256=row["sha256"])
                print(f"{video_id}: {report['candidate_windows']} subtitle/pose candidates", flush=True)
            except (ValueError, OSError, TypeError, KeyError, IndexError, subprocess.SubprocessError) as exc:
                reason = str(exc).splitlines()[0][:300]
                write_json(failed / f"{video_id}.json", {"error": reason})
                print(f"{video_id}: {reason}", flush=True)
        progress = {
            "acquisition": counts(),
            "pose_videos": len(list(POSES.glob("*.summary.json"))),
            "pose_failures": len(list(failed.glob("*.json"))),
        }
        summaries = [json.loads(path.read_text()) for path in POSES.glob("*.summary.json")]
        progress["candidate_windows"] = sum(row["candidate_windows"] for row in summaries)
        write_json(ROOT / "progress.json", progress)
        if (ROOT / "acquisition-finished.json").exists() and (
            progress["pose_videos"] + progress["pose_failures"] >= progress["acquisition"].get("downloaded", 0)
        ):
            return
        sleep(5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("acquire", "extract"))
    args = parser.parse_args()
    for directory in (STATUS, MEDIA, POSES):
        directory.mkdir(parents=True, exist_ok=True)
    if args.stage == "acquire":
        (ROOT / "acquisition-finished.json").unlink(missing_ok=True)
        acquisition()
    else:
        processing()


if __name__ == "__main__":
    main()
