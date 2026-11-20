"""Verify a running Docker Studio with the bundled, attributed France-LSF sample."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen
from uuid import uuid4


def smoke(base, full):
    def request(path, **kwargs):
        with urlopen(Request(base + path, **kwargs), timeout=180) as response:
            return response.read()

    def get(path):
        return json.loads(request(path))

    def upload(path, filename, content, media, **fields):
        boundary = uuid4().hex
        header = "".join(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{key}"\r\n\r\n{value}\r\n'
            for key, value in fields.items()
        )
        header += (f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="{filename}"\r\n'
                   f'Content-Type: {media}\r\n\r\n')
        body = header.encode() + content + f'\r\n--{boundary}--\r\n'.encode()
        return json.loads(request(path, data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}))

    assert get("/healthz")["status"] == "ok"
    assert b"/studio/assets/" in request("/studio/")
    capabilities = get("/api/capabilities")
    assert capabilities["capture"], capabilities
    avatar = request("/api/avatar/model")
    assert avatar.startswith(b"glTF")
    source = request("/api/examples/bonjour.webm")
    job = upload("/api/captures", "bonjour.webm", source, "video/webm", rights_confirmed="true")
    path = "/api/captures/" + job["job_id"]
    end = time.monotonic() + 180
    while job["status"] not in ("ready", "failed") and time.monotonic() < end:
        time.sleep(0.5)
        job = get(path)
    assert job["status"] == "ready" and job["landmarks_available"], job
    motion = get(path + "/motion")
    raw = request(path + "/landmarks")
    landmarks = json.loads(gzip.decompress(raw))
    imported = upload("/api/captures", "landmarks.json.gz", raw, "application/gzip", rights_confirmed="true")
    restored = get("/api/captures/" + imported["job_id"] + "/motion")
    exact = all(
        (original["time_ms"], original["joints"], original["face"]) ==
        (reimported["time_ms"], reimported["joints"], reimported["face"])
        for original, reimported in zip(motion["frames"], restored["frames"], strict=True)
    )
    assert exact and len(landmarks) == motion["quality"]["sampled_frames"]
    report = {
        "avatar": capabilities["avatar"], "avatar_sha256": hashlib.sha256(avatar).hexdigest(),
        "video_capture_id": job["job_id"], "motion_frames": len(motion["frames"]),
        "landmark_frames": len(landmarks), "exact_pose_face_timing_roundtrip": exact,
    }
    if full:
        assert capabilities["emotion"] and capabilities["transcription"], capabilities
        audio = upload("/api/analyze-audio?include_emotion=true&include_transcription=true",
                       "bonjour.webm", source, "video/webm")
        assert audio["transcription"] and audio["transcription"]["text"] == "Bonjour.", audio["warnings"]
        scores = audio["emotion_hypothesis"]
        assert scores and len(scores) == 9 and abs(sum(scores.values()) - 1) < 1e-5, audio["warnings"]
        report.update(transcript=audio["transcription"]["text"], transcription_device=audio["transcription"]["device"],
                      emotion_categories=len(scores), audio_model_version=audio["model_version"])
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8088")
    parser.add_argument("--full", action="store_true", help="Also verify mounted local emotion and transcription models")
    parser.add_argument("--output", type=Path)
    options = parser.parse_args()
    result = json.dumps(smoke(options.base_url.rstrip("/"), options.full), indent=2) + "\n"
    if options.output:
        options.output.write_text(result)
    print(result, end="")
