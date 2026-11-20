"""Extract full-signer or reviewed-ROI MediaPipe poses and weak subtitle pairs."""
import argparse
import html
from fractions import Fraction
import importlib.util
import json
from pathlib import Path
import re
import subprocess
from tempfile import TemporaryFile

import numpy as np


def cue_segments(events):
    """Turn JSON3 word timestamps into non-overlapping subtitle windows."""
    pieces = []
    for event in events:
        start = event.get("tStartMs")
        segments = event.get("segs") or []
        if not isinstance(start, (int, float)) or not segments:
            continue
        text = "".join(s.get("utf8", "") for s in segments)
        text = re.sub(r"\[[^\]]+\]", " ", html.unescape(text))
        text = " ".join(text.split())
        if not re.search(r"\w", text, re.UNICODE):
            continue
        offsets = [s.get("tOffsetMs", 0) for s in segments if s.get("utf8", "").strip()]
        last_word = max(offsets, default=0)
        duration = max(1, event.get("dDurationMs", 1500))
        end = start + min(duration, max(last_word + 1100, 1500))
        pieces.append((int(start), int(end), text))
    pieces.sort()
    merged = []
    for i, (start, end, text) in enumerate(pieces):
        if i + 1 < len(pieces):
            end = min(end, pieces[i + 1][0])
        if end <= start:
            continue
        if merged and merged[-1][2].casefold() == text.casefold() and start - merged[-1][1] < 350:
            merged[-1][1] = end
            continue
        if merged:
            a, b, previous = merged[-1]
            if start - b < 800 and end - a <= 10_000 and (not re.search(r"[.!?]$", previous) or b-a < 1200):
                merged[-1] = [a, end, f"{previous} {text}"]
                continue
        merged.append([start, end, text])
    return merged


def load_extractor(path):
    spec = importlib.util.spec_from_file_location("matignon_extractor", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def extract(video, track, output, extractor, *, source_id, caption_kind, roi=None,
            start_ms=0, end_ms=None, sample_fps=12.5, title="", channel_id="", video_sha256=None):
    cues = cue_segments(json.loads(track.read_text())["events"])
    cues = [(a, b, text) for a, b, text in cues if a >= start_ms and (end_ms is None or b <= end_ms)]
    if not cues:
        raise ValueError("No French subtitle windows in requested time span")
    first_ms, last_ms = cues[0][0], cues[-1][1]
    probe = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height,avg_frame_rate", "-of", "json", str(video),
    ], check=True, capture_output=True, text=True)
    stream = json.loads(probe.stdout)["streams"][0]
    width, height = stream["width"], stream["height"]
    fps = float(Fraction(stream["avg_frame_rate"]))
    module = extractor
    mp, vision = module.mp, module.vision
    options = vision.HolisticLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=module.MODEL),
        running_mode=vision.RunningMode.VIDEO,
    )
    bounds = [round(p * dim) for p, dim in zip(roi or [0, 0, 1, 1], [width, height, width, height])]
    x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
        raise ValueError("Interpreter crop lies outside the source frame")
    scale = float(np.clip(560 / w, 1, 2))
    out_width, out_height = round(w * scale), round(h * scale)
    filters = f"fps={sample_fps},crop={w}:{h}:{x0}:{y0}:exact=1,scale={out_width}:{out_height}"
    command = [
        "ffmpeg", "-v", "error", "-nostdin", "-threads", "2", "-filter_threads", "1",
        "-ss", f"{first_ms / 1000:.3f}", "-i", str(video), "-t", f"{(last_ms - first_ms) / 1000:.3f}",
        "-vf", filters, "-map", "0:v:0", "-an", "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1",
    ]
    pts, has, times = [], [], []
    with TemporaryFile() as errors:
        decoder = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors)
        byte_count = out_width * out_height * 3
        try:
            with vision.HolisticLandmarker.create_from_options(options) as model:
                while raw := decoder.stdout.read(byte_count):
                    if len(raw) != byte_count:
                        raise ValueError("FFmpeg returned a truncated frame")
                    stamp = first_ms + round(len(times) * 1000 / sample_fps)
                    rgb = np.frombuffer(raw, np.uint8).reshape(out_height, out_width, 3)
                    detected = model.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), stamp)
                    xyz, flags = module.assemble(detected)
                    pts.append(xyz.astype(np.float16))
                    has.append(flags)
                    times.append(stamp)
        finally:
            decoder.stdout.close()
            if decoder.poll() is None:
                decoder.wait(timeout=20)
        if decoder.returncode:
            errors.seek(0)
            raise ValueError(errors.read(600).decode(errors="replace"))
    if not times:
        raise ValueError("No video samples in subtitle span")
    timestamps = np.array(times, dtype=np.int64)
    detected = np.array(has, dtype=bool)
    indices = np.searchsorted(timestamps, [(a, b) for a, b, _ in cues])
    output.mkdir(parents=True, exist_ok=True)
    pose_path = output / f"{source_id}.npz"
    tmp = pose_path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, pts=np.stack(pts), has=detected, t_ms=timestamps)
    tmp.replace(pose_path)
    samples = []
    for (a, b, text), (start, end) in zip(cues, indices):
        mask = detected[start:end]
        if len(mask) < 5:
            continue
        pose = float(mask[:, 0].mean())
        any_hand = float((mask[:, 1] | mask[:, 2]).mean())
        both = float((mask[:, 1] & mask[:, 2]).mean())
        complete = timestamps[start] - a <= 160 and b - timestamps[end - 1] <= 160
        samples.append({
            "video_id": source_id, "start_ms": a, "end_ms": b, "french_text": text,
            "pose_file": pose_path.name, "frame_start": int(start), "frame_stop": int(end),
            "pose_coverage": pose, "any_hand_coverage": any_hand, "both_hand_coverage": both,
            "caption_kind": caption_kind, "alignment": "subtitle-time weak alignment",
            "candidate": bool(complete and pose >= .85 and any_hand >= .5 and len(text.split()) >= 3),
        })
    rows_path = output / f"{source_id}.pairs.jsonl"
    tmp = rows_path.with_suffix(".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in samples))
    tmp.replace(rows_path)
    stats = {
        "video_id": source_id, "url": f"https://www.youtube.com/watch?v={source_id}",
        "title": title, "channel_id": channel_id, "video_sha256": video_sha256,
        "caption_kind": caption_kind, "language": "LSF",
        "coordinate_space": "MediaPipe image-normalized XYZ within recorded ROI; Y down",
        "layout": "23 body + 21 left hand + 21 right hand + 92 face",
        "crop_xyxy_pixels": bounds, "source_fps": fps, "requested_sample_fps": sample_fps,
        "source_span_ms": [times[0], times[-1]], "frames": len(times),
        "pose_frames": int(detected[:, 0].sum()), "left_frames": int(detected[:, 1].sum()),
        "right_frames": int(detected[:, 2].sum()), "both_hand_frames": int((detected[:, 1] & detected[:, 2]).sum()),
        "face_frames": int(detected[:, 3].sum()), "subtitle_windows": len(samples),
        "candidate_windows": sum(r["candidate"] for r in samples),
        "linguistic_review": "pending", "reuse_rights": "not verified",
    }
    write_json(output / f"{source_id}.summary.json", stats)
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("captions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--start-ms", type=int, default=0)
    parser.add_argument("--end-ms", type=int)
    parser.add_argument("--roi", type=float, nargs=4)
    parser.add_argument("--extractor", type=Path, default=Path(".runtime/matignon-cv-review/extract_poses.py"))
    args = parser.parse_args()
    result = extract(args.video, args.captions, args.output, load_extractor(args.extractor),
                     source_id=args.video.stem, caption_kind="youtube_french_auto",
                     start_ms=args.start_ms, end_ms=args.end_ms, roi=args.roi)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
