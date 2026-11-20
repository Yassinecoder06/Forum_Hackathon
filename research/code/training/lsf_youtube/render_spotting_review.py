"""Build a private HTML page to eyeball dictionary-sign spotting candidates.

Each row loops the dictionary recording next to the interpreter crop at the
matched time. Concepts that pass repeatedly across occurrences come first.

    PYTHONPATH=research/code .venv/bin/python -m training.lsf_youtube.render_spotting_review \
        SPOTTING_DIR VIDEO.mp4 POSES.summary.json .runtime/library.json --top 24
"""
import argparse
import collections
import html
import json
from pathlib import Path
import shutil
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("spotting", type=Path)
    p.add_argument("video", type=Path)
    p.add_argument("summary", type=Path)
    p.add_argument("library", type=Path)
    p.add_argument("--top", type=int, default=24)
    p.add_argument("--max-percentile", type=float, default=.05)
    a = p.parse_args()
    rows = [json.loads(line) for line in (a.spotting / "candidates.jsonl").open()]
    crop = json.loads(a.summary.read_text())["crop_xyxy_pixels"]
    captures = {r["gloss"]["gloss_id"]: r.get("capture_id") for r in json.loads(a.library.read_text())}
    root = a.library.parent

    by = collections.defaultdict(list)
    for r in rows:
        by[r["concept"]].append(r)
    consistency = {c: sum(x["null_percentile"] <= a.max_percentile for x in rs) / len(rs) for c, rs in by.items()}
    # Only concepts heard at least three times can show consistency; singletons are noise-prone.
    passing = [r for r in rows if r["null_percentile"] <= a.max_percentile and len(by[r["concept"]]) >= 3]
    passing.sort(key=lambda r: (-consistency[r["concept"]], r["null_percentile"], r["score"]))

    clips = a.spotting / "review"
    shutil.rmtree(clips, ignore_errors=True)
    clips.mkdir(parents=True)
    x0, y0, x1, y1 = crop
    items = []
    for i, r in enumerate(passing[:a.top]):
        start = max(0, r["match_ms"][0] - 400) / 1000
        length = (r["match_ms"][1] - r["match_ms"][0] + 800) / 1000
        out = clips / f"{i:03}-interpreter.mp4"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}", "-i", str(a.video), "-t", f"{length:.3f}",
                        "-vf", f"crop={x1 - x0}:{y1 - y0}:{x0}:{y0},scale=-2:360", "-an", "-c:v", "libx264",
                        "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(out)], check=True)
        source = root / "captures" / str(captures.get(r["gloss_id"])) / "source.mp4"
        ref = clips / f"{i:03}-dictionary.mp4"
        if source.is_file():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(source), "-vf", "scale=-2:360", "-an",
                            "-c:v", "libx264", "-preset", "veryfast", "-crf", "26", "-pix_fmt", "yuv420p", str(ref)], check=True)
        items.append((r, out.name, ref.name if ref.is_file() else None, consistency[r["concept"]], len(by[r["concept"]])))

    cards = "\n".join(f"""
<section><h2>{html.escape(r['concept'])} <small>heard “{html.escape(r['word'])}” at {r['spoken_ms']/1000:.1f}s · sign {r['lag_ms']/1000:+.1f}s later</small></h2>
<p>score {r['score']} · null percentile {r['null_percentile']} · this concept passes in {cons:.0%} of {n} occurrences · decoy {html.escape(r['decoy_concept'])} percentile {r['decoy_null_percentile']}</p>
<div class=pair><figure><video src="review/{ref}" autoplay loop muted playsinline></video><figcaption>Dictionary</figcaption></figure>
<figure><video src="review/{clip}" autoplay loop muted playsinline></video><figcaption>Interpreter (matched)</figcaption></figure></div>
<p class=verdict>Same sign? ☐ yes ☐ no ☐ unsure</p></section>""" if ref else "" for r, clip, ref, cons, n in items)
    (a.spotting / "review.html").write_text(f"""<!doctype html><meta charset=utf-8><title>Sign spotting review</title>
<style>body{{font:15px system-ui;margin:24px;max-width:980px;background:#fafafa;color:#222}}section{{background:#fff;border:1px solid #ddd;border-radius:8px;padding:12px 16px;margin:14px 0}}
.pair{{display:flex;gap:12px;flex-wrap:wrap}}video{{height:300px;background:#000;border-radius:4px}}small{{color:#666;font-weight:400}}figure{{margin:0}}</style>
<h1>Dictionary sign spotting — {html.escape(a.video.stem)}</h1>
<p>Unreviewed candidates from caption-guided matching. Similar motion is not proof of the same sign; please judge each pair.</p>{cards}""")
    print(f"{len(items)} review pairs -> {a.spotting / 'review.html'}")


if __name__ == "__main__":
    main()
