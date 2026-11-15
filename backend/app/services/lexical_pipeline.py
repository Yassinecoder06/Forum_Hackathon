"""Run the pinned upstream lexical pose pipeline in its isolated runtime."""
import hashlib
import json
import os
import subprocess
import uuid
from pathlib import Path
from threading import BoundedSemaphore

from .capture_library import ROOT, storage

LOCK = BoundedSemaphore(1)


def run(text):
    script = ROOT / "integration/spoken_pipeline/pipeline.py"
    runtime = Path(os.getenv("SIGNACULTURE_LEXICAL_PYTHON", str(script.parent / ".venv/bin/python")))
    if not script.is_file() or not runtime.is_file():
        raise FileNotFoundError("Le pipeline de poses local est absent. Exécutez integration/spoken_pipeline/setup.sh.")
    if not LOCK.acquire(blocking=False):
        raise BlockingIOError("Une séquence de poses est déjà en cours de génération.")
    try:
        base = storage() / "lexical_pipeline"
        lexicon = base / "lexicon"
        lexicon.mkdir(parents=True, exist_ok=True)
        library = storage() / "library.json"
        fingerprint = hashlib.sha256(library.read_bytes() if library.exists() else b"[]").hexdigest()
        manifest = lexicon / "manifest.json"
        current = json.loads(manifest.read_text()) if manifest.is_file() else {}
        if current.get("source_library_sha256") != fingerprint:
            export = subprocess.run(
                [str(runtime), str(script), "export", "--output", str(lexicon),
                 "--library-root", str(ROOT), "--allow-local-preview"],
                cwd=ROOT, capture_output=True, text=True, timeout=180,
            )
            (base / "export.log").write_text(export.stdout + export.stderr)
            if export.returncode:
                raise RuntimeError("L’export du lexique de poses a échoué.")
        identifier = str(uuid.uuid4())
        folder = base / "runs" / identifier
        folder.mkdir(parents=True)
        result = subprocess.run(
            [str(runtime), str(script), "render", "--lexicon", str(lexicon),
             "--output", str(folder), "--text", text],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
        )
        (folder / "execution.log").write_text(result.stdout + result.stderr)
        try:
            report = json.loads((folder / "result.json").read_text())
        except (OSError, ValueError):
            raise RuntimeError("Le moteur de poses n’a pas produit de résultat lisible.")
        if report.get("status") == "blocked_unknown_or_ambiguous":
            words = report.get("unknowns", []) + [a["token"] for a in report.get("ambiguities", [])]
            raise ValueError("Aucune séquence complète : mots absents ou ambigus — " + ", ".join(words))
        if result.returncode:
            raise RuntimeError("La génération des poses a échoué.")
        urls = {kind + "_url": f"/pipeline/lexical/{identifier}/{kind}" for kind in ("pose", "video", "gif", "report")}
        return {
            "run_id": identifier, "glosses": report["glosses"],
            "frames": report["frames"], "fps": report["fps"],
            "translation_validated": False, "local_preview_only": report["local_preview_only"],
            "upstream_revision": report["upstream_revision"], **urls,
        }
    finally:
        LOCK.release()


def artifact(identifier, kind):
    try:
        identifier = str(uuid.UUID(identifier))
    except (ValueError, AttributeError):
        raise FileNotFoundError()
    names = {
        "pose": ("sequence.pose", "application/octet-stream"),
        "video": ("diagnostic.mp4", "video/mp4"),
        "gif": ("diagnostic.gif", "image/gif"),
        "report": ("result.json", "application/json"),
    }
    if kind not in names:
        raise FileNotFoundError()
    name, mime = names[kind]
    path = storage() / "lexical_pipeline/runs" / identifier / name
    if not path.is_file():
        raise FileNotFoundError()
    return path, mime
