"""Build the local reviewer interface from an existing reference CSV."""
import argparse
import hashlib
import json
from pathlib import Path

from collect_alignment_review import IDENTITY, RATINGS, worksheet


def render(reference, destination):
    rows = [{field: row[field] for field in IDENTITY} for row in worksheet(reference).values()]
    if not rows:
        raise ValueError("Expected at least one worksheet item")
    payload = json.dumps({
        "rows": rows, "fields": IDENTITY + RATINGS, "ratings": RATINGS,
        "key": "signaculture-alignment-" + hashlib.sha256(reference.read_bytes()).hexdigest(),
    }, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("alignment_review.html").read_text(encoding="utf-8")
    destination.write_text(template.replace("__WORKSHEET__", payload), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("destination", type=Path)
    arguments = parser.parse_args()
    render(arguments.reference, arguments.destination)
