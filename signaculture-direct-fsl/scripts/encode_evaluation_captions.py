"""Encode a compact cue-ID to French-caption JSON for offline evaluation."""
import argparse
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.text_encoder import french_features


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("captions", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    texts = json.loads(args.captions.read_text())
    if not isinstance(texts, dict) or not texts or any(not isinstance(text, str) or not text.strip() for text in texts.values()):
        raise ValueError("Expected a nonempty cue-ID to caption mapping")
    tensor = french_features(list(texts.values()), args.device).cpu()
    torch.save({"texts": texts, "features": dict(zip(texts, tensor))}, args.output)


if __name__ == "__main__":
    main()
