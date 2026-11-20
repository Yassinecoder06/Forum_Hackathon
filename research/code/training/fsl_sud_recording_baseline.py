"""Leave one recording out on the local France-LSF SUD annotations."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata

from training.lsfb_cont.scoring import distance


def read_records(source):
    records = []
    for path in sorted(source.glob("*.conllu")):
        blocks = path.read_text().strip().split("\n\n")
        for block in blocks:
            metadata, forms = {}, []
            for line in block.splitlines():
                if line.startswith("# ") and " = " in line:
                    key, value = line[2:].split(" = ", 1)
                    metadata[key] = value
                elif not line.startswith("#"):
                    fields = line.split("\t")
                    if fields[0].isdigit():
                        forms.append(fields[1])
            if not metadata.get("text_fr") or not forms:
                continue
            recording = metadata.get("source_file") or metadata.get("video_url")
            if not recording or not metadata.get("sent_id"):
                raise ValueError(f"Missing recording or segment identity in {path}")
            records.append((recording, metadata["sent_id"], metadata["text_fr"], forms))
    if len({row[1] for row in records}) != len(records):
        raise ValueError("Duplicate segment identity")
    return records


def features(text):
    words = re.findall(r"\w+", unicodedata.normalize("NFKC", text).casefold())
    return Counter(tuple(words[i:i+n]) for n in (1, 2) for i in range(len(words)-n+1))


def retrieve(training, queries):
    counts = [features(row[2]) for row in training]
    document_frequency = Counter(term for row in counts for term in row)
    idf = {term: 1 + math.log((1 + len(counts)) / (1 + count))
           for term, count in document_frequency.items()}

    def vector(values):
        weights = {term: count * idf[term] for term, count in values.items() if term in idf}
        norm = math.sqrt(sum(value**2 for value in weights.values())) or 1
        return {term: value / norm for term, value in weights.items()}

    vectors = [vector(row) for row in counts]
    for query in queries:
        weights = vector(features(query[2]))
        scores = [sum(value * row.get(term, 0) for term, value in weights.items()) for row in vectors]
        best = max(range(len(scores)), key=scores.__getitem__)
        yield training[best][3] if scores[best] else []


def evaluate(records):
    folds = []
    for recording in sorted({row[0] for row in records}):
        training = [row for row in records if row[0] != recording]
        held_out = [row for row in records if row[0] == recording]
        vocabulary = {form for row in training for form in row[3]}
        common = list(Counter(tuple(row[3]) for row in training).most_common(1)[0][0])
        references = [row[3] for row in held_out]
        predictions = list(retrieve(training, held_out))
        folds.append({
            "held_out_documents": sorted({row[1].rsplit("__", 1)[0] for row in held_out}),
            "train_examples": len(training),
            "examples": len(held_out),
            "reference_forms": sum(map(len, references)),
            "unseen_reference_forms": sum(form not in vocabulary for ref in references for form in ref),
            "fully_covered_examples": sum(set(ref) <= vocabulary for ref in references),
            "retrieval_exact": sum(ref == pred for ref, pred in zip(references, predictions, strict=True)),
            "retrieval_edits": sum(distance(ref, pred) for ref, pred in zip(references, predictions, strict=True)),
            "constant_exact": sum(ref == common for ref in references),
            "constant_edits": sum(distance(ref, common) for ref in references),
            "retrieval_empty": sum(not pred for pred in predictions),
        })
    return folds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/text_to_gloss/raw/sud"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = read_records(args.source)
    if len({row[0] for row in records}) < 2:
        parser.error("Need at least two source recordings")
    result = {
        "sign_language": "fsl",
        "annotation_convention": "SUD FORM including annotated neutral-position tokens",
        "linguistically_validated": False,
        "license": "CC BY-SA 4.0",
        "input_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for path in sorted(args.source.glob("*.conllu"))},
        "folds": evaluate(records),
    }
    with args.output.open("x") as output:
        json.dump(result, output, indent=2)
        output.write("\n")


if __name__ == "__main__":
    main()
