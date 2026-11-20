"""Inventory prospective parallel CSVs without inferring annotation validity."""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def audit(path, *, sign_language, text_column='text', gloss_column='gloss'):
    if sign_language not in {'fsl', 'sfb', 'ssr'}:
        raise ValueError('An explicit supported sign-language identifier is required')
    path = Path(path)
    with path.open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames or []
        if text_column not in columns:
            raise ValueError(f'Missing text column: {text_column}')
        counts = collections.Counter()
        splits = collections.Counter()
        kinds = collections.Counter()
        for row in reader:
            language = row.get('sign_language', '').strip()
            if language and language != sign_language:
                raise ValueError('Record language disagrees with declared source language')
            text = row.get(text_column, '').strip()
            gloss = row.get(gloss_column, '').strip()
            counts['rows'] += 1
            counts['nonempty_text'] += bool(text)
            counts['nonempty_gloss'] += bool(gloss)
            counts['nonempty_pairs'] += bool(text and gloss)
            splits[row.get('split', '').strip() or '(unspecified)'] += 1
            kinds[row.get('gloss_type', '').strip() or '(empty)'] += 1
    return dict(sign_language=sign_language, columns=columns, counts=dict(counts),
                split_counts=dict(splits), gloss_types=dict(kinds),
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                language_evidence='explicit source declaration; record labels checked when present',
                gold_supervision_verified=False,
                training_admission='blocked pending provenance, annotation and alignment review',
                missing_gloss_column=gloss_column not in columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv', type=Path)
    ap.add_argument('--sign-language', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    result = audit(args.csv, sign_language=args.sign_language)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
