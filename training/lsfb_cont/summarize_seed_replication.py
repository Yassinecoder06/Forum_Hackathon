"""Aggregate frozen narrow LSFB replications without choosing a winning seed."""
import argparse
import hashlib
import json
import statistics
from pathlib import Path


def summarize(paths, retrieval_path):
    runs = []
    expected = None
    ids = None
    for path in paths:
        manifest = json.loads((path / 'MANIFEST.json').read_text())
        result = json.loads((path / 'RESULT.json').read_text())
        if result['status'] != 'complete':
            raise ValueError('Incomplete run')
        # Ignore only seed and output path: every learning setting must agree.
        matched = {k: v for k, v in manifest.items() if k != 'seed'}
        if expected is not None and matched != expected:
            raise ValueError('Protocol mismatch between seeds')
        expected = matched
        preds = json.loads((path / 'validation_predictions.json').read_text())
        signature = [(r['id'], r['reference']) for r in preds]
        if ids is not None and signature != ids:
            raise ValueError('Development examples or order differ')
        ids = signature
        if any(r['seed'] == manifest['seed'] for r in runs):
            raise ValueError('Duplicate seed')
        metrics = result['validation']
        runs.append(dict(seed=manifest['seed'], epoch=result['best']['epoch'],
                         exact=metrics['exact'], operation_wer=metrics['operation_wer'],
                         multi_sign=metrics['by_length']['multiple_signs'],
                         cap_hits=metrics['cap_hits'], seconds=result['seconds'],
                         result_sha256=hashlib.sha256((path / 'RESULT.json').read_bytes()).hexdigest()))
    retrieval = json.loads(retrieval_path.read_text())['current_french']
    if retrieval['examples'] != len(ids):
        raise ValueError('Retrieval sample count differs')
    # Full same-ID checks are included when the retrieval evidence supplies IDs.
    if [r['id'] for r in json.loads(retrieval_path.read_text())['current_french']['predictions']] != [r[0] for r in ids]:
        raise ValueError('Retrieval example IDs/order differ')
    aggregate = {}
    for key in ('exact', 'operation_wer'):
        values = [r[key] for r in runs]
        aggregate[key] = dict(mean=statistics.mean(values),
                              sample_sd=statistics.stdev(values) if len(values) > 1 else None,
                              minimum=min(values), maximum=max(values))
    return dict(status='complete', sign_language='sfb', task='narrow lexical projection',
                runs=runs, aggregate=aggregate,
                retrieval={k: v for k, v in retrieval.items() if k != 'predictions'},
                seeds_lower_wer_than_retrieval=sum(r['operation_wer'] < retrieval['operation_wer'] for r in runs),
                independent_test=False, qualified_review=False,
                limitations=['Repeatedly inspected 23-example development set, including only six multi-sign cases.',
                             'Each seed selects its lowest development WER; summaries are development diagnostics.',
                             'Seed replicates share examples and are not independent data samples.',
                             'Lexical targets omit nonmanual grammar and are pending linguistic review.'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', nargs='+', type=Path, required=True)
    ap.add_argument('--retrieval', type=Path, default=Path('.runtime/lsfb-narrow-context-data-v1/RETRIEVAL.json'))
    ap.add_argument('--output', type=Path, default=Path('reports/LSFB_SEED_REPLICATION_RESULT.json'))
    args = ap.parse_args()
    result = summarize(args.runs, args.retrieval)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
