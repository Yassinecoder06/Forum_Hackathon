"""Activate structurally verified CV previews, retaining regression review holds.

Defaults to dry-run. No approval, licence or source attribution is inferred.
"""
import argparse

import gzip
import hashlib
import json
import math
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services import capture_library as store


def decision(row, result, allow_unbaselined):
    if row.get('cv_generation'):
        return 'already_active'
    if not result or result.get('status') != 'complete':
        return 'no_verified_candidate'
    comparisons = [c for c in result['comparisons'] if c['capture_id'] == row['capture_id']]
    if not comparisons:
        return 'missing_comparison'
    gate = comparisons[0]['gate']
    if gate == 'no_observation_loss':
        return 'activate'
    if gate == 'review_no_raw_baseline' and allow_unbaselined:
        return 'activate_unbaselined_preview'
    return 'hold_' + gate


def validate(raw_path, motion):
    raw = json.load(gzip.open(raw_path, 'rt'))
    assert raw and motion['frames'], 'Empty capture'
    last = -1
    for frame in raw:
        assert frame['time_ms'] > last, 'Non-monotonic timestamps'
        last = frame['time_ms']
        for key, count in [('PLM', 33), ('PWLM', 33), ('FLM', 468), ('LLM', 21), ('RLM', 21)]:
            points = frame.get(key)
            if points is not None:
                assert len(points) == count
                assert all(math.isfinite(p[k]) for p in points for k in ['x', 'y', 'z'])
        assert frame.get('LWLM') is None and frame.get('RWLM') is None
        for side, key in [('left', 'LLM'), ('right', 'RLM')]:
            assert frame['hand_diagnostics'][side]['observed'] == bool(frame.get(key))
    previous = -1
    for frame in motion['frames']:
        assert frame['time_ms'] > previous
        previous = frame['time_ms']
        assert all(math.isfinite(v) for xyz in frame['joints'].values() for v in xyz)
    return len(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--allow-unbaselined-preview', action='store_true')
    args = parser.parse_args()
    batch = ROOT / '.runtime/cv-v4-all'
    results = json.loads((batch / 'results.json').read_text())['results']
    by_capture = {a['capture_id']:r for r in results for a in r.get('aliases', [])}
    output = ROOT / '.runtime/cv-corpus-activation'
    output.mkdir(exist_ok=True)
    report = {'applied':args.apply, 'keypoint_accuracy_validated':False, 'linguistic_review':'pending', 'entries':[]}
    with store.library_lock():
        registry = store.library()
        prepared = []
        for row in registry:
            result = by_capture.get(row.get('capture_id'))
            status = decision(row, result, args.allow_unbaselined_preview)
            record = {'capture_id':row.get('capture_id'), 'source_id':row.get('source_id'), 'motion_id':row['motion']['motion_id'], 'decision':status}
            report['entries'].append(record)
            if not status.startswith('activate'):
                continue
            folder = batch / 'captures' / result['source_sha256']
            motion_path = folder / 'motion.json'
            raw_path = folder / 'animics-landmarks.json.gz'
            motion = json.loads(motion_path.read_text())
            record['checked_frames'] = validate(raw_path, motion)
            record['source_sha256'] = result['source_sha256']
            record['raw_sha256'] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
            target = Path(row['motion']['path'])
            if not target.is_absolute():
                target = ROOT / target
            assert target.resolve().is_relative_to(store.storage()), 'Asset outside runtime'
            old = json.loads(target.read_text())
            motion.update(attribution=old.get('attribution',row.get('attribution')), cv_review='research_preview_pending_accuracy_review',cv_source_sha256=result['source_sha256'])
            prepared.append((row, motion, target, raw_path))
        # Validate the entire proposed update before modifying any active file.
        if args.apply:
            backup = output / 'library-before.json'
            if not backup.exists():
                shutil.copy2(store.storage() / 'library.json', backup)
            for row, motion, target, raw_path in prepared:
                backup = output / 'backups' / target.name
                backup.parent.mkdir(exist_ok=True)
                if not backup.exists():
                    shutil.copy2(target, backup)
                store.write_json(target, motion)
                row['quality'] = motion['quality']
                row['motion'].update(fps=motion['fps'],duration_ms=motion['duration_ms'],quality_status='cv_corrected_research_preview_pending_review',animation_review='pending')
                row['raw_animics_path'] = str(raw_path)
                row['cv_generation'] = motion['capture_method']
            store.write_json(store.storage() / 'library.json', registry)
        from collections import Counter
        report['counts'] = dict(Counter(r['decision'] for r in report['entries']))
        report['active_corrected_count'] = sum(bool(r.get('cv_generation')) for r in registry)
    store.write_json(output / ('applied.json' if args.apply else 'dry-run.json'), report)
    print(json.dumps({k:v for k,v in report.items() if k != 'entries'}))


if __name__ == '__main__':
    main()
