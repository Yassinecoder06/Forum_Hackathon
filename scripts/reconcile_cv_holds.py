"""Reconcile genuine detector frames without fabricating missing landmarks.

Preserve a whole original frame if the new extraction loses any old observation;
use the corrected frame otherwise. Timestamp union contains only actual samples.
Default builds candidates. --apply activates verified research previews only.
"""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from backend.app.services import capture_library as store
from scripts.activate_cv_corpus import validate

POINT_KEYS = ('PLM', 'PWLM', 'FLM', 'LLM', 'RLM')


def reconcile(before, candidate):
    old = {f['time_ms']: f for f in before}
    new = {f['time_ms']: f for f in candidate}
    assert len(old) == len(before) and len(new) == len(candidate), 'Duplicate timestamps'
    result = []
    previous = 0
    for t in sorted(old.keys() | new.keys()):
        a, b = old.get(t), new.get(t)
        fallback = b is None or (a and any(a.get(k) and not b.get(k) for k in POINT_KEYS))
        selected = a if fallback else b
        f = deepcopy(selected)
        f['capture_origin'] = 'baseline_frame_preserved' if fallback else 'corrected_detector_frame'
        f['dt'] = (t - previous) / 1000 if result else 0
        previous = t
        f['hand_diagnostics'] = deepcopy(f.get('hand_diagnostics', {}))
        for side, key in [('left', 'LLM'), ('right', 'RLM')]:
            f['hand_diagnostics'].setdefault(side, {'source':'legacy_detector_frame_preserved','review_required':True})
            f['hand_diagnostics'][side]['observed'] = bool(f.get(key))
        result.append(f)
    bytime = {f['time_ms']:f for f in result}
    for a in before:
        b = bytime[a['time_ms']]
        assert all(not a.get(k) or b.get(k) for k in POINT_KEYS), 'Observation loss'
        if b['capture_origin'] == 'baseline_frame_preserved':
            assert all(a.get(k) == b.get(k) for k in POINT_KEYS), 'Original coordinates changed'
    return result


def native_frames(raw, original, candidate):
    old = {f['time_ms']:f for f in original['frames']}
    new = {f['time_ms']:f for f in candidate['frames']}
    frames = []
    for f in raw:
        selected = old if f['capture_origin'] == 'baseline_frame_preserved' else new
        pose = selected.get(f['time_ms'])
        if pose is None:
            assert not f.get('PLM'), 'Observed body has no matching actual native pose'
            continue
        pose = deepcopy(pose)
        pose['capture_origin'] = f['capture_origin']
        pose['hand_tracking'] = f['hand_diagnostics']
        frames.append(pose)
    assert frames, 'No actual native frames'
    # An endpoint hold is playback metadata, never an observed raw sample.
    duration = max(original['duration_ms'], candidate['duration_ms'])
    if frames[-1]['time_ms'] < duration:
        last = deepcopy(frames[-1]); last['time_ms'] = duration
        last['playback_origin'] = 'endpoint_hold'; frames.append(last)
    return frames, duration


def native_joint_metadata(frames):
    # Required keys must exist in every frame. Sparse tracked fingers stay optional.
    required=sorted(set.intersection(*(set(f['joints']) for f in frames)))
    tracked=sorted({k for f in frames for k in f['joints']})
    return required,tracked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    batch = ROOT / '.runtime/cv-v4-all'
    by_capture = {a['capture_id']:(r,a) for r in json.loads((batch/'results.json').read_text())['results'] for a in r.get('aliases',[])}
    output = ROOT / '.runtime/cv-v6-reconcile'
    output.mkdir(exist_ok=True)
    report = {'applied':args.apply, 'accuracy_validated':False,'linguistic_review':'pending','method':'whole-frame baseline fallback at genuine source timestamps','entries':[]}
    with store.library_lock():
        registry = store.library()
        work = []
        for row in registry:
            if row.get('cv_generation'):
                continue
            result, alias = by_capture[row['capture_id']]
            assert result['status'] == 'complete'
            source = Path(alias['source'])
            assert hashlib.sha256(source.read_bytes()).hexdigest() == result['source_sha256']
            raw_before = Path(alias['baseline_raw'])
            raw_new = batch/'captures'/result['source_sha256']/'animics-landmarks.json.gz'
            before = json.load(gzip.open(raw_before,'rt'))
            candidate = json.load(gzip.open(raw_new,'rt'))
            raw = reconcile(before,candidate)
            target = Path(row['motion']['path'])
            if not target.is_absolute(): target = ROOT/target
            assert target.resolve().is_relative_to(store.storage())
            original = json.loads(target.read_text())
            motion = json.loads((raw_new.parent/'motion.json').read_text())
            motion['frames'], motion['duration_ms'] = native_frames(raw,original,motion)
            motion['attribution'] = original.get('attribution',row.get('attribution'))
            motion['capture_method'] = 'cv-v6-corrected-with-authentic-baseline-frame-fallback'
            motion['cv_review'] = 'research_preview_pending_accuracy_review'
            motion['cv_source_sha256'] = result['source_sha256']
            gaps = [b['time_ms']-a['time_ms'] for a,b in zip(raw,raw[1:])]
            motion['fps'] = 1000/statistics.median(gaps)
            motion['joints'],motion['tracked_joints'] = native_joint_metadata(motion['frames'])
            motion['quality'].update(sampled_frames=len(raw),pose_frames=sum(bool(f.get('PLM')) for f in raw),pose_coverage=sum(bool(f.get('PLM')) for f in raw)/len(raw),face_coverage=sum(bool(f.get('FLM')) for f in raw)/len(raw),missing_frames=sum(not bool(f.get('PLM')) for f in raw))
            for side,key in [('left','LLM'),('right','RLM')]:
                motion['quality'][side+'_hand_coverage'] = sum(bool(f.get(key)) for f in raw)/len(raw)
            folder = output/row['capture_id'];folder.mkdir(exist_ok=True)
            raw_path = folder/'animics-landmarks.json.gz'
            with gzip.open(raw_path,'wt',compresslevel=1) as handle: json.dump(raw,handle,allow_nan=False)
            store.write_json(folder/'motion.json',motion)
            checked = validate(raw_path,motion)
            entry = {'capture_id':row['capture_id'],'source_id':row.get('source_id'),'source_sha256':result['source_sha256'],'baseline_raw_sha256':hashlib.sha256(raw_before.read_bytes()).hexdigest(),'corrected_raw_sha256':hashlib.sha256(raw_new.read_bytes()).hexdigest(),'output_raw_sha256':hashlib.sha256(raw_path.read_bytes()).hexdigest(),'baseline_frames':len(before),'corrected_frames':len(candidate),'union_actual_frames':checked,'whole_baseline_frames_preserved':sum(f['capture_origin']=='baseline_frame_preserved' for f in raw),'old_observations_lost':0,'approval':'pending'}
            report['entries'].append(entry);work.append((row,motion,target,raw_path))
        if args.apply:
            backup = output/'library-before.json'
            if not backup.exists():shutil.copy2(store.storage()/'library.json',backup)
            for row,motion,target,raw_path in work:
                backup = output/'backups'/target.name;backup.parent.mkdir(exist_ok=True)
                if not backup.exists():shutil.copy2(target,backup)
                store.write_json(target,motion)
                row['quality'] = motion['quality']
                row['motion'].update(fps=motion['fps'],duration_ms=motion['duration_ms'],quality_status='cv_reconciled_baseline_fallback_pending_review',animation_review='pending')
                row['raw_animics_path'] = str(raw_path)
                row['cv_generation'] = motion['capture_method']
            store.write_json(store.storage()/'library.json',registry)
        report['count'] = len(work)
        report['active_corrected_count'] = sum(bool(r.get('cv_generation')) for r in registry)
        report['baseline_frames_preserved'] = sum(r['whole_baseline_frames_preserved'] for r in report['entries'])
    report_path=output/('applied.json' if args.apply else 'candidates.json')
    if report['count'] or not report_path.exists():store.write_json(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k!='entries'}))


if __name__ == '__main__':main()
