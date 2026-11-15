"""Derive roll-compensated facial features from immutable actual raw face points."""
import argparse
from copy import deepcopy
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.app.services import capture_library as store
from backend.app.workers.face_features import measure_face
METHOD='eye-normal-projection-legacy-scale-v1'


def derive(motion,raw,source_dimensions=None):
    result=deepcopy(motion);bytime={f['time_ms']:f for f in raw}
    dimensions=next((f for f in raw if f.get('image_width') and f.get('image_height')),None)
    if not dimensions:dimensions=source_dimensions
    if not dimensions:raise ValueError('No source dimensions; do not guess face projection')
    count=0;delta=0
    for before,frame in zip(motion['frames'],result['frames']):
        f=bytime.get(frame['time_ms'])
        if not f:
            frame['playback_origin']='endpoint_hold'
            frame['face']=None
            continue
        face=f.get('FLM')
        frame['face']=measure_face(face,f.get('image_width',dimensions['image_width']),f.get('image_height',dimensions['image_height'])) if face else None
        if face and frame['face']:
            count+=1
            if before.get('face'):
                delta=max(delta,*(abs(before['face'][k]-v) for k,v in frame['face'].items()))
        assert before['joints']==frame['joints'],'Body/fingers changed'
        assert before.get('hand_tracking')==frame.get('hand_tracking'),'Hand observations changed'
    result['face_feature_method']=METHOD
    result['face_review']='pending_reference_accuracy_review'
    return result,count,delta


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--apply',action='store_true');a=p.parse_args()
    output=ROOT/'.runtime/cv-face-roll';output.mkdir(exist_ok=True)
    report={'applied':a.apply,'method':METHOD,'accuracy_validated':False,'entries':[]}
    with store.library_lock():
        registry=store.library();prepared=[]
        candidates={r['source_sha256']:r for r in json.loads((ROOT/'.runtime/cv-v4-all/results.json').read_text())['results']}
        for row in registry:
            raw=Path(row['raw_animics_path']);target=Path(row['motion']['path'])
            if not target.is_absolute():target=ROOT/target
            motion=json.loads(target.read_text())
            if motion.get('face_feature_method')==METHOD:continue
            frames=json.load(gzip.open(raw,'rt'));dimensions=None
            if not any(f.get('image_width') and f.get('image_height') for f in frames):
                source_sha=motion['cv_source_sha256'];assert source_sha in candidates
                candidate_raw=ROOT/'.runtime/cv-v4-all/captures'/source_sha/'animics-landmarks.json.gz'
                dimensions=next(f for f in json.load(gzip.open(candidate_raw,'rt')) if f.get('image_width') and f.get('image_height'))
            updated,count,delta=derive(motion,frames,dimensions)
            report['entries'].append({'capture_id':row['capture_id'],'source_id':row.get('source_id'),'face_frames':count,'max_absolute_feature_change':delta,'raw_sha256':hashlib.sha256(raw.read_bytes()).hexdigest(),'body_fingers_unchanged':True,'hand_observation_flags_unchanged':True})
            prepared.append((row,target,updated))
        if a.apply:
            for row,target,motion in prepared:
                backup=output/'backups'/target.name;backup.parent.mkdir(exist_ok=True)
                if not backup.exists():shutil.copy2(target,backup)
                store.write_json(target,motion)
        report['count']=len(prepared)
    report_path=output/('applied.json' if a.apply else 'candidates.json')
    if report['count'] or not report_path.exists():store.write_json(report_path,report)
    print(json.dumps({k:v for k,v in report.items() if k!='entries'}))


if __name__=='__main__':main()
