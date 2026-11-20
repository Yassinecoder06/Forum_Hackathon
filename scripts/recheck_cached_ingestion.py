"""Replay one real local capture per source in an isolated, offline library."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts import ingest_commons_signs as commons, ingest_dictasign as dicta
from backend.app.services import capture_library as store

ROOT=Path(__file__).resolve().parents[1]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def forbid(*args,**kwargs):raise AssertionError('Network is forbidden in cached replay')


def main():
    original=ROOT/'.runtime';library_hash=sha(original/'library.json')
    selected=[]
    for kind,filename,urlkey,required in [('commons','commons_lsf.json','source_url',['source.mp4','motion.json']),('dictasign','dictasign_lsf.json','video_url',['source.mp4','motion.json','animics-landmarks.json.gz'])]:
        entries=json.loads((ROOT/'data/source_inventory'/filename).read_text())['entries']
        for e in entries:
            if kind=='commons' and not e['eligible_for_motion']:continue
            job=str(uuid.uuid5(uuid.NAMESPACE_URL,e[urlkey]));folder=original/'captures'/job
            if not all((folder/f).exists() for f in required):continue
            motion=json.loads((folder/'motion.json').read_text());q=motion['quality']
            if q['pose_coverage']<.8 or max(q['left_hand_coverage'],q['right_hand_coverage'])<.35:continue
            selected.append((kind,e,job,required));break
        else:raise ValueError('No reusable capture for '+kind)
    before={str(original/'captures'/job/f):sha(original/'captures'/job/f) for _,_,job,required in selected for f in required}
    commons.fetch=dicta.fetch=forbid
    results=[]
    with tempfile.TemporaryDirectory(prefix='signaculture-cached-ingestion-') as temp:
        os.environ['SIGNACULTURE_STORAGE']=temp
        for kind,e,job,required in selected:
            folder=Path(temp)/'captures'/job;folder.mkdir(parents=True)
            for f in required:shutil.copy2(original/'captures'/job/f,folder/f)
            result=commons.ingest(e) if kind=='commons' else dicta.extract_entry(e)
            assert result['status']=='imported'
            row=next(r for r in store.library() if r['capture_id']==job)
            assert row['sign_language']=='fsl'
            assert row['gloss']['review_status']=='pending'
            assert row['motion']['animation_review']=='pending'
            assert row['motion']['synthetic'] is False
            if kind=='dictasign':assert row['motion']['rights_status']=='pending'
            results.append(dict(source=kind,capture_id=job,source_sha256=sha(folder/'source.mp4'),
                                registered_language=row['sign_language'],rights_status=row['motion']['rights_status'],
                                review_status=row['gloss']['review_status'],synthetic=row['motion']['synthetic'],
                                quality=row['quality'],status='imported in isolated offline library'))
    assert sha(original/'library.json')==library_hash
    for name,digest in before.items():assert sha(Path(name))==digest
    report=dict(status='complete',network_requests=0,active_library_unchanged=True,selected_capture_files_unchanged=True,results=results,
                limitations='Exercises registration and metadata on cached genuine captures; does not rerun CV or grant linguistic approval.')
    (ROOT/'docs/verification/CACHED_INGESTION_RECHECK.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
