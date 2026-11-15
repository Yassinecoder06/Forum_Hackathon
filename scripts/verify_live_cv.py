"""Validate every registered runtime capture with the actual application validator."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from backend.app.main import registry,asset
from backend.app.services import capture_library as store

root,_,motions=registry();indexed={m.motion_id:m for m in motions}
rows=store.library();old=json.loads((ROOT/'.runtime/cv-v6-reconcile/library-before.json').read_text())
assert len(rows)==len(old)
errors=[];faces=0;reconciled=0
for row,original in zip(rows,old):
    assert row['gloss']==original['gloss'] and row.get('attribution')==original.get('attribution')
    for key in ['motion_id','gloss_id','license','rights_status','synthetic']:
        assert row['motion'].get(key)==original['motion'].get(key)
    try:
        path=asset(root,indexed[row['motion']['motion_id']])
        motion=json.loads(path.read_text())
        assert motion.get('cv_source_sha256') and row.get('cv_generation')
        assert motion.get('face_feature_method')=='eye-normal-projection-legacy-scale-v1'
        faces+=1
        reconciled+=row['cv_generation']=='cv-v6-corrected-with-authentic-baseline-frame-fallback'
    except Exception as error:errors.append({'motion_id':row['motion']['motion_id'],'error':str(error)})
report={'registered_captures':len(rows),'application_asset_validator_passed':len(rows)-len(errors),'face_method_count':faces,'reconciled_frame_fallback_count':reconciled,'gloss_rights_attribution_unchanged':True,'accuracy_validated':False,'linguistic_review':'pending','errors':errors}
store.write_json(ROOT/'.runtime/cv-completion/live-verification.json',report)
print(json.dumps(report));raise SystemExit(bool(errors))
