"""Apply reversible playback-only masks/windows; original coordinates stay exact."""
import hashlib,json,shutil
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from backend.app.services import capture_library as store
from backend.app.services.sign_core import clip_windows


def main():
    folder=ROOT/'.runtime/motion-quality';audit=json.loads((folder/'audit.json').read_text());records={r['id']:r for r in audit['records']};applied=[]
    with store.library_lock():
        rows=store.library()
        snapshot=folder/'library-before.json'
        if not snapshot.exists():shutil.copy2(store.storage()/'library.json',snapshot)
        for row in rows:
            identifier=row['motion']['motion_id'];p=Path(row['motion']['path']);p=p if p.is_absolute() else ROOT/p
            original=json.loads(p.read_text());motion=json.loads(p.read_text());record=records[identifier]
            backup=folder/'backups'/p.name;backup.parent.mkdir(exist_ok=True)
            if not backup.exists():shutil.copy2(p,backup)
            bytime={f['time_ms']:f for f in motion['frames']}
            events=record['events'];masked=0
            for event in events:
                if event['kind']!='isolated_finger_length_burst':continue
                f=bytime[event['time_ms']];quality=f.setdefault('playback_quality',{})
                quality[event['side']+'_hand_rejected']=True;quality['reason']='isolated multi-segment length burst; source retained';masked+=1
            (start,end),=clip_windows([p])
            motion['playback_window']={'start_ms':start,'end_ms':end,'method':'captured-wrist core with 120ms lead and 180ms release; pending linguistic review'}
            motion['playback_review']='engineering_preview_pending_human_review'
            assert [f['joints'] for f in motion['frames']]==[f['joints'] for f in original['frames']]
            assert [f.get('hand_tracking') for f in motion['frames']]==[f.get('hand_tracking') for f in original['frames']]
            store.write_json(p,motion)
            row.setdefault('quality',{})['playback_masked_hand_frames']=masked
            applied.append({'id':identifier,'before_sha256':hashlib.sha256(backup.read_bytes()).hexdigest(),'after_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'masked_hand_frames':masked,'window':[start,end],'original_coordinates_unchanged':True})
        store.write_json(store.storage()/'library.json',rows)
    report={'clips':len(applied),'masked_hand_frames':sum(r['masked_hand_frames'] for r in applied),'raw_files_modified':False,'original_coordinates_unchanged':True,'linguistic_review':'pending','records':applied}
    store.write_json(folder/'applied.json',report);print(json.dumps({k:v for k,v in report.items() if k!='records'}))


if __name__=='__main__':main()
