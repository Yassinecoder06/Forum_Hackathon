"""Identify legacy France-LSF captures from local source records; no linguistic approval."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import uuid
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from backend.app.services import capture_library as store
ROOT=Path(__file__).resolve().parents[1]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def identify(row, inventory, runtime):
    sid=row.get('source_id')
    if sid not in inventory:raise ValueError('No source inventory match')
    entry,inventory_digest,kind=inventory[sid]
    url=entry['source_url'] if kind=='commons' else entry['video_url']
    if kind=='commons':
        if entry.get('language_variant')!='LSF_FR' or entry.get('eligible_for_motion') is not True:
            raise ValueError('Commons language/eligibility not explicit')
    elif entry.get('language')!='fsl':raise ValueError('Dicta-Sign language not explicit')
    job=str(uuid.uuid5(uuid.NAMESPACE_URL,url))
    if row.get('capture_id')!=job:raise ValueError('Source URL/capture ID mismatch')
    folder=runtime/'captures'/job
    if kind=='commons' and not (folder/'source_provenance.json').exists():
        # Early original Commons captures predate sidecars. Require exact bytes
        # against that file's recorded Commons checksum, not a filename guess.
        raw=(folder/'source.mp4').read_bytes()
        if not entry.get('source_sha1') or hashlib.sha1(raw).hexdigest()!=entry['source_sha1']:
            raise ValueError('Early Commons capture does not match recorded original checksum')
        return dict(source_id=sid,inventory_sha256=inventory_digest,source_sha256=hashlib.sha256(raw).hexdigest(),
                    language_basis='explicit France-LSF inventory and exact original Commons SHA1 match',
                    linguistic_review='pending; source identity does not establish semantic correctness')
    provenance=json.loads((folder/'source_provenance.json').read_text())
    expected=provenance.get('source_sha256') if kind=='dictasign' else provenance.get('sha256')
    if not expected or sha(folder/'source.mp4')!=expected:raise ValueError('Source video hash mismatch')
    if kind=='commons':
        if provenance.get('canonical_licence_source')!=url:raise ValueError('Source provenance URL mismatch')
    elif provenance.get('video_url')!=url or provenance.get('language')!='fsl':
        raise ValueError('Source provenance language/URL mismatch')
    return dict(source_id=sid,inventory_sha256=inventory_digest,source_sha256=expected,
                language_basis='explicit France-LSF source inventory and matching cached capture provenance',
                linguistic_review='pending; source identity does not establish semantic correctness')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--apply',action='store_true');ap.add_argument('--report',type=Path);args=ap.parse_args()
    runtime=store.storage();path=runtime/'library.json';before=sha(path);rows=json.loads(path.read_text());inventory={}
    for kind,name in [('commons','commons_lsf.json'),('dictasign','dictasign_lsf.json')]:
        p=ROOT/'data/source_inventory'/name;digest=sha(p)
        for e in json.loads(p.read_text())['entries']:
            sid=e['source_id'] if kind=='commons' else 'DICTASIGN_PORTAL_'+e['concept_id']
            if sid in inventory:raise ValueError('Duplicate source ID')
            inventory[sid]=(e,digest,kind)
    migrated=copy.deepcopy(rows);verified=[];unresolved=[]
    for row in migrated:
        if row.get('sign_language') is not None:continue
        try:
            evidence=identify(row,inventory,runtime)
            row.update(sign_language='fsl',language_status='source_inventory_identified_pending_linguistic_review',language_evidence=evidence)
            verified.append(row['capture_id'])
        except (ValueError,OSError,KeyError) as error:unresolved.append(dict(capture_id=row.get('capture_id'),reason=str(error)))
    # New evidence may only add these three fields; labels, reviews, rights and paths stay identical.
    for old,new in zip(rows,migrated):
        stripped={k:v for k,v in new.items() if k not in {'sign_language','language_status','language_evidence'}}
        original={k:v for k,v in old.items() if k not in {'sign_language','language_status','language_evidence'}}
        assert stripped==original
    applied=False
    if args.apply:
        if unresolved:raise ValueError('Unresolved legacy rows; inspect dry-run report before changing the library')
        with store.library_lock():
            if sha(path)!=before:raise ValueError('Library changed during audit')
            backup=runtime/'library.before-language-migration.json'
            if backup.exists() and verified:raise ValueError('Backup exists; refusing to overwrite historical evidence')
            if verified:
                backup.write_bytes(path.read_bytes());store.write_json(path,migrated);applied=True
    report=dict(status='complete',apply_requested=args.apply,applied=applied,total_rows=len(rows),
                newly_verified=len(verified),unresolved=unresolved,before_sha256=before,after_sha256=sha(path),
                non_language_fields_unchanged=True,qualified_linguistic_approval=False,network_requests=0)
    report_path=args.report or ROOT/'docs/verification'/('CAPTURE_LANGUAGE_MIGRATION.json' if applied else 'CAPTURE_LANGUAGE_MIGRATION_RECHECK.json')
    report_path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
