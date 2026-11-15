"""Expand with new upstream SUD documents; keep v001 validation/test byte-identical."""
import json,sqlite3,urllib.request,shutil
from pathlib import Path
from difflib import SequenceMatcher
from catalog import ROOT,parse_stk,norm,dump,sha,research_eligible
REV='5c485156b6eee26b2f83ca153e825bf4201fd99b'
BASE='https://raw.githubusercontent.com/surfacesyntacticud/SUD_French_Sign_Language-STKAutogramm/'+REV+'/'
def main():
    folder=ROOT/'data/text_to_gloss/raw/sud';folder.mkdir(exist_ok=True)
    tree=json.loads((ROOT/'research/sud_tree.json').read_text())
    files=[e['path'] for e in tree['tree'] if e['path'].endswith('.conllu')]+['LICENSE.txt','README.md']
    for f in files:
        dest=folder/f
        if not dest.exists():dest.write_bytes(urllib.request.urlopen(BASE+f,timeout=30).read())
    original=ROOT/'data/text_to_gloss/v001';out=ROOT/'data/text_to_gloss/v002';out.mkdir(exist_ok=True)
    data={s:[json.loads(x) for x in (original/(s+'.jsonl')).read_text().splitlines()] for s in ['train','val','test']}
    old_docs={r['document'] for rs in data.values() for r in rs};added=[];excluded=[];overlap=[]
    for f in sorted(folder.glob('*.conllu')):
        segments=parse_stk(f)
        if segments[0]['document'] in old_docs:
            overlap.append(dict(file=f.name,segments=len(segments),reason='Same document/recording as UD; not independent data'));continue
        for s in segments:
            m=s['metadata'];fr=m.get('text_fr','');gloss=' '.join(t[1] for t in s['tokens'])
            if not fr.strip() or not gloss.strip():excluded.append(dict(id=s['segment_id'],reason='Missing text'));continue
            group='sud-recording:'+s['document']
            r=dict(pair_id='sud:'+s['segment_id'],source_id='sud',language_variant='LSF_FR',french_text=fr,gloss_sequence=gloss,
                source_segment_ids=['sud:'+s['segment_id']],document=s['document'],recording=s['recording'],speaker_group=s['signer'],
                annotation_scheme='SUD-FSL-STK-FORM',license_id='CC-BY-SA-4.0',rights_status='LICENSED_RESEARCH',linguistic_review='PENDING_LOCAL_REVIEW',
                source_annotation='PUBLISHED_EXPERT_ANNOTATION',quality_flags=['FREE_TRANSLATION','PARTIAL_SYNTACTIC_ANNOTATION','NONMANUALS_NOT_IN_TARGET','SOURCE_README_DOES_NOT_DESCRIBE_ADDITIONAL_DOCUMENT'],
                source_revision=REV,split='train',split_group=group)
            # Conservative exclusion against frozen holdouts. Never alter their membership.
            def duplicate(other):
                u,v=norm(fr),norm(other['french_text'])
                return (u==v or gloss==other['gloss_sequence'] or
                    (min(len(u),len(v))>=20 and SequenceMatcher(None,u,v).ratio()>=.90))
            if any(duplicate(o) for z in ['val','test'] for o in data[z]):
                excluded.append(dict(id=s['segment_id'],reason='Near/exact overlap with frozen holdout'));continue
            if any(duplicate(o) for o in data['train']+added):
                excluded.append(dict(id=s['segment_id'],reason='Near/exact duplicate training record'));continue
            assert research_eligible(r)
            added.append(r)
    data['train']+=added
    for z in ['val','test']:shutil.copyfile(original/(z+'.jsonl'),out/(z+'.jsonl'))
    content=''.join(json.dumps(r,ensure_ascii=False,sort_keys=True)+'\n' for r in sorted(data['train'],key=lambda r:r['pair_id']))
    if (out/'train.jsonl').exists() and (out/'train.jsonl').read_text()!=content:raise ValueError('Frozen v002 change')
    (out/'train.jsonl').write_text(content)
    manifest=dict(version='v002',mode='SOURCE_ANNOTATED_RESEARCH_ONLY',parent_manifest_sha256=sha(original/'dataset_manifest.json'),
        revisions={'ud':'d1a6c6487eeddf67a14a5e61116b7cc12a0d1f6a','sud':REV},files={s+'.jsonl':sha(out/(s+'.jsonl')) for s in data},
        raw_files={p.name:sha(p) for p in folder.iterdir() if p.is_file()},counts={s:len(rs) for s,rs in data.items()},group_counts={s:len(set(r['split_group'] for r in rs)) for s,rs in data.items()},
        added_pairs=len(added),excluded=excluded,overlapping_documents_not_added=overlap,strict_approved_pairs=0,
        split_algorithm='v001 val/test frozen byte-identical; only new SUD documents added to train after duplicate/near-duplicate checks',
        limitations=['Upstream source manual gloss/free-translation annotations; new documents lack dedicated provenance detail in README','Not a formal parallel translation corpus','Not locally approved','Cross-corpus signer identities unknown'])
    dump(out/'dataset_manifest.json',manifest)
    db=sqlite3.connect(ROOT/'data/catalog/catalog.sqlite');db.execute('PRAGMA foreign_keys=ON')
    db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?)',('sud',BASE,REV,'CC-BY-SA-4.0',BASE+'LICENSE.txt','PUBLIC_TEXT','Manual source annotations/free translations; partial syntax in additional presentations; not independently locally approved'))
    for f in folder.glob('*.conllu'):
        for s in parse_stk(f):db.execute('INSERT OR REPLACE INTO annotations VALUES(?,?,?)',('sud:'+s['segment_id'],'sud',json.dumps(s,ensure_ascii=False)))
    for r in added:
        db.execute('INSERT OR REPLACE INTO pairs VALUES(?,?,?,?,?,?,?,?)',(r['pair_id'],'sud',r['french_text'],r['gloss_sequence'],r['split_group'],r['split'],r['linguistic_review'],json.dumps(r,ensure_ascii=False)))
        db.execute('INSERT OR REPLACE INTO pair_annotations VALUES(?,?,?)',(r['pair_id'],r['source_segment_ids'][0],0))
        db.execute('INSERT OR REPLACE INTO derivations VALUES(?,?,?,?,?)',('derive:'+r['pair_id'],r['pair_id'],'training/import_sud.py',json.dumps({'revision':REV,'target':'FORM; no nonmanuals; no local approval'}),sha(folder/(r['document']+'.conllu'))))
    db.commit();db.close();print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
