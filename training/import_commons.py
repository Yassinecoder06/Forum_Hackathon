"""Import the lead's already-audited Commons inventory. Makes NO network requests."""
import argparse,json,sqlite3
from pathlib import Path
from catalog import ROOT,dump,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--inventory',type=Path,required=True);a=p.parse_args()
    inventory=json.loads(a.inventory.read_text());rows=inventory['entries']
    if not all(r['language_variant']=='LSF_FR' and r['kind']=='isolated_sign_video' for r in rows):raise ValueError('Unexpected source variety/modality')
    target=ROOT/'data/catalog/commons_lexical_metadata.json';target.write_bytes(a.inventory.read_bytes())
    db=sqlite3.connect(ROOT/'data/catalog/catalog.sqlite');db.execute('PRAGMA foreign_keys=ON')
    db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?)',('commons-laura','https://commons.wikimedia.org/wiki/Category:Vidéos_LSF_par_Laura_Jauvert','inventory sha256:'+sha(a.inventory),'PER_FILE_CC_BY','https://creativecommons.org/licenses/by/3.0/','PUBLIC_LICENSED_VIDEO_METADATA','Lead-provided per-file audit; no network requests; lexical clips only'))
    for r in rows:
        db.execute('INSERT OR REPLACE INTO lexical_entries VALUES(?,?,?,?,?,?,?)',(r['source_id'],'commons-laura',r['concept_fr'],None,'LEXICAL_CONCEPT_NOT_SENTENCE_PAIR',r['source_url'],'PENDING_LOCAL_REVIEW'))
        db.execute('INSERT OR REPLACE INTO assets VALUES(?,?,?,?,?,?,?)',('video:'+r['source_id'],'commons-laura',r['source_id'],'VIDEO_REFERENCE',r['download_url'],r['license'],'PENDING_MOTION_REVIEW'))
    db.commit();db.close();print(json.dumps(dict(lexical_entries=len(rows),eligible_video_references=sum(r['eligible_for_motion'] for r in rows),sentence_training_pairs=0,inventory_sha256=sha(a.inventory))))
if __name__=='__main__':main()
