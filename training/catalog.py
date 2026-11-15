"""Source-specific STK import, provenance catalog and frozen research exports."""
import argparse, hashlib, json, re, sqlite3, unicodedata
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
REV='d1a6c6487eeddf67a14a5e61116b7cc12a0d1f6a'
RAW=ROOT/'data/text_to_gloss/raw/stk/fsl_stkautogramm-ud-test.conllu'
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def norm(s): return ' '.join(re.findall(r'\w+',unicodedata.normalize('NFKC',s).casefold()))
def dump(path,obj): Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
def parse_stk(path=RAW):
    rows=[]
    for block in Path(path).read_text().strip().split('\n\n'):
        meta={}; tokens=[]
        for line in block.splitlines():
            if line.startswith('# ') and ' = ' in line:
                k,v=line[2:].split(' = ',1); meta[k]=v
            elif line and not line.startswith('#'):
                cols=line.split('\t')
                if cols[0].isdigit():
                    if len(cols)!=10: raise ValueError('Invalid CoNLL-U')
                    tokens.append(cols)
        sid=meta['sent_id']; doc=sid.split('__')[0]
        # Two LS-Colin extracts share CUC022_low.mp4: one recording, one group.
        recording=meta.get('video_url') if doc.startswith('LS-Colin') else meta.get('source_file',doc)
        rows.append(dict(segment_id=sid,document=doc,recording=recording,
                         signer=doc+':'+meta.get('speaker_id','UNKNOWN'),metadata=meta,tokens=tokens))
    return rows

def research_eligible(r):
    return (r['language_variant']=='LSF_FR' and r['rights_status']=='LICENSED_RESEARCH'
        and r['linguistic_review']=='PENDING_LOCAL_REVIEW'
        and r['source_annotation']=='PUBLISHED_EXPERT_ANNOTATION'
        and r['license_id']=='CC-BY-SA-4.0' and bool(r['french_text'].strip())
        and bool(r['gloss_sequence'].strip()) and bool(r['source_segment_ids']))
def approved_eligible(r):
    # Original brief's production gate remains strict and independent.
    return (r['language_variant']=='LSF_FR' and r['rights_status']=='TRAIN_APPROVED'
        and r['linguistic_review']=='APPROVED' and bool(r['french_text'].strip())
        and bool(r['gloss_sequence'].strip()))

def build():
    raw=parse_stk(); pairs=[]
    for s in raw:
        m=s['metadata']; fr=m.get('text_fr',''); gloss=' '.join(t[1] for t in s['tokens'])
        if not fr.strip() or not gloss.strip(): continue
        # Source sometimes repeats one free translation over consecutive syntactic units.
        # Merge only adjacent units in the same document and signer; never synthesize text.
        if pairs and pairs[-1]['document']==s['document'] and pairs[-1]['speaker_group']==s['signer'] and norm(pairs[-1]['french_text'])==norm(fr):
            p=pairs[-1]; p['source_segment_ids'].append(s['segment_id']); p['gloss_sequence']+=' '+gloss
            p['quality_flags'].append('MERGED_ADJACENT_IDENTICAL_FREE_TRANSLATION')
        else:
            pairs.append(dict(pair_id='stk:'+s['segment_id'],source_id='stk',language_variant='LSF_FR',
                french_text=fr,gloss_sequence=gloss,source_segment_ids=[s['segment_id']],
                document=s['document'],recording=s['recording'],speaker_group=s['signer'],
                annotation_scheme='UD-FSL-STK-FORM',license_id='CC-BY-SA-4.0',
                rights_status='LICENSED_RESEARCH',linguistic_review='PENDING_LOCAL_REVIEW',
                source_annotation='PUBLISHED_EXPERT_ANNOTATION',
                quality_flags=['FREE_TRANSLATION','UD_PARALLEL_NO','NONMANUALS_NOT_IN_TARGET'],
                source_revision=REV))
    parent=list(range(len(pairs)))
    def find(i):
        while parent[i]!=i: parent[i]=parent[parent[i]]; i=parent[i]
        return i
    def union(i,j): parent[find(j)]=find(i)
    edges=[]
    for i,a in enumerate(pairs):
        for j in range(i):
            b=pairs[j]; fa,fb=norm(a['french_text']),norm(b['french_text'])
            same_group=a['recording']==b['recording'] or a['speaker_group']==b['speaker_group']
            exact=fa==fb or a['gloss_sequence']==b['gloss_sequence']
            near=min(len(fa),len(fb))>=20 and SequenceMatcher(None,fa,fb).ratio()>=.90
            if same_group or exact or near:
                union(i,j)
                if exact or near: edges.append([a['pair_id'],b['pair_id'],'exact' if exact else 'near'])
    comps={}
    for i,p in enumerate(pairs): comps.setdefault(find(i),[]).append(p)
    # Fixed assignment based on provenance only, before any model or metric inspection.
    # Largest component training, next validation, remaining test. No random row split.
    groups=sorted(comps.values(),key=lambda g:(-len(g),min(p['pair_id'] for p in g)))
    if len(groups)<3: raise ValueError('Insufficient independent recording components')
    for i,g in enumerate(groups):
        split='train' if i==0 else 'val' if i==1 else 'test'
        gid='component:'+hashlib.sha256('|'.join(sorted(p['pair_id'] for p in g)).encode()).hexdigest()[:16]
        for p in g: p.update(split=split,split_group=gid)
    out=ROOT/'data/text_to_gloss/v001';out.mkdir(exist_ok=True)
    for split in ['train','val','test']:
        path=out/(split+'.jsonl'); content=''.join(json.dumps(p,ensure_ascii=False,sort_keys=True)+'\n' for p in sorted(pairs,key=lambda p:p['pair_id']) if p['split']==split)
        if path.exists() and path.read_text()!=content: raise ValueError('Frozen export changed; create a new version')
        path.write_text(content)
    manifest=dict(version='v001',experiment='SOURCE_ANNOTATED_RESEARCH_ONLY',source_revision=REV,
        source_sha256=sha(RAW),raw_segments=len(raw),derived_pairs=len(pairs),strict_approved_pairs=0,
        counts=dict(Counter(p['split'] for p in pairs)),groups=[dict(split=g[0]['split'],count=len(g),recordings=sorted(set(p['recording'] for p in g)),documents=sorted(set(p['document'] for p in g))) for g in groups],
        split_algorithm='recording/signer/exact-French/exact-gloss/near-French union; size descending; first train, second val, rest test',
        near_duplicate_threshold=.90,duplicate_edges=edges,seed=42,
        files={s+'.jsonl':sha(out/(s+'.jsonl')) for s in ['train','val','test']},
        limitations=['Single source; source-disjoint evaluation impossible','Signer identities across unrelated corpora unknown','Free translations, not formal parallel benchmark','No local linguistic approval'])
    dump(out/'dataset_manifest.json',manifest)
    db=sqlite3.connect(ROOT/'data/catalog/catalog.sqlite');db.execute('PRAGMA foreign_keys=ON')
    db.executescript('''
    CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY,url TEXT,revision TEXT,license_id TEXT,terms_url TEXT,access TEXT,notes TEXT);
    CREATE TABLE IF NOT EXISTS annotations(segment_id TEXT PRIMARY KEY,source_id TEXT REFERENCES sources,raw_json TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS pairs(pair_id TEXT PRIMARY KEY,source_id TEXT REFERENCES sources,french_text TEXT,raw_gloss_sequence TEXT,split_group TEXT,split TEXT,linguistic_review TEXT CHECK(linguistic_review IN ('PENDING_LOCAL_REVIEW','APPROVED')),record_json TEXT NOT NULL);
    CREATE TABLE IF NOT EXISTS pair_annotations(pair_id TEXT REFERENCES pairs,segment_id TEXT REFERENCES annotations,ordinal INTEGER,PRIMARY KEY(pair_id,segment_id));
    CREATE TABLE IF NOT EXISTS derivations(derivation_id TEXT PRIMARY KEY,pair_id TEXT REFERENCES pairs,tool TEXT,parameters_json TEXT,input_sha256 TEXT);
    CREATE TABLE IF NOT EXISTS reviews(review_id TEXT PRIMARY KEY,pair_id TEXT REFERENCES pairs,reviewer_role TEXT,decision TEXT,timestamp TEXT,notes TEXT);
    CREATE TABLE IF NOT EXISTS lexical_entries(entry_id TEXT PRIMARY KEY,source_id TEXT REFERENCES sources,label_fr TEXT,raw_gloss TEXT,representation TEXT,source_url TEXT,review_status TEXT);
    CREATE TABLE IF NOT EXISTS assets(asset_id TEXT PRIMARY KEY,source_id TEXT REFERENCES sources,entry_id TEXT REFERENCES lexical_entries,kind TEXT,url TEXT,license_id TEXT,review_status TEXT);
    CREATE TABLE IF NOT EXISTS model_runs(run_id TEXT PRIMARY KEY,model_base TEXT,model_revision TEXT,dataset_sha256 TEXT,hyperparameters_json TEXT,metrics_json TEXT,checkpoint_uri TEXT);
    ''')
    db.execute('INSERT OR REPLACE INTO sources VALUES(?,?,?,?,?,?,?)',('stk','https://github.com/UniversalDependencies/UD_French_Sign_Language-STKAutogramm',REV,'CC-BY-SA-4.0','https://creativecommons.org/licenses/by-sa/4.0/','PUBLIC_TEXT','Guillaume et al.; free translations; Parallel:no; video rights separate'))
    for s in raw: db.execute('INSERT OR REPLACE INTO annotations VALUES(?,?,?)',(s['segment_id'],'stk',json.dumps(s,ensure_ascii=False)))
    for p in pairs:
        assert research_eligible(p) and not approved_eligible(p)
        db.execute('INSERT OR REPLACE INTO pairs VALUES(?,?,?,?,?,?,?,?)',(p['pair_id'],'stk',p['french_text'],p['gloss_sequence'],p['split_group'],p['split'],p['linguistic_review'],json.dumps(p,ensure_ascii=False)))
        for i,sid in enumerate(p['source_segment_ids']):db.execute('INSERT OR REPLACE INTO pair_annotations VALUES(?,?,?)',(p['pair_id'],sid,i))
        db.execute('INSERT OR REPLACE INTO derivations VALUES(?,?,?,?,?)',('derive:'+p['pair_id'],p['pair_id'],'training/catalog.py',json.dumps({'target':'integer token FORM in source order; adjacent identical French merge','revision':REV}),sha(RAW)))
    db.commit();db.close()
    dump(ROOT/'research/inspected_examples.json',[dict(raw=raw[i],derived=next(p for p in pairs if raw[i]['segment_id'] in p['source_segment_ids'])) for i in [0,1,2,3,4]])
    print(json.dumps(manifest,ensure_ascii=False,indent=2))
if __name__=='__main__': build()
