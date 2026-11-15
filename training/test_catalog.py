import json,sqlite3
from itertools import combinations
from difflib import SequenceMatcher
import pytest
from catalog import ROOT,parse_stk,norm,sha,research_eligible,approved_eligible

DATA=ROOT/'data/text_to_gloss/v001'
def records():return {s:[json.loads(x) for x in (DATA/(s+'.jsonl')).read_text().splitlines()] for s in ['train','val','test']}
def test_frozen_manifests():
    m=json.loads((DATA/'dataset_manifest.json').read_text())
    for f,h in m['files'].items():assert sha(DATA/f)==h
    assert sha(ROOT/'data/text_to_gloss/raw/stk/fsl_stkautogramm-ud-test.conllu')==m['source_sha256']
def test_no_cross_split_leakage():
    d=records()
    for a,b in combinations(d,2):
        for key in ['pair_id','split_group','recording','speaker_group','source_segment_ids']:
            def vals(rs):
                return {v for r in rs for v in (r[key] if isinstance(r[key],list) else [r[key]])}
            assert not vals(d[a])&vals(d[b]),key
        for x in d[a]:
            for y in d[b]:
                assert norm(x['french_text'])!=norm(y['french_text'])
                assert x['gloss_sequence']!=y['gloss_sequence']
                u,v=norm(x['french_text']),norm(y['french_text'])
                assert min(len(u),len(v))<20 or SequenceMatcher(None,u,v).ratio()<.90

def test_raw_annotation_and_merges_preserved():
    raw={s['segment_id']:s for s in parse_stk()}; seen=[]
    for rs in records().values():
        for r in rs:
            ss=[raw[s] for s in r['source_segment_ids']];seen+=r['source_segment_ids']
            assert r['gloss_sequence']==' '.join(t[1] for s in ss for t in s['tokens'])
            assert all(norm(s['metadata']['text_fr'])==norm(r['french_text']) for s in ss)
    assert len(seen)==len(set(seen))==108

def test_strict_gate_is_not_research_gate():
    r=records()['train'][0];assert research_eligible(r);assert not approved_eligible(r)
    for key,value in [('linguistic_review','APPROVED'),('license_id','UNKNOWN'),('language_variant','LSF_CH'),('source_annotation','SYNTHETIC')]:
        assert not research_eligible({**r,key:value})
    assert not approved_eligible({**r,'rights_status':'TRAIN_APPROVED'})

def test_catalog_integrity_and_lineage():
    c=sqlite3.connect(ROOT/'data/catalog/catalog.sqlite')
    assert c.execute('PRAGMA integrity_check').fetchone()==('ok',)
    assert c.execute('PRAGMA foreign_key_check').fetchall()==[]
    assert c.execute('select count(*) from annotations where source_id="stk"').fetchone()[0]==108
    assert c.execute('select count(*) from pair_annotations where pair_id like "stk:%"').fetchone()[0]==108
    assert c.execute('select count(*) from derivations where pair_id like "stk:%"').fetchone()[0]==106
    assert c.execute("select count(*) from pairs where linguistic_review='APPROVED'").fetchone()[0]==0


def test_expansion_preserves_frozen_holdouts_and_no_leakage():
    expanded=ROOT/'data/text_to_gloss/v002'
    if not expanded.exists():pytest.skip('Expansion not built')
    m=json.loads((expanded/'dataset_manifest.json').read_text())
    for s in ['val','test']:assert sha(expanded/(s+'.jsonl'))==sha(DATA/(s+'.jsonl'))
    for f,h in m['files'].items():assert sha(expanded/f)==h
    train=[json.loads(x) for x in (expanded/'train.jsonl').read_text().splitlines()]
    holdout=records()['val']+records()['test']
    for a in train:
        assert research_eligible(a) and not approved_eligible(a)
        for b in holdout:
            assert a['recording']!=b['recording'] and a['speaker_group']!=b['speaker_group']
            assert a['gloss_sequence']!=b['gloss_sequence']
            u,v=norm(a['french_text']),norm(b['french_text'])
            assert u!=v and (min(len(u),len(v))<20 or SequenceMatcher(None,u,v).ratio()<.90)

def test_lexical_inventory_never_becomes_pairs():
    c=sqlite3.connect(ROOT/'data/catalog/catalog.sqlite')
    assert c.execute("select count(*) from lexical_entries where source_id='commons-laura'").fetchone()[0]==447
    assert c.execute("select count(*) from pairs where source_id='commons-laura'").fetchone()[0]==0


def test_export_authenticity_not_just_relabelled_flags():
    from experiment import validate_exports
    d=records();validate_exports(DATA,d,'source-annotated-research')
    r=d['train'][0];d['train'][0]={**r,'french_text':'Invented input'}
    with pytest.raises(ValueError,match='differs from source-derived catalog'):
        validate_exports(DATA,d,'source-annotated-research')
