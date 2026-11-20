"""Freeze a real runtime vocabulary and authored SOFTWARE tests, never LSF data."""
import json,hashlib
from pathlib import Path
from catalog import ROOT,dump,sha
MAIN=Path('/home/jmal/Projects/Active/Forum_Hackathon')
def main():
    out=ROOT/'research/concept_benchmark';out.mkdir(exist_ok=True)
    library=MAIN/'.runtime/library.json';curated=MAIN/'data/glossary.json';motions=MAIN/'data/motions_manifest.json'
    raw=json.loads(library.read_text());entries=[];provenance=[]
    for r in raw:
        g,m=r['gloss'],r['motion']
        if m.get('synthetic') or m.get('rights_status')!='cleared':continue
        entries.append({'id':g['gloss_id'],'concept':g['concept_fr']})
        provenance.append(dict(id=g['gloss_id'],source_id=r.get('source_id'),motion_id=m['motion_id'],review_status=g['review_status'],source_url=r.get('attribution',{}).get('source_url'),license=m.get('license')))
    mg={m['motion_id']:m for m in json.loads(motions.read_text())}
    for g in json.loads(curated.read_text()):
        m=mg.get(g['motion_id'],{})
        if g['concept_fr']=='bonjour' and m.get('synthetic') is False and m.get('rights_status')=='cleared':
            entries.append({'id':g['gloss_id'],'concept':g['concept_fr']});provenance.append(dict(id=g['gloss_id'],source_id=g['source_id'],motion_id=g['motion_id'],review_status=g['review_status'],license=m['license']))
    entries.sort(key=lambda r:r['id']);assert len(entries)==440 and len({r['id'] for r in entries})==440
    by_concept={r['concept']:r['id'] for r in entries}
    # Authored French software expectations. These are NOT signed utterances,
    # LSF ground truth, STK examples, or supervised model-training material.
    specs=[
      ('exact_bonjour','exact','Bonjour.',['bonjour'],[],[]),
      ('exact_merci','exact','Merci.',['merci'],[],[]),
      ('exact_chien','exact','Chien.',['chien'],[],[]),
      ('exact_pair','multi_exact','Du pain et du café.',['pain','café'],[],[]),
      ('plural_chiens','lemma','Chiens.',['chien'],[],[]),
      ('plural_enfants','lemma','Enfants.',['enfant'],[],[]),
      ('feminine_state','lemma','Fatiguée.',['fatigué'],[],[]),
      ('syn_docteur','synonym','Docteur.',['médecin'],[],[]),
      ('syn_bouquin','synonym','Bouquin.',['livre'],[],[]),
      ('syn_epuise','synonym','Épuisé.',['fatigué'],[],[]),
      ('syn_bosser','synonym','Bosser.',['travailler'],[],[]),
      ('syn_rigoler','synonym','Rigoler.',['rire'],[],[]),
      ('syn_heureux','synonym','Heureux.',['content'],[],[]),
      ('multi_synonym','multi_semantic','Un docteur et un bouquin.',['médecin','livre'],[],[]),
      ('multi_states','multi_semantic','Épuisé et triste.',['fatigué','triste'],[],[]),
      ('multi_action','multi_semantic',"Le chat boit de l'eau.",['chat','boire','eau'],[],[]),
      ('negation_ne_pas','negation','Ne bois pas de café.',['boire','café'],['NEGATION_UNSUPPORTED'],[]),
      ('negation_jamais','negation','Jamais de café.',['café'],['NEGATION_UNSUPPORTED'],[]),
      ('question_ou','question','Où est le médecin ?',['médecin'],['QUESTION_UNSUPPORTED'],[]),
      ('number_three','number','Trois chats.',['chat'],['NUMBER_UNSUPPORTED'],[]),
      ('named_person_place','named_entity','Marie adore Paris.',['adorer'],['NAMED_ENTITY_UNSUPPORTED'],[]),
      ('unknown_only','unknown','Photosynthèse.',[],[],['photosynthèse']),
      ('mixed_unknown','unknown',"Du café et de la cryptomonnaie.",['café'],[],['cryptomonnaie']),
      ('unknown_named','named_entity','Nora.',[],['NAMED_ENTITY_UNSUPPORTED'],[]),
    ]
    cases=[dict(case_id=i,category=c,text=t,expected_concepts=concepts,expected_concept_ids=[by_concept[x] for x in concepts],required_warning_codes=flags,required_unsupported_terms=unknown) for i,c,t,concepts,flags,unknown in specs]
    for name,value in [('vocabulary.json',entries),('benchmark.json',dict(kind='AUTHORED_SOFTWARE_EXPECTATIONS_NOT_LSF_GROUND_TRUTH_NOT_TRAINING',cases=cases)),('vocabulary_provenance.json',provenance)]:
        p=out/name
        if p.exists():raise ValueError('Frozen artifact already exists: '+name)
        dump(p,value)
    baseline_source=MAIN/'backend/app/services/text_lookup.py'
    # Freeze the exact existing lead implementation without importing/rebasing main.
    (ROOT/'research/code/training/literal_lookup_snapshot.py').write_bytes(baseline_source.read_bytes())
    dump(out/'manifest.json',dict(vocabulary_count=440,case_count=len(cases),purpose='French semantic lexical concept planning software check ONLY; not LSF translation evaluation',data_policy='No LSF training pairs authored or altered',sources={str(p):sha(p) for p in [library,curated,motions,baseline_source]},files={p.name:sha(p) for p in out.iterdir() if p.is_file()},baseline='Lead .venv/bin/python; exact snapshot of existing lookup_text, GlossEntry objects, real fr_core_news_sm; no alias additions',evaluation_policy='One model/prompt/settings lock before inference; no prompt changes or retries of benchmark failures'))
    print('Frozen 440 real vocabulary entries and 24 authored software tests before model inference.')
if __name__=='__main__':main()
