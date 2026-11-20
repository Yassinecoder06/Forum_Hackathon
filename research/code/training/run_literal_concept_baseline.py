"""Run the EXISTING lead lookup snapshot with real GlossEntry records in lead Python."""
import json,sys,time
from pathlib import Path
MAIN=Path('/home/jmal/Projects/Active/Forum_Hackathon')
sys.path.insert(0,str(MAIN/'backend'))
from app.schemas import GlossEntry
import literal_lookup_snapshot as lookup
from concept_planner import linguistic_guards,normalize

def main():
    payload=json.load(sys.stdin);vocab=payload['vocabulary'];wanted={r['id']:r['concept'] for r in vocab}
    all_gloss=[r['gloss'] for r in json.loads((MAIN/'.runtime/library.json').read_text())]+json.loads((MAIN/'data/glossary.json').read_text())
    records={g['gloss_id']:g for g in all_gloss}
    entries=[GlossEntry(**records[i]) for i in wanted]
    if any(e.concept_fr!=wanted[e.gloss_id] for e in entries):raise ValueError('Lead vocabulary changed')
    start=time.monotonic();mode=lookup.mode();load_seconds=time.monotonic()-start
    if mode!='spacy-fr-lemmatizer':raise ValueError('Real French lemmatizer is required for this baseline')
    results=[]
    for r in payload['cases']:
        start=time.monotonic();out=lookup.lookup_text(r['text'],entries);latency=time.monotonic()-start
        ids=[g.gloss_id for g in out['gloss_sequence']];guard=linguistic_guards(r['text'])
        filtered=list(dict.fromkeys(i for i in ids if normalize(wanted[i]) not in guard['named']))
        results.append(dict(case_id=r['case_id'],raw=dict(concept_ids=ids,unsupported=out['unknowns'],warnings=[],latency_seconds=latency),
            shared_guards=dict(concept_ids=filtered,unsupported=guard['unsupported']+out['unknowns'],warnings=guard['warnings'],latency_seconds=latency),tokens=out['tokens'],lemmas=out['lemmas']))
    import spacy
    print(json.dumps(dict(results=results,mode=mode,spacy_version=spacy.__version__,fr_model_version=lookup.model().meta['version'],model_load_seconds=load_seconds,python=sys.executable),ensure_ascii=False))
if __name__=='__main__':main()
