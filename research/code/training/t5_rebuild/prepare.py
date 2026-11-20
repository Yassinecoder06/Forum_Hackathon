"""Source-faithful LSFB annotation preparation, with preserved raw events."""
import hashlib,json,tarfile
from pathlib import Path
REPO='ppoitier/lsfb-cont'
REVISION='64b0c112f59d3710367fb736ef1808f37ff15e28'

def dump(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)

def records_from_tar(path):
    records={}
    with tarfile.open(path) as tar:
        for m in tar:
            suffix=next((s for s in ['annotations.translation.json','annotations.both_hands.json','signer.txt'] if m.name.endswith('.'+s)),None)
            if not suffix or not m.isfile():continue
            if m.size>20_000_000:raise ValueError('Unexpected annotation size')
            raw=tar.extractfile(m).read().decode('utf-8');rid=m.name[:-len(suffix)-1]
            records.setdefault(rid,{})[suffix]=raw.strip() if suffix=='signer.txt' else json.loads(raw)
    return records

def canonical_event(event):
    # The corpus author supplies lemma; never infer a lemma by stripping morphology.
    raw=event['gloss'];lemma=event.get('lemma')
    label=lemma.strip() if event.get('sign_type')=='lexical' and isinstance(lemma,str) and lemma.strip() else raw.strip()
    return (event.get('sign_type') or 'unspecified')+':'+label

def candidates(records,source='shard_000002.tar',overlap=.8):
    rows=[];seen=set()
    for rid,record in sorted(records.items()):
        for tr in record.get('annotations.translation.json',[]):
            a,b=tr['start_ms'],tr['end_ms'];text=' '.join(tr['text'].split());events=[];keys=set()
            if b<=a or not text:continue
            for event in record.get('annotations.both_hands.json',[]):
                s,e=event['start_ms'],event['end_ms'];key=(s,e,event.get('gloss'))
                if e>s and isinstance(event.get('gloss'),str) and event['gloss'].strip() and max(0,min(b,e)-max(a,s))/(e-s)>=overlap and key not in keys:
                    events.append(event);keys.add(key)
            events.sort(key=lambda e:(e['start_ms'],e['end_ms'],e['gloss']))
            if not events:continue
            # Preserve all source events and repetitions with distinct timestamps.
            target=[canonical_event(e) for e in events]
            duplicate=(text.casefold(),tuple(target))
            if duplicate in seen:continue
            seen.add(duplicate)
            rows.append({'id':f'{rid}:{a}:{b}','recording_id':rid,'signer_id':record.get('signer.txt'),'french_text':text,'units':target,'raw_glosses':[e['gloss'] for e in events],'events':events,'translation_interval_ms':[a,b],'source':REPO,'source_revision':REVISION,'source_shard':source,'language':'LSFB','license':'CC-BY-NC-SA-4.0','review_status':'technical_source_check_only','linguistic_review':'pending','target_policy':'author lemma for lexical events; original label otherwise; raw morphology retained in events','alignment':'weak interval overlap >=0.8 of each sign duration'})
    return rows

def gate_rows(rows):
    # Train-source records only. Selected for short, inspectable lexical correspondences.
    cases=[('Tu devais avoir peur !','S001'),('On ne sait jamais…','S001'),('Bien droite… c’est bien !','S001'),('Mais oui, c’est bien !','S001'),('Aie, elle n’a pas l’habitude.','S001'),('Au charbon, oui.','S001'),('Le goût est devenu beaucoup plus fort !','S001'),('Non, non, je ne suis pas d’accord !','S001'),('Exactement.','S001'),('C’est « oraliser ».','S001'),('C’est génial !','S036'),('Normal, tu es sourd.','S036'),('Attends !','S036'),('Oui, je sais.','S036'),('Je comprends.','S062'),('Les dents ?','S065')]
    normalize=lambda s:' '.join(s.split())
    selected=[]
    for text,signer in cases:
        matches=[r for r in rows if normalize(r['french_text'])==normalize(text) and r['signer_id']==signer]
        if len(matches)!=1:raise ValueError(f'Expected one source match: {text}: {len(matches)}')
        row=matches[0]
        if not all(e['sign_type']=='lexical' for e in row['events']):raise ValueError('Nonlexical event in short learning gate')
        selected.append(row)
    return selected

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('archive',type=Path);p.add_argument('output',type=Path);a=p.parse_args()
    records=records_from_tar(a.archive);rows=candidates(records,source=a.archive.name);gate=gate_rows(rows)
    vocabulary=sorted({u for r in gate for u in r['units']})
    result={'kind':'train-only memorization diagnostic; not a translation benchmark','source_revision':REVISION,'source_sha256':hashlib.file_digest(a.archive.open('rb'),'sha256').hexdigest(),'target_policy':'author-provided lexical lemma, with original raw labels and variations preserved','language':'LSFB','rows':gate,'vocabulary':vocabulary,'linguistic_approval':False,'generalization_claim':False}
    dump(a.output,result);dump(a.output.with_name('all_source_candidates.json'),rows)
    print(json.dumps({'source_candidates':len(rows),'gate_pairs':len(gate),'gate_units':len(vocabulary),'output':str(a.output)}))
