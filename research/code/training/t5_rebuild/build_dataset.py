"""Freeze an atomic-unit LSFB dataset without mining held-out target labels."""
import argparse,gzip,hashlib,json,unicodedata,collections
from pathlib import Path
from prepare import candidates,canonical_event,dump

def norm(text):return ' '.join(unicodedata.normalize('NFKC',text).casefold().split())
def build(raw,isol,output):
    source=json.loads(isol.read_text());vocabulary={c['unit'] for c in source['classes']};origins={u:{'ISOL_TRAIN'} for u in vocabulary};train=[];validation=[];manifest=[]
    for number in [2,3,4,5,6,7,1]:
        path=raw/f'annotations_{number:06d}.json.gz'
        if not path.is_file():raise ValueError('Missing source annotations: '+str(path))
        data=json.load(gzip.open(path,'rt',encoding='utf-8'));assert data['source_shard']==number and data['official_role']==('validation' if number==1 else 'train')
        manifest.append({k:data[k] for k in ['source','revision','source_shard','official_role','tar_sha256']})
        rows=candidates(data['records'],source=f'shard_{number:06d}.tar')
        if number!=1:
            for record in data['records'].values():
                for event in record.get('annotations.both_hands.json',[]):
                    if isinstance(event.get('gloss'),str) and event['gloss'].strip():
                        u=canonical_event(event);vocabulary.add(u);origins.setdefault(u,set()).add('CONT_TRAIN_ANNOTATION')
            train.extend(rows)
        else:validation.extend(rows)
    # Retain original annotations; no gesture/pointing events are silently removed.
    validation_inputs={norm(r['french_text']) for r in validation};seen=set();clean=[];removed=[]
    for row in train:
        key=(norm(row['french_text']),tuple(row['units']))
        if norm(row['french_text']) in validation_inputs:removed.append({'id':row['id'],'reason':'French text appears in validation'});continue
        if key in seen:removed.append({'id':row['id'],'reason':'duplicate French/target pair in training'});continue
        seen.add(key);clean.append(row)
    train=clean
    train_recordings={r['recording_id'] for r in train};val_recordings={r['recording_id'] for r in validation}
    if train_recordings&val_recordings:raise ValueError('Official partitions overlap recording IDs')
    output.mkdir(parents=True,exist_ok=True)
    for name,rows in [('train',train),('val',validation)]:
        (output/(name+'.jsonl')).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    units=sorted(vocabulary);dump(output/'vocabulary.json',{'units':units,'origins':{u:sorted(origins[u]) for u in units},'source_rule':'ISOL official train inventory + all CONT official training annotation units only; no validation/test labels added','isol_source_revision':source['revision'],'isol_train_classes_observed':len(source['classes'])})
    observed={u for r in train for u in r['units']};val_tokens=[u for r in validation for u in r['units']]
    report={'language':'LSFB','target_policy':'author lexical lemma, original nonlexical annotation label; raw events/morphology preserved','train_pairs':len(train),'val_pairs':len(validation),'total_atomic_units':len(units),'val_reference_units':len(val_tokens),'val_units_outside_inventory':sum(u not in vocabulary for u in val_tokens),'val_units_not_in_paired_train':sum(u not in observed for u in val_tokens),'paired_train_types':len(observed),'val_inventory_coverage':sum(u in vocabulary for u in val_tokens)/max(1,len(val_tokens)),'train_recordings':len(train_recordings),'val_recordings':len(val_recordings),'overlapping_signers':sorted({r['signer_id'] for r in train}&{r['signer_id'] for r in validation}),'signer_disjoint':not bool({r['signer_id'] for r in train}&{r['signer_id'] for r in validation}),'removed_train_records':removed,'sources':manifest,'test_shard_0_opened':False,'linguistic_approval':False,'scope':'weakly source-aligned canonical-unit research, including nonlexical annotations; not a certified translator','files':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in output.glob('*') if p.name in ['train.jsonl','val.jsonl','vocabulary.json']}}
    dump(output/'manifest.json',report);return report
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('raw',type=Path);p.add_argument('isol',type=Path);p.add_argument('output',type=Path);a=p.parse_args();print(json.dumps(build(a.raw,a.isol,a.output),ensure_ascii=False,indent=2))
