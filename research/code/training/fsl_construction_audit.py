"""Train-only, recording-held-out construction/retrieval diagnostic for FSL UD."""
import collections
import hashlib
import json
import math
import re
import unicodedata
from pathlib import Path

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
TRAIN=ROOT/'data/parallel/fsl_ud_feasibility_train.jsonl'
VAL=ROOT/'data/parallel/fsl_ud_feasibility_val.jsonl'
OUT=ROOT/'research/private/fsl-construction-audit-v1'

def read(p):return [json.loads(x) for x in p.read_text().splitlines()]
def norm(s):return ' '.join(unicodedata.normalize('NFKC',s).casefold().split())
def seqkey(xs):return tuple(xs)
def tokens(s):return re.findall(r"[\w\[\]-]+|[^\w\s]",norm(s))
def ngrams(xs,n):return collections.Counter(tuple(xs[i:i+n]) for i in range(len(xs)-n+1))
def overlap(a,b):
    na,nb=len(a),len(b)
    return sum((ngrams(a,n)&ngrams(b,n)).total() for n in (1,2,3))/max(1,sum(max(0,len(a)-n+1) for n in (1,2,3)))
def distance(a,b):
    p=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        c=[i]+[0]*len(b)
        for j,y in enumerate(b,1):c[j]=min(c[j-1]+1,p[j]+1,p[j-1]+(x!=y))
        p=c
    return p[-1]
def score(rows,preds):
    refs=[r['gloss_sequence'] for r in rows]
    edits=sum(distance(a,b) for a,b in zip(refs,preds));den=sum(map(len,refs))
    return dict(examples=len(rows),exact=sum(a==b for a,b in zip(refs,preds)),wer=edits/max(1,den),
                token_accuracy=1-edits/max(1,den),
                one_sign=dict(n=sum(len(x)==1 for x in refs),exact=sum(a==b for a,b in zip(refs,preds) if len(a)==1)),
                multi_sign=dict(n=sum(len(x)>1 for x in refs),exact=sum(a==b for a,b in zip(refs,preds) if len(a)>1)))

def main():
    train,val=read(TRAIN),read(VAL);OUT.mkdir(parents=True,exist_ok=False)
    assert all(r['sign_language']=='fsl' for r in train+val)
    assert not {r['source_recording_id'] for r in train}&{r['source_recording_id'] for r in val}
    vocab=collections.Counter(t for r in train for t in r['gloss_sequence'])
    train_sequences=collections.Counter(seqkey(r['gloss_sequence']) for r in train)
    train_pairs=collections.Counter(pair for r in train for pair in zip(r['gloss_sequence'],r['gloss_sequence'][1:]))
    targets=[]; diagnostics=[]
    for q in val:
        qt=tokens(q['french_text']);rank=[]
        for r in train:
            rt=tokens(r['french_text'])
            sim=overlap(qt,rt)
            rank.append((sim,r))
        best=max(rank,key=lambda x:x[0])[1]
        pred=best['gloss_sequence'];targets.append(pred)
        ref=q['gloss_sequence'];label_set=set(ref)
        unknown=[x for x in ref if x not in vocab]
        pair_list=list(zip(ref,ref[1:]));unseen_pairs=[p for p in pair_list if p not in train_pairs]
        diagnostics.append(dict(id=q['id'],french_text=q['french_text'],reference=ref,prediction=pred,
            retrieved_train_id=best['id'],retrieval_similarity=max(rank,key=lambda x:x[0])[0],
            identical_reference_sequence_in_train=train_sequences[seqkey(ref)],
            reference_labels_seen=sum(x in vocab for x in ref),reference_label_count=len(ref),
            unseen_labels=unknown,seen_unigrams=len(label_set&set(vocab)),
            unseen_adjacent_pairs=unseen_pairs,all_reference_labels_seen=not unknown,
            all_adjacent_pairs_seen=not unseen_pairs,annotation_scheme=q['annotation_convention'],
            gold_translation_pair=q['gold_translation_pair']))
    metrics=score(val,targets)
    metrics.update(train_examples=len(train),train_label_vocabulary=len(vocab),validation_labels=sum(len(r['gloss_sequence']) for r in val),
       validation_oov_label_occurrences=sum(not x['all_reference_labels_seen'] for x in diagnostics),
       fully_seen_label_sequences=sum(x['all_reference_labels_seen'] for x in diagnostics),
       fully_seen_adjacent_pair_sequences=sum(x['all_adjacent_pairs_seen'] for x in diagnostics),
       exact_target_sequences_repeated_in_train=sum(x['identical_reference_sequence_in_train']>0 for x in diagnostics),
       train_recordings=sorted({r['source_recording_id'] for r in train}),val_recordings=sorted({r['source_recording_id'] for r in val}),
       train_sha256=hashlib.sha256(TRAIN.read_bytes()).hexdigest(),val_sha256=hashlib.sha256(VAL.read_bytes()).hexdigest(),
       scope='France-LSF UD published annotation label prediction from free French translations; not validated semantic translation.',
       test_split_opened=False,qualified_review=False)
    (OUT/'retrieval_predictions.json').write_text(json.dumps(diagnostics,ensure_ascii=False,indent=2)+'\n')
    (OUT/'RESULT.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(metrics,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
