"""Atomic-unit scoring keeps unknown references and invalid predictions distinct."""
import numpy as np

def distance(a,b):
    previous=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        row=[i]
        for j,y in enumerate(b,1):row.append(min(row[-1]+1,previous[j]+1,previous[j-1]+(x!=y)))
        previous=row
    return previous[-1]

def unpack(ids,reverse,eos=1,pad=0,reference=False):
    ids=[int(i) for i in ids if i!=-100]
    if ids and ids[0]==pad:ids=ids[1:]
    result=[];invalid=0;stopped=False
    for idx in ids:
        if idx==eos:stopped=True;break
        if idx in reverse:result.append(reverse[idx])
        else:result.append('<UNSEEN_REFERENCE>' if reference else f'<INVALID:{idx}>');invalid+=1
    return result,invalid,stopped

def evaluate_arrays(predictions,labels,reverse,eos=1,pad=0):
    predictions=np.where(predictions<0,pad,predictions)
    pred=[unpack(x,reverse,eos,pad) for x in predictions];refs=[unpack(x,reverse,eos,pad,True) for x in labels]
    n=len(refs);total=sum(len(x[0]) for x in refs)
    return {'wer':sum(distance(r[0],p[0]) for r,p in zip(refs,pred))/max(1,total),
        'exact_match':sum(r[0]==p[0] and p[2] and not p[1] for r,p in zip(refs,pred))/max(1,n),
        'empty_rate':sum(not p[0] for p in pred)/max(1,n),'invalid_sequence_rate':sum(bool(p[1]) for p in pred)/max(1,n),
        'eos_rate':sum(p[2] for p in pred)/max(1,n),'reference_oov_rate':sum(r[1] for r in refs)/max(1,total),
        'mean_predicted_units':sum(len(p[0]) for p in pred)/max(1,n),'mean_reference_units':total/max(1,n)}
