"""Technically filtered, unreviewed LSFB memorization gate; no test data opened."""
import json,random,time,collections,hashlib,argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,set_seed
from peft import get_peft_model,LoraConfig,TaskType,PeftModel
from .prepare import lexical_operation
from .scoring import distance,score

ROOT=Path('.runtime/lsfb-recovery-v2')

def save(name,value):
    path=ROOT/name;temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(path)

def prepare(bounded_run=False):
    original=Path('.runtime/lsfb-cont-task-v3/candidates.jsonl')
    source=[json.loads(x) for x in original.read_text().splitlines()]
    candidates=[];removed=0;reasons=collections.Counter()
    for row in source:
        if row['split'] not in ('train','val') or row['technical_exclusions']:continue
        seen=set();events=[]
        for event in row['aligned_events']:
            op=lexical_operation(event)
            if op is None:continue
            key=(event['start_ms'],event['end_ms'],json.dumps(op,sort_keys=True))
            if key in seen:removed+=1;continue
            seen.add(key);events.append(event)
        if any(a.get('lemma')==b.get('lemma') and a['end_ms']>b['start_ms'] for a,b in zip(events,events[1:])):
            reasons['ambiguous_overlapping_same_lemma']+=1;continue
        if not 1<=len(events)<=6:reasons['length_outside_1_to_6']+=1;continue
        if any(lexical_operation(e)['op']!='sign' for e in events):reasons['fingerspelling']+=1;continue
        if any(f in row['review_flags'] for f in ['boundary_excluded_events','ellipsis_or_fragment']):reasons['boundaries_or_fragment']+=1;continue
        if len(row['french_text'].split())>25 or '[' in row['french_text']:reasons['complex_french']+=1;continue
        target=[e['lemma'].strip() for e in events]
        candidates.append(dict(id=row['id'],split=row['split'],conversation_id=row['conversation_id'],
                               input='Traduire en signes LSFB: '+row['french_text'],target=target,
                               target_text=json.dumps(target,ensure_ascii=False),review_status='pending',
                               source_content_sha256=row['content_sha256']))
    frequency=collections.Counter(t for r in candidates if r['split']=='train' for t in r['target'])
    vocab={t for t,n in frequency.most_common(128) if n>=5}
    bounded=[r for r in candidates if all(t in vocab for t in r['target'])]
    train=[r for r in bounded if r['split']=='train'];val=[r for r in bounded if r['split']=='val']
    assert not {r['conversation_id'] for r in train}&{r['conversation_id'] for r in val}
    if not bounded_run:
        rng=random.Random(42);train=rng.sample(train,min(32,len(train)));val=rng.sample(val,min(32,len(val)))
    if len(train)<32 or len(val)<8:raise ValueError('Insufficient candidates for diagnostic gate')
    save('DATA.json',dict(source_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
         identical_interval_duplicates_removed=removed,exclusions=dict(reasons),
         bounded_train_candidates=sum(r['split']=='train' for r in bounded),
         bounded_val_candidates=sum(r['split']=='val' for r in bounded),
         vocabulary=sorted(vocab),train=train,val=val,linguistically_validated=False))
    return train,val

def main():
    global ROOT
    parser=argparse.ArgumentParser();parser.add_argument('--bounded',action='store_true');args=parser.parse_args()
    if args.bounded:
        gate=json.loads((ROOT/'RESULT.json').read_text())
        if gate['status']!='memorization_gate_passed':raise ValueError('Memorization gate has not passed')
        ROOT=Path('.runtime/lsfb-bounded-v1')
    if ROOT.exists():raise ValueError('Choose a new output directory; run already exists')
    ROOT.mkdir(parents=True);set_seed(42);torch.set_num_threads(2)
    train,val=prepare(args.bounded);started=time.monotonic()
    # Rescore historical predictions under the new explicit recovery protocol.
    old=[json.loads(x) for x in Path('.runtime/lsfb-cont-local-v3/validation_predictions.jsonl').read_text().splitlines()]
    save('HISTORICAL_RESCORE.json',score([r['prediction'] for r in old],[r['reference'] for r in old]))
    tok=AutoTokenizer.from_pretrained('.runtime/lsfb-cont-local-v3/tokenizer',local_files_only=True)
    base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',torch_dtype=torch.bfloat16,local_files_only=True)
    if args.bounded:
        model=PeftModel.from_pretrained(base,'.runtime/lsfb-recovery-v2/adapter',is_trainable=True).cuda()
    else:
        model=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=16,lora_alpha=32,lora_dropout=0,target_modules=['q','v'])).cuda()
    optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4 if args.bounded else 5e-4)
    scaler=torch.amp.GradScaler('cuda',enabled=False);history=[]
    def evaluate(rows):
        model.eval();pred=[];caps=0
        for i in range(0,len(rows),8):
            enc=tok([r['input'] for r in rows[i:i+8]],padding=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():out=model.generate(**enc,max_new_tokens=96,num_beams=1,do_sample=False,use_cache=True)
            caps+=sum(tok.eos_token_id not in x.tolist() for x in out)
            pred+=tok.batch_decode(out,skip_special_tokens=True)
        exact=invalid=edits=total=0
        for r,p in zip(rows,pred):
            try:
                parsed=json.loads(p)
                if not isinstance(parsed,list) or any(not isinstance(x,str) for x in parsed):raise ValueError()
                exact+=parsed==r['target']
            except (ValueError,TypeError):invalid+=1;parsed=['<INVALID>']+p.split()
            edits+=distance(r['target'],parsed);total+=len(r['target'])
        return dict(exact=exact/len(rows),invalid=invalid/len(rows),wer=edits/total,cap_hits=caps,
                    rows=[dict(id=r['id'],input=r['input'],reference=r['target'],prediction=p) for r,p in zip(rows,pred)])
    best=None
    for step in range(1201):
        if step%100==0:
            tr=evaluate(train);va=evaluate(val)
            summary=dict(step=step,elapsed_seconds=time.monotonic()-started,
                         train={k:v for k,v in tr.items() if k!='rows'},val={k:v for k,v in va.items() if k!='rows'})
            history.append(summary);save('progress.json',dict(status='running',**summary));print(json.dumps(summary),flush=True)
            save('predictions.json',dict(train=tr,val=va));save('history.json',history)
            if args.bounded and va['invalid']==0 and (best is None or va['wer']<best['val']['wer']):
                best=summary
                model.save_pretrained(ROOT/'best_adapter');tok.save_pretrained(ROOT/'tokenizer')
                save('best_predictions.json',dict(train=tr,val=va));save('BEST.json',best)
            if not args.bounded and tr['exact']>=.9 and tr['invalid']==0 and tr['cap_hits']==0:
                model.save_pretrained(ROOT/'adapter');tok.save_pretrained(ROOT/'tokenizer')
                save('RESULT.json',dict(status='memorization_gate_passed',linguistically_validated=False,**summary))
                save('progress.json',dict(status='memorization_gate_passed',**summary));return
        if step==1200:break
        model.train();optim.zero_grad(set_to_none=True)
        batch=random.sample(train,8)
        enc=tok([r['input'] for r in batch],padding=True,return_tensors='pt').to('cuda')
        labels=tok([r['target_text'] for r in batch],padding=True,return_tensors='pt')['input_ids'].to('cuda')
        labels[labels==tok.pad_token_id]=-100
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**enc,labels=labels,use_cache=False).loss
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        scaler.scale(loss).backward();scaler.unscale_(optim)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
        scaler.step(optim);scaler.update()
    model.save_pretrained(ROOT/'adapter');tok.save_pretrained(ROOT/'tokenizer')
    status='bounded_exploratory_complete' if args.bounded else 'memorization_gate_failed'
    save('RESULT.json',dict(status=status,linguistically_validated=False,best=best,**summary))
    save('progress.json',dict(status=status,best=best,**summary))

if __name__=='__main__':main()
