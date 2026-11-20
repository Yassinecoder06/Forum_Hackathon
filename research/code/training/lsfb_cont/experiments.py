"""Controlled local experiments. Development data only; no linguistic approvals."""
import argparse,json,random,time,re,collections,hashlib
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,set_seed
from peft import get_peft_model,LoraConfig,TaskType
from .scoring import distance

def normalize(text):
    return ' '.join(re.findall(r'\w+',text.casefold()))

def metrics(rows,texts,vocab):
    if len(rows)!=len(texts):
        raise ValueError('Prediction/reference count mismatch')
    exact=invalid=edits=total=unsupported=0
    for r,p in zip(rows,texts):
        try:
            seq=json.loads(p)
            if not isinstance(seq,list) or not seq or any(not isinstance(t,str) for t in seq):raise ValueError()
            exact+=seq==r['target'];unsupported+=any(t not in vocab for t in seq)
        except (ValueError,TypeError):invalid+=1;seq=['<INVALID>']+p.split()
        edits+=distance(r['target'],seq);total+=len(r['target'])
    return dict(examples=len(rows),exact=exact/max(1,len(rows)),invalid=invalid/max(1,len(rows)),
                wer=edits/max(1,total),unsupported=unsupported/max(1,len(rows)))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',required=True);ap.add_argument('--revision',required=True)
    ap.add_argument('--name',required=True);ap.add_argument('--steps',type=int,default=1200)
    ap.add_argument('--clean-ambiguity',action='store_true');ap.add_argument('--all-linear',action='store_true')
    args=ap.parse_args();out=Path('.runtime')/args.name
    if out.exists():raise ValueError('Run directory exists')
    out.mkdir();started=time.monotonic();set_seed(73);torch.set_num_threads(2)
    def save(name,x):
        p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
    source=Path('.runtime/lsfb-bounded-v1/DATA.json');data=json.load(open(source));train=data['train'];val=data['val'];vocab=set(data['vocabulary'])
    original_inputs={normalize(r['input']) for r in train};novel=[i for i,r in enumerate(val) if normalize(r['input']) not in original_inputs]
    groups=collections.defaultdict(list)
    for r in train:groups[normalize(r['input'])].append(r)
    ambiguous={k for k,rs in groups.items() if len({tuple(r['target']) for r in rs})>1}
    if args.clean_ambiguity:
        train=[r for r in train if normalize(r['input']) not in ambiguous]
    save('MANIFEST.json',dict(**vars(args),seed=73,learning_rate=3e-4,batch=8,precision='bfloat16',
         source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),train_rows=len(train),val_rows=len(val),
         novel_validation_indices=novel,ambiguous_input_groups=len(ambiguous),linguistically_validated=False,
         official_test_opened=False,train_ids=[r['id'] for r in train]))
    tokenizer_source='.runtime/lsfb-cont-local-v3/tokenizer' if args.model=='google-t5/t5-base' else args.model
    tok=AutoTokenizer.from_pretrained(tokenizer_source,revision=args.revision,use_fast=False)
    base=AutoModelForSeq2SeqLM.from_pretrained(args.model,revision=args.revision,torch_dtype=torch.bfloat16)
    modules=['q','k','v','o','wi','wo'] if args.all_linear else ['q','v']
    model=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=16,lora_alpha=32,lora_dropout=0,target_modules=modules)).cuda()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4)
    def generate(inputs,beams=1):
        model.eval();texts=[];caps=0
        for i in range(0,len(inputs),8):
            batch=tok(inputs[i:i+8],padding=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():pred=model.generate(**batch,max_new_tokens=96,num_beams=beams,do_sample=False,use_cache=True)
            caps+=sum(tok.eos_token_id not in p.tolist() for p in pred)
            texts+=tok.batch_decode(pred,skip_special_tokens=True)
        return texts,caps
    def evaluate(rows,beams=1,inputs=None):
        texts,caps=generate(inputs or [r['input'] for r in rows],beams)
        return dict(**metrics(rows,texts,vocab),cap_hits=caps),texts
    best=None;history=[]
    for step in range(args.steps+1):
        if step%200==0 or step==args.steps:
            tm,_=evaluate(train);vm,preds=evaluate(val)
            nm=metrics([val[i] for i in novel],[preds[i] for i in novel],vocab)
            record=dict(step=step,seconds=time.monotonic()-started,train=tm,val=vm,novel_val=nm)
            history.append(record);save('progress.json',dict(status='running',**record));save('history.json',history)
            print(json.dumps(record),flush=True)
            # Novel-source metric drives selection; full51 remains a development diagnostic.
            if nm['invalid']==0 and nm['unsupported']==0 and (best is None or nm['wer']<best['novel_val']['wer']):
                best=record;model.save_pretrained(out/'best_adapter');tok.save_pretrained(out/'tokenizer')
                save('BEST.json',best);save('best_predictions.json',[dict(**r,prediction=p) for r,p in zip(val,preds)])
        if step==args.steps:break
        model.train();optimizer.zero_grad(set_to_none=True)
        batch=random.sample(train,8);inputs=tok([r['input'] for r in batch],padding=True,return_tensors='pt').to('cuda')
        labels=tok([r['target_text'] for r in batch],padding=True,return_tensors='pt')['input_ids'].to('cuda');labels[labels==tok.pad_token_id]=-100
        with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**inputs,labels=labels,use_cache=False).loss
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1);optimizer.step()
    model.save_pretrained(out/'final_adapter');tok.save_pretrained(out/'tokenizer')
    save('RESULT.json',dict(status='complete',best=best,final=record,linguistically_validated=False))
    save('progress.json',dict(status='complete',best=best,final=record))

if __name__=='__main__':main()
