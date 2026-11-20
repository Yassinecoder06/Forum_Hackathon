"""Train only the separate, checksum-verified Belgian LSFB extraction."""
import argparse,collections,hashlib,json,math,random,re,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,set_seed
from peft import get_peft_model,PeftModel,LoraConfig,TaskType
from .scoring import distance
from .experiments import normalize

def measure(rows,texts):
    from urllib.parse import unquote
    preds=[[unquote(t) for t in text.split()] for text in texts]
    total=sum(len(r['target']) for r in rows);edits=sum(distance(r['target'],p) for r,p in zip(rows,preds))
    by={}
    for name,ids in [('one_sign',[i for i,r in enumerate(rows) if len(r['target'])==1]),('multiple_signs',[i for i,r in enumerate(rows) if len(r['target'])>1])]:
        if ids:by[name]={'examples':len(ids),'exact':sum(rows[i]['target']==preds[i] for i in ids)/len(ids),'wer':sum(distance(rows[i]['target'],preds[i]) for i in ids)/sum(len(rows[i]['target']) for i in ids)}
    return {'examples':len(rows),'exact':sum(r['target']==p for r,p in zip(rows,preds))/max(1,len(rows)),'operation_wer':edits/max(1,total),'raw_token_wer':edits/max(1,total),'invalid':sum(not p for p in preds)/max(1,len(rows)),'by_length':by}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,default=Path('/home/jmal/Dataset/lsfb-cont-usable-v1'))
    ap.add_argument('--output',type=Path,default=Path('.runtime/lsfb-extracted-train-v1'));ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--input-mode',choices=['context','current'],default='context');ap.add_argument('--seed',type=int,default=84);args=ap.parse_args()
    if args.output.exists():raise ValueError('Output exists')
    args.output.mkdir();out=args.output;started=time.monotonic();set_seed(args.seed);torch.set_num_threads(2)
    def save(name,x):
        p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,ensure_ascii=False,indent=2));t.replace(p)
    report=json.load(open(args.data/'SCAN_REPORT.json'))
    assert report['language']=='LSFB' and report['status']=='complete'
    for name,digest in report['files'].items():assert hashlib.sha256((args.data/name).read_bytes()).hexdigest()==digest
    load=lambda name:[json.loads(x) for x in (args.data/name).read_text().splitlines()]
    train,val=load('train.jsonl'),load('val.jsonl')
    if args.input_mode=='current':
        for r in train+val:r['input']='Planifier les signes lexicaux en LSFB.\nTexte actuel: '+r['current_french']
    assert all(r['language']=='LSFB' and r['review_status']=='pending' for r in train+val)
    assert not {r['conversation_id'] for r in train}&{r['conversation_id'] for r in val}
    assert not {normalize(r['current_french']) for r in train}&{normalize(r['current_french']) for r in val}
    vocab=set(t for r in train for t in r['target'])
    tok=AutoTokenizer.from_pretrained('.runtime/lsfb-cont-local-v3/tokenizer',use_fast=True,local_files_only=True)
    def encode(r):
        x=tok(r['input']);y=tok(r['target_text'])['input_ids']
        assert len(x['input_ids'])<=384 and len(y)<=192
        assert tok.decode(y,skip_special_tokens=True)==r['target_text']
        return x,y
    encoded=[encode(r) for r in train]
    def batch(indices):
        x=tok.pad([encoded[i][0] for i in indices],padding=True,return_tensors='pt').to('cuda')
        labels=torch.full((len(indices),max(len(encoded[i][1]) for i in indices)),-100,device='cuda',dtype=torch.long)
        for j,i in enumerate(indices):labels[j,:len(encoded[i][1])]=torch.tensor(encoded[i][1],device='cuda')
        return x,labels
    revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
    base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision=revision,torch_dtype=torch.bfloat16,local_files_only=True)
    model=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=16,lora_alpha=32,lora_dropout=.05,target_modules=['q','k','v','o','wi','wo'])).cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.enable_input_require_grads()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01)
    planned=math.ceil(len(train)/8)*args.epochs
    manifest=dict(model='google-t5/t5-base',revision=revision,language='LSFB',data=str(args.data),data_hash=hashlib.sha256((args.data/'SCAN_REPORT.json').read_bytes()).hexdigest(),
                  train=len(train),val=len(val),vocabulary=len(vocab),batch_size=2,accumulation=4,planned_steps=planned,
                  epochs=args.epochs,input_mode=args.input_mode,precision='bfloat16',seed=args.seed,linguistically_validated=False,official_test_labels_opened=False,
                  selection='lowest lexical WER on all narrow development rows; reversible plain labels; no discarded errors',
                  source_limit=384,target_limit=192,generation_new_tokens=192)
    save('MANIFEST.json',manifest);history=[];best=None;step=0
    sample=random.Random(84).sample(val,min(96,len(val)));train_sample=random.Random(85).sample(train,min(32,len(train)))
    def evaluate(rows,phase='selection_validation'):
        model.eval();texts=[];caps=0
        for i in range(0,len(rows),4):
            x=tok([r['input'] for r in rows[i:i+4]],padding=True,return_tensors='pt').to('cuda')
            with torch.inference_mode():p=model.generate(**x,max_new_tokens=192,num_beams=1,do_sample=False,use_cache=True)
            caps+=sum(tok.eos_token_id not in z.tolist() for z in p);texts+=tok.batch_decode(p,skip_special_tokens=True)
            if i%32==0:
                save('progress.json',dict(status=phase,step=step,planned_steps=planned,
                     evaluated_examples=len(texts),total_examples=len(rows),seconds=time.monotonic()-started))
        return dict(**measure(rows,texts),cap_hits=caps),texts
    for epoch in range(args.epochs):
        order=list(range(len(train)));random.shuffle(order)
        for start in range(0,len(order),8):
            group=order[start:start+8];optimizer.zero_grad(set_to_none=True);model.train();losses=[]
            for j in range(0,len(group),2):
                ids=group[j:j+2];x,y=batch(ids)
                with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**x,labels=y,use_cache=False).loss
                if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
                (loss*len(ids)/len(group)).backward();losses.append(float(loss.detach()))
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1)
            lr=2e-4*min(1,(step+1)/max(1,int(planned*.05)))*max(.05,1-step/planned)
            for param in optimizer.param_groups:param['lr']=lr
            optimizer.step();step+=1
            if step%20==0:
                progress=dict(status='training',step=step,planned_steps=planned,epoch=epoch+start/len(order),loss=sum(losses)/len(losses),seconds=time.monotonic()-started)
                save('progress.json',progress);print(json.dumps(progress),flush=True)
        tm,_=evaluate(train_sample,'training_sample_evaluation');vm,preds=evaluate(sample)
        record=dict(epoch=epoch+1,step=step,train_sample=tm,val_selection=vm,seconds=time.monotonic()-started);history.append(record);save('history.json',history);print(json.dumps(record),flush=True)
        if best is None or vm['raw_token_wer']<best['val_selection']['raw_token_wer']:
            best=record;model.save_pretrained(out/'best_adapter');tok.save_pretrained(out/'tokenizer');save('BEST.json',best)
    # Freshly reload selected weights for the complete Belgian validation split.
    del optimizer,model,base;torch.cuda.empty_cache()
    base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision=revision,torch_dtype=torch.bfloat16,local_files_only=True)
    model=PeftModel.from_pretrained(base,out/'best_adapter').cuda().eval()
    full,preds=evaluate(val,'full_validation')
    seen=[i for i,r in enumerate(val) if all(t in vocab for t in r['target'])]
    full['seen_vocabulary_subset']=measure([val[i] for i in seen],[preds[i] for i in seen])
    save('validation_predictions.json',[dict(id=r['id'],reference=r['target'],prediction=p,pose_path=r['pose_path']) for r,p in zip(val,preds)])
    result=dict(status='complete',best=best,validation=full,seconds=time.monotonic()-started,linguistically_validated=False)
    save('RESULT.json',result);save('progress.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
