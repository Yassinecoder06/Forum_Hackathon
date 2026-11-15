"""Actual seq2seq research experiment; never claims human-approved translation."""
import argparse, json, random, sqlite3, time, hashlib
from contextlib import nullcontext
from difflib import SequenceMatcher
from pathlib import Path
import numpy as np
import torch
import sacrebleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from catalog import ROOT, norm, sha, dump, approved_eligible, research_eligible

def read(split,data,mode):
    rows=[json.loads(x) for x in (data/(split+'.jsonl')).read_text().splitlines()]
    gate=approved_eligible if mode=='approved' else research_eligible
    if not rows or not all(gate(r) for r in rows):raise ValueError('Data failed '+mode+' gate')
    return rows

def validate_exports(data, splits, mode):
    manifest=json.loads((data/'dataset_manifest.json').read_text())
    for split,rows in splits.items():
        if sha(data/(split+'.jsonl'))!=manifest['files'][split+'.jsonl']:
            raise ValueError('Frozen dataset checksum mismatch: '+split)
        if mode=='source-annotated-research':
            with sqlite3.connect('file:'+str(ROOT/'data/catalog/catalog.sqlite')+'?mode=ro',uri=True) as db:
                for r in rows:
                    found=db.execute('SELECT record_json FROM pairs WHERE pair_id=?',(r['pair_id'],)).fetchone()
                    if not found or json.loads(found[0])!=r:
                        raise ValueError('Record differs from source-derived catalog: '+r['pair_id'])
                    source=db.execute('SELECT revision,license_id FROM sources WHERE source_id=?',(r['source_id'],)).fetchone()
                    if source!=(r['source_revision'],r['license_id']):
                        raise ValueError('Unverified source revision/license')

def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def metrics(refs,preds,vocab):
    rt=[r.split() for r in refs];pt=[p.split() for p in preds]; nt=sum(map(len,pt))
    return dict(exact_match=sum(a==b for a,b in zip(refs,preds))/len(refs),
        token_wer=sum(distance(a,b) for a,b in zip(rt,pt))/sum(map(len,rt)),
        corpus_bleu=sacrebleu.corpus_bleu(preds,[refs],tokenize='none').score,
        bleu_signature='sacrebleu '+sacrebleu.__version__+'; case:mixed; tok:none; smooth:exp; effective_order:no; scale:0-100',
        predicted_oov_vs_train=sum(t not in vocab for ts in pt for t in ts)/max(1,nt),
        reference_oov_vs_train=sum(t not in vocab for ts in rt for t in ts)/sum(map(len,rt)),
        motion_registry_oov=None,motion_registry_note='No approved source-gloss-to-motion mapping; train vocabulary OOV is separate',
        examples=len(refs),reference_tokens=sum(map(len,rt)),predicted_tokens=nt)

def baselines(train,rows):
    # Training-only retrieval. Stable tie breaking by pair ID. No test lexicon construction.
    retrieved=[];lexical=[]
    lex={}
    for r in sorted(train,key=lambda r:r['pair_id']):
        for g in r['gloss_sequence'].split():
            label=norm(g)
            if label and len(label.split())==1:lex.setdefault(label,g)
    for r in rows:
        best=max(sorted(train,key=lambda r:r['pair_id']),key=lambda t:SequenceMatcher(None,norm(t['french_text']),norm(r['french_text'])).ratio())
        retrieved.append(best['gloss_sequence'])
        lexical.append(' '.join(lex.get(t,'<UNK>') for t in norm(r['french_text']).split()))
    return {'retrieval':retrieved,'lexical':lexical}

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=ROOT/'data/text_to_gloss/v001')
    p.add_argument('--base',default='google-t5/t5-small');p.add_argument('--base-path',required=True)
    p.add_argument('--revision',required=True);p.add_argument('--run',required=True)
    p.add_argument('--mode',choices=['approved','source-annotated-research'],default='approved')
    p.add_argument('--lora',action='store_true');p.add_argument('--epochs',type=int,default=8)
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda');p.add_argument('--lr',type=float,default=3e-4)
    p.add_argument('--rank',type=int,default=8);p.add_argument('--alpha',type=int,default=16)
    p.add_argument('--dropout',type=float,default=.05);p.add_argument('--targets',default='q,v')
    p.add_argument('--accumulation',type=int,default=4);p.add_argument('--patience',type=int,default=0)
    p.add_argument('--min-delta',type=float,default=0.0);p.add_argument('--warmup-ratio',type=float,default=0.0)
    p.add_argument('--weight-decay',type=float,default=.01)
    p.add_argument('--validation-loss',choices=['token','example'],default='token')
    a=p.parse_args();a.base_path=str(Path(a.base_path).resolve());random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    data={s:read(s,a.data,a.mode) for s in ['train','val']};train=data['train']
    validate_exports(a.data,data,a.mode)
    for s in ['val']:
        assert not {r['split_group'] for r in train}&{r['split_group'] for r in data[s]}
    out=ROOT/'research/runs'/a.run;out.mkdir(parents=True,exist_ok=False)
    ckpt=ROOT/'training/checkpoints'/a.run;ckpt.mkdir(parents=True,exist_ok=False)
    tokenizer=AutoTokenizer.from_pretrained(a.base_path,use_fast=not a.base.startswith("google/mt5"))
    model=AutoModelForSeq2SeqLM.from_pretrained(a.base_path)
    if a.lora:
        from peft import LoraConfig,TaskType,get_peft_model
        model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=a.rank,lora_alpha=a.alpha,lora_dropout=a.dropout,target_modules=a.targets.split(','),bias='none'))
    trainable_parameters_before_freeze=sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_tensors=[dict(name=n,parameters=p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    if not trainable_parameters_before_freeze:raise ValueError('No trainable parameters')
    dump(out/'training_plan.json',dict(arguments={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},trainable_parameters_before_freeze=trainable_parameters_before_freeze,
        trainable_tensors=trainable_tensors,experiment_script_sha256=sha(Path(__file__)),
        heldout_policy='Test file is not read or decoded until checkpoint selection completes',
        validation_loss_aggregation=a.validation_loss,early_stopping='validation loss only',
        gradient_remainder='divide last accumulation group by its actual size',
        dropout='base model dropout retained plus specified adapter dropout',
        data_manifest_sha256=sha(a.data/'dataset_manifest.json')))
    model.to(a.device);model.config.use_cache=False
    autocast=lambda:torch.autocast('cuda',dtype=torch.bfloat16) if a.device=='cuda' else nullcontext()
    if a.device=='cuda':torch.cuda.reset_peak_memory_stats()
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=a.lr,weight_decay=a.weight_decay)
    total_steps=((len(train)+a.accumulation-1)//a.accumulation)*a.epochs
    warmup_steps=int(total_steps*a.warmup_ratio)
    from transformers import get_linear_schedule_with_warmup
    scheduler=get_linear_schedule_with_warmup(opt,warmup_steps,total_steps) if a.warmup_ratio else None
    def encode(rows):
        x=tokenizer(['French to LSF gloss: '+r['french_text'] for r in rows],return_tensors='pt',padding=True)
        y=tokenizer(text_target=[r['gloss_sequence'] for r in rows],return_tensors='pt',padding=True)['input_ids']
        if max(x['input_ids'].shape[1],y.shape[1])>512:raise ValueError('Overlength; do not silently truncate')
        y[y==tokenizer.pad_token_id]=-100
        return {**{k:v.to(a.device) for k,v in x.items()},'labels':y.to(a.device)}
    start=time.monotonic();logs=[];best=float('inf');best_epoch=None;stale_epochs=0;optimizer_steps=0
    for epoch in range(a.epochs):
        model.train();order=list(train);random.shuffle(order);losses=[]
        opt.zero_grad()
        for i,row in enumerate(order):
            with autocast():loss=model(**encode([row])).loss
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            group_start=(i//a.accumulation)*a.accumulation
            group_size=min(a.accumulation,len(order)-group_start)
            (loss/group_size).backward();losses.append(loss.item())
            if (i+1)%a.accumulation==0 or i+1==len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();opt.zero_grad();optimizer_steps+=1
                if scheduler:scheduler.step()
        model.eval();vl=[];val_loss_sum=0.0;val_token_count=0
        with torch.no_grad():
            for row in data['val']:
                x=encode([row]);n=int((x['labels']!=-100).sum())
                with autocast():value=model(**x).loss.item()
                vl.append(value);val_loss_sum+=value*n;val_token_count+=n
        val_token_loss=val_loss_sum/val_token_count
        selection_loss=val_token_loss if a.validation_loss=='token' else float(np.mean(vl))
        if not np.isfinite(selection_loss):raise ValueError('Nonfinite validation loss')
        log=dict(epoch=epoch+1,train_loss=float(np.mean(losses)),val_loss=selection_loss,
            val_loss_token_weighted=val_token_loss,val_loss_mean_example=float(np.mean(vl)),
            optimizer_steps=optimizer_steps,learning_rate=opt.param_groups[0]['lr'],elapsed_seconds=time.monotonic()-start)
        logs.append(log);print(json.dumps(log),flush=True)
        # Validation only chooses checkpoint. Test decoded once after selection.
        if log['val_loss']<best-a.min_delta:
            best=log['val_loss'];best_epoch=epoch+1;stale_epochs=0;model.save_pretrained(ckpt);tokenizer.save_pretrained(ckpt)
        else:stale_epochs+=1
        dump(out/'loss_history.json',logs)
        if a.patience and stale_epochs>=a.patience:
            print(json.dumps(dict(early_stop_epoch=epoch+1,best_epoch=best_epoch,patience=a.patience)),flush=True)
            break
    train_seconds=time.monotonic()-start
    if a.lora:
        from peft import PeftModel
        model.load_adapter(ckpt,adapter_name='selected');model.set_adapter('selected')
    else:
        del model;torch.cuda.empty_cache();model=AutoModelForSeq2SeqLM.from_pretrained(ckpt).to(a.device)
    model.eval();model.config.use_cache=True
    # Training/selection is finished. Read frozen test now and decode it exactly once.
    dump(out/'selection_complete.json',dict(best_epoch=best_epoch,epochs_executed=len(logs),best_validation_loss=best,
        validation_loss_aggregation=a.validation_loss,train_seconds=train_seconds,test_read=False))
    data['test']=read('test',a.data,a.mode);validate_exports(a.data,{'test':data['test']},a.mode)
    assert not {r['split_group'] for r in train}&{r['split_group'] for r in data['test']}
    vocab={t for r in train for t in r['gloss_sequence'].split()};results={}
    for split in ['val','test']:
        rows=data[split];pred=[]
        with torch.no_grad():
            for row in rows:
                x=encode([row]);x.pop('labels')
                with autocast():y=model.generate(**x,max_new_tokens=128,do_sample=False,num_beams=1)
                pred.append(tokenizer.decode(y[0],skip_special_tokens=True))
        candidates={**baselines(train,rows),'model':pred};refs=[r['gloss_sequence'] for r in rows]
        results[split]={name:metrics(refs,ps,vocab) for name,ps in candidates.items()}
        with (out/(split+'_predictions.jsonl')).open('w') as f:
            for i,row in enumerate(rows):
                f.write(json.dumps(dict(pair_id=row['pair_id'],source_id=row['source_id'],source_revision=row['source_revision'],split_group=row['split_group'],source_segment_ids=row['source_segment_ids'],french_text=row['french_text'],reference=refs[i],predictions={k:v[i] for k,v in candidates.items()},model_base=a.base,model_revision=a.revision,linguistic_review='PENDING_LOCAL_REVIEW',fuzzy_replacement=False),ensure_ascii=False)+'\n')
    report=dict(run_id=a.run,model_base=a.base,model_revision=a.revision,mode=a.mode,lora=a.lora,
        trainable_parameters=trainable_parameters_before_freeze,
        trainable_parameter_count_stage='before selected adapter load/freeze',
        epochs=len(logs),max_epochs=a.epochs,best_epoch=best_epoch,patience=a.patience,
        optimizer_steps=optimizer_steps,validation_loss_aggregation=a.validation_loss,
        lora_config=dict(rank=a.rank,alpha=a.alpha,dropout=a.dropout,targets=a.targets.split(',')),
        warmup_steps=warmup_steps,weight_decay=a.weight_decay,
        learning_rate=a.lr,batch_size=1,gradient_accumulation=a.accumulation,precision='bf16 autocast' if a.device=='cuda' else 'fp32',seed=42,
        counts={s:len(rs) for s,rs in data.items()},group_counts={s:len(set(r['split_group'] for r in rs)) for s,rs in data.items()},
        dataset_manifest_sha256=sha(a.data/'dataset_manifest.json'),train_seconds=train_seconds,total_seconds=time.monotonic()-start,
        cuda_device=torch.cuda.get_device_name() if a.device=='cuda' else None,
        peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if a.device=='cuda' else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if a.device=='cuda' else None,
        torch_version=torch.__version__,loss_history=logs,metrics=results,
        checkpoint=str(ckpt),limitations='ENGINEERING PROOF ONLY; free translations; tiny single-source corpus; no local linguistic approval; no SOTA claim')
    dump(out/'metrics.json',report)
    checkpoint_hashes={p.name:sha(p) for p in ckpt.iterdir() if p.is_file()}
    dump(ckpt/'research_metadata.json',dict(checkpoint_sha256=checkpoint_hashes,model_base=a.base,model_revision=a.revision,run_id=a.run,lora=a.lora,tokenizer_use_fast=not a.base.startswith('google/mt5'),prefix='French to LSF gloss: ',linguistic_review='PENDING_LOCAL_REVIEW',dataset_manifest_sha256=report['dataset_manifest_sha256']))
    db=sqlite3.connect(ROOT/'data/catalog/catalog.sqlite');db.execute('INSERT INTO model_runs VALUES(?,?,?,?,?,?,?)',(a.run,a.base,a.revision,report['dataset_manifest_sha256'],json.dumps(vars(a),default=str),json.dumps(report),str(ckpt)));db.commit();db.close()
    print(json.dumps(report,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
