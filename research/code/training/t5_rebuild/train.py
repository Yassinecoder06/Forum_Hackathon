"""Full T5-base LSFB run, selected by generated validation WER, never test loss."""
import argparse,hashlib,json,shutil,time,math
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tokenizers import AddedToken
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,Seq2SeqTrainer,Seq2SeqTrainingArguments,EarlyStoppingCallback,TrainerCallback,set_seed
from prepare import dump
from metrics import evaluate_arrays
BASE='google-t5/t5-base';REV='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1';PREFIX='translate French to LSFB glosses: '

def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('gate',type=Path);p.add_argument('output',type=Path);p.add_argument('--epochs',type=int,default=20);p.add_argument('--patience',type=int,default=3);p.add_argument('--fp16',action='store_true');a=p.parse_args()
    gate=json.loads(a.gate.read_text());assert gate['passed'] and gate['raw_train_exact']==1.0
    source=json.loads((a.data/'manifest.json').read_text());assert source['language']=='LSFB' and source['train_pairs']>=100 and source['val_pairs']>=10
    for name,digest in source['files'].items():assert hashlib.sha256((a.data/name).read_bytes()).hexdigest()==digest
    if source['val_inventory_coverage']<.9:raise ValueError('Validation inventory coverage below 90%; expand training-source inventory before scaling')
    vocabulary=json.loads((a.data/'vocabulary.json').read_text())['units'];a.output.mkdir(parents=True,exist_ok=True)
    config={'base':BASE,'revision':REV,'dataset_manifest_sha256':hashlib.sha256((a.data/'manifest.json').read_bytes()).hexdigest(),'optimizer':'AdamW','lr':2e-5,'weight_decay':.01,'batch_size':4,'max_epochs':a.epochs,'patience':a.patience,'selection':'generated constrained validation WER; raw evaluation reported after selection','precision':'fp16 autocast with FP32 master weights' if a.fp16 else 'float32','max_source_tokens':256,'max_target_units':96,'generation_cap':128,'beam_size':1,'target':'atomic canonical LSFB units; unseen references remain errors','seed':42,'gate_sha256':hashlib.sha256(a.gate.read_bytes()).hexdigest(),'test_opened':False}
    plan=a.output/'RUN_PLAN.json'
    if plan.exists() and json.loads(plan.read_text())!=config:raise ValueError('Run configuration changed; use a separate output directory')
    dump(plan,config);set_seed(42);torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(BASE,revision=REV,use_fast=False);model=AutoModelForSeq2SeqLM.from_pretrained(BASE,revision=REV)
    old_size=len(tok);pieces=[tok.encode(u.split(':',1)[-1],add_special_tokens=False) for u in vocabulary]
    names=[f'<lsfb_{i:05d}>' for i in range(len(vocabulary))];assert tok.add_tokens([AddedToken(n,normalized=False,special=False) for n in names])==len(names)
    model.resize_token_embeddings(len(tok),mean_resizing=False);ids={u:tok.convert_tokens_to_ids(n) for u,n in zip(vocabulary,names)};reverse={i:u for u,i in ids.items()}
    with torch.no_grad():
        weight=model.get_input_embeddings().weight;norm=weight[:old_size].norm(dim=1).mean()
        for u,part in zip(vocabulary,pieces):
            valid=[i for i in part if i not in [tok.unk_token_id,tok.eos_token_id,tok.pad_token_id]]
            if valid:
                value=weight[valid].mean(0);weight[ids[u]].copy_(value*norm/value.norm().clamp_min(1e-8))
    model.config.use_cache=False
    dump(a.output/'unit_vocabulary.json',{'unit_to_id':ids,'base_vocabulary_size':old_size,'eos_id':tok.eos_token_id,'pad_id':tok.pad_token_id,'unknown_reference_id':tok.unk_token_id,'unknown_reference_policy':'retained in evaluation denominator and cannot match an invalid predicted token','source_vocabulary_sha256':hashlib.sha256((a.data/'vocabulary.json').read_bytes()).hexdigest()})
    class Pairs(Dataset):
        def __init__(self,name):
            self.rows=[];self.items=[];excluded=[]
            for r in [json.loads(l) for l in (a.data/(name+'.jsonl')).read_text().splitlines()]:
                x=tok(PREFIX+r['french_text'],return_token_type_ids=False)
                if len(x['input_ids'])>256 or len(r['units'])>96:
                    excluded.append({'id':r['id'],'source_tokens':len(x['input_ids']),'target_units':len(r['units'])});continue
                if name=='train' and any(u not in ids for u in r['units']):raise ValueError('Train target absent from source vocabulary')
                x['labels']=[ids.get(u,tok.unk_token_id) for u in r['units']]+[tok.eos_token_id];self.rows.append(r);self.items.append(x)
            dump(a.output/(name+'_length_exclusions.json'),excluded)
        def __len__(self):return len(self.items)
        def __getitem__(self,i):return self.items[i]
    train=Pairs('train');val=Pairs('val');assert len(train)>=100 and len(val)>=10
    dump(a.output/'effective_dataset.json',{'train_examples':len(train),'val_examples':len(val),'vocabulary_units':len(ids),'length_filter':'fixed before generation; no label truncation','val_refs_outside_inventory':sum(u not in ids for r in val.rows for u in r['units']),'raw_dataset_manifest':source})
    class Finite(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if any(isinstance(v,(float,int)) and not math.isfinite(v) for v in (logs or {}).values()):raise ValueError('Nonfinite optimization log; stop and inspect before scaling')
    class AtomicTrainer(Seq2SeqTrainer):
        constrained=True
        def evaluate(self,*args,**kwargs):
            kwargs.update(max_new_tokens=128,num_beams=1,use_cache=True)
            if self.constrained:kwargs['prefix_allowed_tokens_fn']=lambda batch_id,sequence:list(reverse)+[tok.eos_token_id]
            return super().evaluate(*args,**kwargs)
        def predict(self,*args,**kwargs):
            kwargs.update(max_new_tokens=128,num_beams=1,use_cache=True)
            if self.constrained:kwargs['prefix_allowed_tokens_fn']=lambda batch_id,sequence:list(reverse)+[tok.eos_token_id]
            return super().predict(*args,**kwargs)
    args=Seq2SeqTrainingArguments(output_dir=str(a.output/'checkpoints'),num_train_epochs=a.epochs,per_device_train_batch_size=4,per_device_eval_batch_size=16,learning_rate=2e-5,weight_decay=.01,optim='adamw_torch',lr_scheduler_type='constant',gradient_checkpointing=True,fp16=a.fp16,bf16=False,predict_with_generate=True,eval_strategy='epoch',save_strategy='epoch',save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='wer',greater_is_better=False,logging_steps=20,logging_nan_inf_filter=False,report_to='none',seed=42)
    trainer=AtomicTrainer(model=model,args=args,train_dataset=train,eval_dataset=val,data_collator=DataCollatorForSeq2Seq(tok,model=model),processing_class=tok,compute_metrics=lambda p:evaluate_arrays(p.predictions,p.label_ids,reverse,tok.eos_token_id,tok.pad_token_id),callbacks=[Finite(),EarlyStoppingCallback(early_stopping_patience=a.patience)])
    checkpoints=[]
    for folder in (a.output/'checkpoints').glob('checkpoint-*'):
        if all((folder/name).exists() for name in ['model.safetensors','optimizer.pt','trainer_state.json','scheduler.pt']):checkpoints.append(folder)
    latest=str(max(checkpoints,key=lambda p:int(p.name.split('-')[-1]))) if checkpoints else None
    print(json.dumps({'stage':'training','effective_train':len(train),'effective_val':len(val),'atomic_units':len(ids),'resume':latest}),flush=True)
    started=time.monotonic();trainer.train(resume_from_checkpoint=latest)
    dump(a.output/'selection_complete.json',{'best_checkpoint':trainer.state.best_model_checkpoint,'best_validation_wer':trainer.state.best_metric,'steps':trainer.state.global_step,'seconds':time.monotonic()-started,'test_shard_0_read':False})
    trainer.save_model(str(a.output/'final'));tok.save_pretrained(a.output/'final');shutil.copyfile(a.output/'unit_vocabulary.json',a.output/'final/unit_vocabulary.json')
    constrained=trainer.predict(val,metric_key_prefix='validation_constrained');trainer.constrained=False
    raw=trainer.predict(val,metric_key_prefix='validation_raw')
    dump(a.output/'validation_results.json',{'constrained':constrained.metrics,'raw':raw.metrics,'target_policy':source['target_policy'],'linguistic_approval':False})
    dump(a.output/'history.json',trainer.state.log_history);dump(a.output/'TRAINING_COMPLETE.json',{'steps':trainer.state.global_step,'final_model':'final','test_evaluated':False,'next_stage':'locked official test evaluation with same vocabulary and length rules'})
    print(json.dumps({'stage':'training_complete','validation_constrained':constrained.metrics,'validation_raw':raw.metrics}),flush=True)
if __name__=='__main__':main()
