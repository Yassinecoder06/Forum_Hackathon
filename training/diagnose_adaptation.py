"""Audit tokenizer/teacher forcing and overfit FOUR genuine TRAIN examples only."""
import json,random,time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from peft import LoraConfig,TaskType,get_peft_model
from catalog import ROOT,sha,dump
from experiment import read,validate_exports,metrics
TARGETS=['q','k','v','o','wi_0','wi_1','wo']
CONFIG=dict(base='google/mt5-small',revision='73fb5dbe4756edadc8fbe8c769b0a109493acf7a',rank=16,alpha=32,dropout=.05,target_modules=TARGETS,lr=1e-3,seed=42)
def main():
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    folder=ROOT/'research/runs/mt5-train-only-diagnostic';folder.mkdir(exist_ok=False)
    data=ROOT/'data/text_to_gloss/v002';rows=read('train',data,'source-annotated-research');validate_exports(data,{'train':rows},'source-annotated-research')
    tok=AutoTokenizer.from_pretrained(ROOT/'training/weights/mt5-small',use_fast=False)
    base=AutoModelForSeq2SeqLM.from_pretrained(ROOT/'training/weights/mt5-small')
    audit=[]
    for r in rows:
        ids=tok(text_target=r['gloss_sequence'])['input_ids'];decoded=tok.decode(ids,skip_special_tokens=True)
        audit.append(dict(pair_id=r['pair_id'],raw_target=r['gloss_sequence'],target_ids=ids,target_tokens=tok.convert_ids_to_tokens(ids),round_trip=decoded,round_trip_equal=decoded==r['gloss_sequence'],unk_count=ids.count(tok.unk_token_id),eos_at_end=ids[-1]==tok.eos_token_id,sentinel_target_count=sum('<extra_id_' in tok.convert_ids_to_tokens(i) for i in ids)))
    selected=sorted(rows,key=lambda r:r['pair_id'])[:4]
    model=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=16,lora_alpha=32,lora_dropout=.05,target_modules=TARGETS,bias='none')).cuda()
    model.config.use_cache=False
    parameters=[dict(name=n,count=p.numel()) for n,p in model.named_parameters() if p.requires_grad]
    def encode(r):
        x=tok('French to LSF gloss: '+r['french_text'],return_tensors='pt').to('cuda')
        y=tok(text_target=r['gloss_sequence'],return_tensors='pt')['input_ids'].cuda();y[y==tok.pad_token_id]=-100
        return {**x,'labels':y}
    batch=tok(text_target=[r['gloss_sequence'] for r in selected],padding=True,return_tensors='pt')['input_ids'];labels=batch.clone();labels[batch==tok.pad_token_id]=-100
    shifted=model.prepare_decoder_input_ids_from_labels(labels=labels)
    shift_ok=bool(torch.all(shifted[:,0]==model.config.decoder_start_token_id) and torch.equal(shifted[:,1:],torch.where(labels[:,:-1]==-100,tok.pad_token_id,labels[:,:-1])))
    assert shift_ok and not sum(a['unk_count']+a['sentinel_target_count'] for a in audit) and all(a['eos_at_end'] for a in audit)
    dump(folder/'tokenizer_audit.json',dict(config=CONFIG,train_sha256=sha(data/'train.jsonl'),records=61,tokenizer_class=type(tok).__name__,tokenizer_vocab_size=len(tok),model_vocab_size=model.config.vocab_size,pad_id=tok.pad_token_id,eos_id=tok.eos_token_id,unk_id=tok.unk_token_id,decoder_start_id=model.config.decoder_start_token_id,padding_label_value=-100,decoder_shift_verified=shift_ok,trainable_parameters_before_freeze=sum(p['count'] for p in parameters),trainable_tensors=parameters,round_trip_mismatches=[a['pair_id'] for a in audit if not a['round_trip_equal']],all_train_targets=audit,selected_pair_ids=[r['pair_id'] for r in selected],selection_rule='first four training records sorted by pair_id; no heldout data read'))
    opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=1e-3,weight_decay=0)
    logs=[];start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    def evaluate(step):
        model.eval();losses=[];pred=[]
        with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
            for r in selected:
                x=encode(r);losses.append(model(**x).loss.item());x.pop('labels')
                ids=model.generate(**x,do_sample=False,num_beams=1,max_new_tokens=128)
                pred.append(tok.decode(ids[0],skip_special_tokens=True))
        refs=[r['gloss_sequence'] for r in selected];m=metrics(refs,pred,{t for s in refs for t in s.split()})
        result=dict(step=step,teacher_forced_loss=float(np.mean(losses)),train_exact_match=m['exact_match'],train_wer=m['token_wer'],predictions=pred,elapsed_seconds=time.monotonic()-start)
        logs.append(result);print(json.dumps(result),flush=True)
    evaluate(0)
    for step in range(1,201):
        model.train();opt.zero_grad()
        for r in selected:
            with torch.autocast('cuda',dtype=torch.bfloat16):loss=model(**encode(r)).loss
            assert torch.isfinite(loss)
            (loss/4).backward()
        grad=torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
        if step==1: first_gradient_norm=float(grad)
        opt.step()
        if step%25==0:evaluate(step)
        if step%25==0 and logs[-1]['train_exact_match']==1.0:break
    dump(folder/'diagnostic_results.json',dict(config=CONFIG,first_gradient_norm=first_gradient_norm,steps=step,history=logs,selected_examples=[dict(pair_id=r['pair_id'],french_text=r['french_text'],reference=r['gloss_sequence']) for r in selected],runtime_seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20,scope='TRAIN ONLY: no validation or test read, no checkpoint deployed',conclusion='Optimization verified' if logs[-1]['train_exact_match']==1 else 'Training loss/outputs recorded; exact overfit not attained'))
if __name__=='__main__':main()
