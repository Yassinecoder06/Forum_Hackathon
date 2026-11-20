"""T5-base train-only learning gate with atomic LSFB units and EOS checks.
This measures memorization; it is never a held-out translation benchmark.
"""
import argparse,hashlib,json,random,time
from pathlib import Path
import torch
from tokenizers import AddedToken
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,set_seed
from prepare import dump
BASE='google-t5/t5-base';REV='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1'
PREFIX='translate French to LSFB glosses: '

def decode_units(ids,reverse,eos,pad):
    ids=list(ids)
    if ids and ids[0]==pad:ids=ids[1:]
    units=[];invalid=[];stopped=False
    for idx in ids:
        if idx==eos:stopped=True;break
        if idx in reverse:units.append(reverse[idx])
        else:invalid.append(idx);units.append(f'<INVALID:{idx}>')
    return units,invalid,stopped

def main():
    p=argparse.ArgumentParser();p.add_argument('data',type=Path);p.add_argument('output',type=Path);p.add_argument('--max-steps',type=int,default=800);p.add_argument('--check-every',type=int,default=20);a=p.parse_args()
    set_seed(42);torch.set_num_threads(2);source=json.loads(a.data.read_text());rows=source['rows'];vocabulary=source['vocabulary']
    assert source['language']=='LSFB' and 8<=len(rows)<=16 and len({r['french_text'] for r in rows})==len(rows)
    assert all(set(r['units'])<=set(vocabulary) and r['units'] for r in rows)
    a.output.mkdir(parents=True,exist_ok=False)
    dump(a.output/'experiment.json',{'base':BASE,'revision':REV,'data_sha256':hashlib.sha256(a.data.read_bytes()).hexdigest(),'pairs':len(rows),'language':'LSFB','vocabulary':vocabulary,'optimizer':'AdamW','lr':2e-5,'weight_decay':.01,'batch_size':4,'max_steps':a.max_steps,'precision':'float32','dropout':'disabled only for train-only memorization diagnostic','gate':'all training sequences exact in raw greedy and vocabulary-constrained generation, with EOS; not generalization','embedding_initialization':'mean of original label subword embeddings, scaled to mean embedding norm','linguistic_review':'pending; source annotation and technical checks only'})
    tok=AutoTokenizer.from_pretrained(BASE,revision=REV,use_fast=False)
    model=AutoModelForSeq2SeqLM.from_pretrained(BASE,revision=REV)
    old_size=len(tok);old_pieces=[tok.encode(u.split(':',1)[-1],add_special_tokens=False) for u in vocabulary]
    names=[f'<lsfb_{i:05d}>' for i in range(len(vocabulary))]
    assert tok.add_tokens([AddedToken(n,normalized=False,special=False) for n in names])==len(names)
    model.resize_token_embeddings(len(tok),mean_resizing=False)
    units={u:tok.convert_tokens_to_ids(n) for u,n in zip(vocabulary,names)};reverse={i:u for u,i in units.items()}
    with torch.no_grad():
        weight=model.get_input_embeddings().weight;norm=weight[:old_size].norm(dim=1).mean()
        for u,pieces in zip(vocabulary,old_pieces):
            valid=[i for i in pieces if i not in [tok.unk_token_id,tok.eos_token_id,tok.pad_token_id]]
            if valid:
                value=weight[valid].mean(0);weight[units[u]].copy_(value*norm/value.norm().clamp_min(1e-8))
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout):module.p=0.
    model.gradient_checkpointing_enable();model.config.use_cache=False;model.to('cuda' if torch.cuda.is_available() else 'cpu')
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5,weight_decay=.01)
    dump(a.output/'unit_vocabulary.json',{'units':[{'unit':u,'token':n,'token_id':units[u]} for u,n in zip(vocabulary,names)],'source_label_policy':source['target_policy']})
    def batch(part):
        x=tok([PREFIX+r['french_text'] for r in part],padding=True,return_tensors='pt',return_token_type_ids=False)
        labels=[[units[u] for u in r['units']]+[tok.eos_token_id] for r in part]
        target=torch.full((len(part),max(map(len,labels))),-100,dtype=torch.long)
        for i,label in enumerate(labels):target[i,:len(label)]=torch.tensor(label)
        assert all(label[-1]==tok.eos_token_id for label in labels)
        return {**{k:v.to(model.device) for k,v in x.items()},'labels':target.to(model.device)}
    logs=[];started=time.monotonic();cap=max(len(r['units']) for r in rows)+5
    def evaluate(step):
        model.eval();predictions=[];raw_correct=constrained_correct=0;total_loss=total_labels=0
        with torch.inference_mode():
            for start in range(0,len(rows),4):
                part=rows[start:start+4];x=batch(part);n=int((x['labels']!=-100).sum());total_loss+=float(model(**x).loss)*n;total_labels+=n;x.pop('labels')
                for constrained in [False,True]:
                    options={'prefix_allowed_tokens_fn':lambda batch_id,sequence:list(reverse)+[tok.eos_token_id]} if constrained else {}
                    output=model.generate(**x,max_new_tokens=cap,num_beams=1,do_sample=False,use_cache=True,**options)
                    for row,ids in zip(part,output.cpu().tolist()):
                        predicted,invalid,stopped=decode_units(ids,reverse,tok.eos_token_id,tok.pad_token_id);correct=predicted==row['units'] and stopped and not invalid
                        if constrained:constrained_correct+=correct
                        else:raw_correct+=correct
                        predictions.append({'id':row['id'],'reference':row['units'],'prediction':predicted,'constrained':constrained,'invalid_token_ids':invalid,'eos':stopped,'exact':correct})
        result={'step':step,'train_loss':total_loss/total_labels,'raw_train_exact':raw_correct/len(rows),'constrained_train_exact':constrained_correct/len(rows),'seconds':time.monotonic()-started};logs.append(result)
        dump(a.output/'history.json',logs);dump(a.output/'gate_predictions.json',predictions);print(json.dumps(result),flush=True)
        return raw_correct==len(rows) and constrained_correct==len(rows)
    passed=evaluate(0);order=list(range(len(rows)));step=0
    while not passed and step<a.max_steps:
        random.shuffle(order)
        for start in range(0,len(order),4):
            step+=1;model.train();optimizer.zero_grad();loss=model(**batch([rows[i] for i in order[start:start+4]])).loss
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss: gate failed')
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
            if step%a.check_every==0 or step==a.max_steps:
                passed=evaluate(step)
                if passed or step==a.max_steps:break
    model.config.use_cache=True;model.save_pretrained(a.output/'model');tok.save_pretrained(a.output/'model')
    dump(a.output/'gate_result.json',{'passed':passed,'steps':step,'raw_train_exact':logs[-1]['raw_train_exact'],'constrained_train_exact':logs[-1]['constrained_train_exact'],'pairs':len(rows),'generalization_score':False,'linguistic_approval':False,'next_step':'audit broader corpus before scaling' if passed else 'debug learning before any larger run'})
    print(json.dumps(json.loads((a.output/'gate_result.json').read_text())),flush=True)
if __name__=='__main__':main()
