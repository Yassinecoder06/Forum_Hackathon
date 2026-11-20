"""French→FSL size comparison using genuine published FORM labels, feasibility only."""
import argparse,json,time,random,hashlib,math
from pathlib import Path
import torch,numpy as np
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,set_seed
from peft import get_peft_model,LoraConfig,TaskType,PeftModel
from training.lsfb_cont.scoring import distance
import sacrebleu

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--name',required=True);p.add_argument('--epochs',type=int,default=40);a=p.parse_args();out=Path('reports/private')/a.name;out.mkdir(exist_ok=False,parents=True);set_seed(71);torch.set_num_threads(2);beg=time.time()
 save=lambda n,v:(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
 rows={s:[json.loads(x) for x in Path(f'data/parallel/fsl_ud_feasibility_{s}.jsonl').read_text().splitlines()] for s in ['train','val']}
 assert all(r['sign_language']=='fsl' for rs in rows.values() for r in rs)
 assert not {r['split_group'] for r in rows['train']}&{r['split_group'] for r in rows['val']}
 if a.model=='.runtime/mt5-base':
  expected=Path('.runtime/mt5-base-weight.sha256').read_text().strip();assert len(expected)==64 and hashlib.sha256((Path(a.model)/'pytorch_model.bin').read_bytes()).hexdigest()==expected,'Base weight checksum mismatch'
 tok=AutoTokenizer.from_pretrained(a.model,local_files_only=True,use_fast=False);prefix='translate French to France-LSF glosses: '
 missing_chars=[c for c in sorted(set(''.join(' '.join(r['gloss_sequence']) for r in rows['train']))) if tok.unk_token_id in tok(c)['input_ids']]
 if missing_chars:tok.add_tokens(missing_chars)
 def encode(r):
  x=tok(prefix+r['french_text']);y=tok(' '.join(r['gloss_sequence']))['input_ids']
  assert len(x['input_ids'])<=256 and len(y)<=256 
  return x,y
 encoded=[encode(r) for r in rows['train']];assert all(tok.unk_token_id not in y for x,y in encoded);val_encoded=[encode(r) for r in rows['val']]
 dtype=torch.bfloat16;model=AutoModelForSeq2SeqLM.from_pretrained(a.model,local_files_only=True,torch_dtype=dtype)
 if missing_chars:model.resize_token_embeddings(len(tok))
 model=get_peft_model(model,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=16,lora_alpha=32,lora_dropout=.05,modules_to_save=['shared','lm_head'] if missing_chars else None,target_modules=['q','k','v','o','wi','wi_0','wi_1','wo'])).cuda();model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 opt=torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],lr=3e-4,weight_decay=.01)
 cfg={'model_path':a.model,'added_training_characters':missing_chars,'validation_unknown_subtokens':sum(y.count(tok.unk_token_id) for x,y in val_encoded),'language':'fsl','prefix':prefix,'seed':71,'train':len(rows['train']),'val':len(rows['val']),'epochs':a.epochs,'mode':'published_expert_FORM_plus_free_translation; pending_local_review','precision':'bfloat16','lora_rank':16,'selection':'validation_token_loss','official_test_opened':False,'data_sha256':{s:hashlib.sha256(Path(f'data/parallel/fsl_ud_feasibility_{s}.jsonl').read_bytes()).hexdigest() for s in rows}}
 save('CONFIG.json',cfg)
 def batch(r):
  xx=tok([prefix+x['french_text'] for x in r],padding=True,return_tensors='pt').to('cuda');yy=tok([' '.join(x['gloss_sequence']) for x in r],padding=True,return_tensors='pt')['input_ids'].cuda();yy[yy==tok.pad_token_id]=-100;return xx,yy
 def validation():
  model.eval();loss=nt=0
  with torch.inference_mode():
   for r in rows['val']:
    x,y=batch([r]);n=int((y!=-100).sum());loss+=float(model(**x,labels=y,use_cache=False).loss)*n;nt+=n
  return loss/nt
 best=float('inf');stale=0;history=[];step=0
 for epoch in range(1,a.epochs+1):
  order=list(range(len(rows['train'])));random.shuffle(order);losses=[]
  for k in range(0,len(order),8):
   ids=order[k:k+8];opt.zero_grad(set_to_none=True);model.train()
   for i in ids:
    x,y=batch([rows['train'][i]]);loss=model(**x,labels=y,use_cache=False).loss
    if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
    (loss/len(ids)).backward();losses.append(float(loss.detach()))
   torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step();step+=1
  vl=validation();row={'epoch':epoch,'step':step,'train_loss':float(np.mean(losses)),'validation_loss':vl,'seconds':time.time()-beg};history.append(row);save('history.json',history);save('progress.json',row);print(json.dumps(row),flush=True)
  if vl<best:best=vl;stale=0;model.save_pretrained(out/'adapter');tok.save_pretrained(out/'tokenizer');save('BEST.json',row)
  else:stale+=1
  torch.save({'epoch':epoch,'optimizer':opt.state_dict(),'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},out/'optimizer-last.pt')
  if stale>=6:break
 del model,opt;torch.cuda.empty_cache();base=AutoModelForSeq2SeqLM.from_pretrained(a.model,local_files_only=True,torch_dtype=dtype);
 if missing_chars:base.resize_token_embeddings(len(tok))
 model=PeftModel.from_pretrained(base,out/'adapter').cuda().eval();preds=[];caps=0
 with torch.inference_mode():
  for r in rows['val']:
   x,_=batch([r]);y=model.generate(**x,max_new_tokens=192,num_beams=1,use_cache=True);caps+=int(tok.eos_token_id not in y[0]);preds.append(tok.decode(y[0],skip_special_tokens=True))
 refs=[' '.join(r['gloss_sequence']) for r in rows['val']];vocab={g for r in rows['train'] for g in r['gloss_sequence']};pt=[p.split() for p in preds];rt=[r.split() for r in refs]
 result={'status':'complete','best':json.loads((out/'BEST.json').read_text()),'examples':len(refs),'exact':sum(x==y for x,y in zip(pt,rt))/len(refs),'wer':sum(distance(x,y) for x,y in zip(rt,pt))/sum(map(len,rt)),'bleu':sacrebleu.corpus_bleu(preds,[refs],tokenize='none').score,'chrf':sacrebleu.corpus_chrf(preds,[refs]).score,'unknown_prediction_rate':sum(t not in vocab for p in pt for t in p)/max(1,sum(map(len,pt))),'cap_hits':caps,'seconds':time.time()-beg,'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2,'qualified_review':False}
 save('predictions.json',[{'id':r['id'],'reference':ref,'prediction':pred} for r,ref,pred in zip(rows['val'],refs,preds)]);save('RESULT.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
