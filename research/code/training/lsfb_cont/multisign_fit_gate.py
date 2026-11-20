"""Train-only fit diagnostic: no development/test labels opened or selected."""
import json,hashlib,random,time
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,set_seed
from peft import PeftModel
from .train_narrow_context import measure

def main():
 out=Path('.runtime/lsfb-multisign-fit-v1');out.mkdir(exist_ok=False);start=time.time();set_seed(91);torch.set_num_threads(2)
 source=Path('.runtime/lsfb-narrow-context-data-v1/train.jsonl');rows=[json.loads(x) for x in source.read_text().splitlines()];rows=[r for r in rows if len(r['target'])>1];assert len(rows)==49
 for r in rows:r['input']='Planifier les signes lexicaux en LSFB.\nTexte actuel: '+r['current_french']
 assert len({r['input'] for r in rows})==len(rows)
 save=lambda name,obj:(out/name).write_text(json.dumps(obj,ensure_ascii=False,indent=2))
 tok=AutoTokenizer.from_pretrained('.runtime/lsfb-narrow-current-run-v1/tokenizer',local_files_only=True)
 base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',local_files_only=True,torch_dtype=torch.bfloat16)
 model=PeftModel.from_pretrained(base,'.runtime/lsfb-narrow-current-run-v1/best_adapter',is_trainable=True).cuda();model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 # Remove adapter dropout for memorization diagnosis, explicitly different from generalization training.
 for m in model.modules():
  if isinstance(m,torch.nn.Dropout):m.p=0.
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=0.)
 save('MANIFEST.json',{'task':'train-only multi-sign memorization; no validation/test opened','rows':49,'seed':91,'train_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'warm_start':'.runtime/lsfb-narrow-current-run-v1/best_adapter','threshold_exact':.95,'max_steps':600,'lr':3e-4,'dropout':0.,'linguistically_validated':False})
 history=[]
 for step in range(601):
  if step%100==0:
   model.eval();texts=[];caps=0
   with torch.inference_mode():
    for i in range(0,len(rows),4):
     x=tok([r['input'] for r in rows[i:i+4]],padding=True,return_tensors='pt').to('cuda');p=model.generate(**x,max_new_tokens=64,use_cache=True);texts+=tok.batch_decode(p,skip_special_tokens=True);caps+=sum(tok.eos_token_id not in z.tolist() for z in p)
   score=measure(rows,texts);record={'step':step,'seconds':time.time()-start,'train_metrics':score,'cap_hits':caps};history.append(record);save('history.json',history);save('progress.json',record);save('train_predictions.json',[{'id':r['id'],'reference':r['target'],'prediction':p} for r,p in zip(rows,texts)]);print(json.dumps(record),flush=True)
   if score['exact']>=.95 and caps==0:
    model.save_pretrained(out/'adapter');tok.save_pretrained(out/'tokenizer');save('RESULT.json',{'status':'fit_gate_passed','last':record,'generalization_evaluated':False});return
   if step==600:save('RESULT.json',{'status':'fit_gate_failed','last':record,'generalization_evaluated':False});return
  model.train();opt.zero_grad(set_to_none=True);batch=random.sample(rows,8)
  for j in range(0,8,2):
   rr=batch[j:j+2];x=tok([r['input'] for r in rr],padding=True,return_tensors='pt').to('cuda');y=tok([r['target_text'] for r in rr],padding=True,return_tensors='pt')['input_ids'].cuda();y[y==tok.pad_token_id]=-100;loss=model(**x,labels=y,use_cache=False).loss
   if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
   (loss/4).backward()
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
if __name__=='__main__':main()
