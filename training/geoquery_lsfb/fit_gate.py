"""Full-graph train-only memorization gate; no heldout inputs/targets are read."""
import json,hashlib,time,random,argparse,shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,set_seed
from peft import LoraConfig,TaskType,get_peft_model
from .codec import GraphCodec

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=Path('.runtime/geoquery-graph-fit-v1'));ap.add_argument('--steps',type=int,default=300);args=ap.parse_args()
 root=Path('.runtime/geoquery-graph-codec-v1');out=args.output;out.mkdir(exist_ok=False);set_seed(92);torch.set_num_threads(2);start=time.time();save=lambda n,v:(out/n).write_text(json.dumps(v,ensure_ascii=False,indent=2))
 allrows=[json.loads(x) for x in (root/'train.jsonl').read_text().splitlines()];rows=random.Random(92).sample(allrows,8);assert all(r['split']=='train' and r['language']=='LSFB' for r in rows)
 codec=GraphCodec(json.loads((root/'hns_dictionary.json').read_text()));tok=AutoTokenizer.from_pretrained('.runtime/lsfb-cont-local-v3/tokenizer',local_files_only=True);length=max(r['target_tokens'] for r in rows);cap=length+32
 base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',local_files_only=True,torch_dtype=torch.bfloat16);model=get_peft_model(base,LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,r=16,lora_alpha=32,lora_dropout=0.,target_modules=['q','k','v','o','wi','wo'])).cuda();model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=5e-4,weight_decay=0.)
 save('MANIFEST.json',{'task':'train-only full LSFB graph fit gate; no heldout rows read','seed':92,'rows':[r['id'] for r in rows],'train_sha256':hashlib.sha256((root/'train.jsonl').read_bytes()).hexdigest(),'model_revision':'a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1','rank':16,'lr':5e-4,'precision':'bfloat16','max_steps':args.steps,'checkpoint_gate':'at least7/8exact graphs and all8structurally valid,no cap hits','generation_cap':cap,'target_max':length,'input_provenance':'DeepL French','local_approval':False})
 history=[]
 def evaluate(step):
  model.eval();texts=[];caps=0
  with torch.inference_mode():
   for i in range(0,8,2):
    x=tok([r['input'] for r in rows[i:i+2]],padding=True,return_tensors='pt').to('cuda');p=model.generate(**x,max_new_tokens=cap,use_cache=True);texts+=tok.batch_decode(p,skip_special_tokens=True);caps+=sum(tok.eos_token_id not in y.tolist() for y in p)
  exact=valid=0;predictions=[]
  for row,text in zip(rows,texts):
   try:graph=codec.decode(text);valid+=1;equal=graph==row['lsfb_graph'];exact+=equal
   except ValueError:graph=None;equal=False
   predictions.append({'id':row['id'],'reference':row['lsfb_graph'],'prediction_text':text,'prediction_graph':graph,'exact':equal})
  record={'step':step,'train_examples':8,'exact_graphs':exact,'valid_graphs':valid,'cap_hits':caps,'seconds':time.time()-start};history.append(record);save('history.json',history);save('progress.json',record);save('predictions.json',predictions);print(json.dumps(record),flush=True);return record
 for step in range(args.steps+1):
  if (step%50==0 or step==args.steps) and step>0:
   record=evaluate(step)
   if record['exact_graphs']>=7 and record['valid_graphs']==8 and record['cap_hits']==0:
    model.save_pretrained(out/'adapter');tok.save_pretrained(out/'tokenizer');shutil.copy2(root/'hns_dictionary.json',out/'hns_dictionary.json');save('RESULT.json',{'status':'fit_gate_passed','result':record,'heldout_evaluated':False});return
   if step==args.steps:
    model.save_pretrained(out/'failed_adapter');tok.save_pretrained(out/'tokenizer');shutil.copy2(root/'hns_dictionary.json',out/'hns_dictionary.json');save('RESULT.json',{'status':'fit_gate_failed','result':record,'heldout_evaluated':False});return
  model.train();opt.zero_grad(set_to_none=True);batch=random.sample(rows,4)
  for row in batch:
   x=tok(row['input'],return_tensors='pt').to('cuda');y=tok(row['target_text'],return_tensors='pt')['input_ids'].cuda();loss=model(**x,labels=y,use_cache=False).loss
   if not torch.isfinite(loss):raise ValueError('Nonfinite training loss')
   (loss/4).backward()
  torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True);opt.step()
  if step%10==0:save('progress.json',{'phase':'training','step':step,'loss':float(loss.detach()),'seconds':time.time()-start});print('train',step,float(loss.detach()),flush=True)
if __name__=='__main__':main()
