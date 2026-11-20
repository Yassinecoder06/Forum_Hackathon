"""Full-graph feasibility run after a passed train-only gate. No synthetic expansion."""
import argparse,json,hashlib,time,math,random,collections
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,set_seed
from peft import PeftModel
from .codec import GraphCodec
from .audit_baseline import retrieve,score

def graph_metrics(rows,predictions):
 if len(rows)!=len(predictions):raise ValueError('Missing predictions')
 exact=valid=tp=fp=fn=0
 for row,p in zip(rows,predictions):
  a={tuple(t) for t in row['lsfb_graph']};b={tuple(t) for t in p} if p is not None else set();valid+=p is not None;exact+=p==row['lsfb_graph'];tp+=len(a&b);fp+=len(b-a);fn+=len(a-b)
 return {'examples':len(rows),'exact_graphs':exact,'graph_exact':exact/len(rows),'valid_graphs':valid,'invalid_graphs':len(rows)-valid,'triple_micro_f1':2*tp/max(1,2*tp+fp+fn),'invalid_outputs':'score as empty predicted triple set; kept in denominator and logs'}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--gate',type=Path,default=Path('.runtime/geoquery-graph-fit-v2'));ap.add_argument('--output',type=Path,default=Path('.runtime/geoquery-full-graph-v1'));ap.add_argument('--epochs',type=int,default=8);ap.add_argument('--resume',action='store_true');args=ap.parse_args()
 gate=json.loads((args.gate/'RESULT.json').read_text());assert gate['status']=='fit_gate_passed','Do not bypass train-only learnability gate'
 root=Path('.runtime/geoquery-graph-codec-v1');audit=json.loads((root/'AUDIT.json').read_text());rows={}
 for name,digest in audit['files'].items():assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
 for s in ['train','val']:
  path=root/f'{s}.jsonl';assert hashlib.sha256(path.read_bytes()).hexdigest()==audit['files'][path.name];rows[s]=[json.loads(x) for x in path.read_text().splitlines()]
 assert not {r['template_type'] for r in rows['train']}&{r['template_type'] for r in rows['val']}
 assert not {r['id'] for r in rows['train']}&{r['id'] for r in rows['val']}
 assert all(r['language']=='LSFB' for rr in rows.values() for r in rr)
 out=args.output
 if args.resume and (out/'RESULT.json').exists():raise ValueError('Completed run cannot be resumed; choose a new run')
 out.mkdir(exist_ok=args.resume);start=time.time();set_seed(93);torch.set_num_threads(2)
 def save(n,v):
  p=out/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2));t.replace(p)
 revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1';tok=AutoTokenizer.from_pretrained(args.gate/'tokenizer',local_files_only=True);codec=GraphCodec(json.loads((root/'hns_dictionary.json').read_text()))
 def batch(rr):
  x=tok([r['input'] for r in rr],padding=True,return_tensors='pt').to('cuda');y=tok([r['target_text'] for r in rr],padding=True,return_tensors='pt')['input_ids'].cuda();assert x['input_ids'].shape[1]<=256 and y.shape[1]<=1280;y[y==tok.pad_token_id]=-100;return x,y
 base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision=revision,local_files_only=True,torch_dtype=torch.bfloat16);model=PeftModel.from_pretrained(base,args.gate/'adapter',is_trainable=True).cuda();model.enable_input_require_grads();model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False});model.config.use_cache=False
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=2e-4,weight_decay=.01);best=float('inf');stale=0;history=[];first=1;steps=0
 manifest={'task':'French machine-translated input to full LSFB graph; feasibility','model_revision':revision,'train':len(rows['train']),'development':len(rows['val']),'initialization':'train-only graph gate adapter; eight rows are contained in training split','seed':93,'epochs_max':args.epochs,'effective_batch':4,'lr_peak':2e-4,'precision':'BF16','max_source_tokens':256,'max_target_tokens':1280,'generation_cap':1280,'selection':'lowest development token-weighted cross entropy; patience3','official_test_opened':False,'source_files':audit['files'],'no_local_linguistic_approval':True,'source_signer':'single native signer per source; no unseen-signer claim','warmstart_gate_manifest':json.loads((args.gate/'MANIFEST.json').read_text())}
 if args.resume:
  old=json.loads((out/'MANIFEST.json').read_text())
  assert old==manifest,'Resume configuration/data differs from original run'
 save('MANIFEST.json',manifest);(out/'hns_dictionary.json').write_text(json.dumps(codec.hns,ensure_ascii=False))
 if args.resume:
  state=torch.load(out/'last-state.pt',map_location='cuda',weights_only=True);model.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);best=state['best'];stale=state['stale'];first=state['epoch']+1;steps=state['steps'];history=state['history'];random.setstate(state['python_rng']);torch.set_rng_state(state['torch_rng'].cpu());torch.cuda.set_rng_state(state['cuda_rng'].cpu())
 for epoch in range(first,args.epochs+1):
  order=list(range(len(rows['train'])));random.shuffle(order);losses=[]
  for k in range(0,len(order),4):
   ids=order[k:k+4];opt.zero_grad(set_to_none=True);model.train()
   for i in ids:
    x,y=batch([rows['train'][i]]);loss=model(**x,labels=y,use_cache=False).loss
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    (loss/len(ids)).backward();losses.append(float(loss.detach()))
   torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1,error_if_nonfinite=True)
   lr=2e-4*min(1,(steps+1)/20)
   for group in opt.param_groups:group['lr']=lr
   opt.step();steps+=1
   if steps%20==0:save('progress.json',{'phase':'training','epoch':epoch,'step':steps,'loss':losses[-1],'seconds':time.time()-start});print('training',epoch,steps,losses[-1],flush=True)
  model.eval();total=nt=0
  with torch.inference_mode():
   for r in rows['val']:
    x,y=batch([r]);n=int((y!=-100).sum());total+=float(model(**x,labels=y,use_cache=False).loss)*n;nt+=n
  vl=total/nt;row={'epoch':epoch,'step':steps,'train_loss':sum(losses)/len(losses),'validation_token_loss':vl,'seconds':time.time()-start};history.append(row);save('history.json',history);save('progress.json',row);print(json.dumps(row),flush=True)
  if vl<best:best=vl;stale=0;model.save_pretrained(out/'adapter');tok.save_pretrained(out/'tokenizer');save('BEST.json',row)
  else:stale+=1
  torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'best':best,'stale':stale,'epoch':epoch,'steps':steps,'history':history,'python_rng':random.getstate(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()},out/'last-state.pt')
  if stale>=3:break
 del model,base,opt;torch.cuda.empty_cache();base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision=revision,local_files_only=True,torch_dtype=torch.bfloat16);model=PeftModel.from_pretrained(base,out/'adapter').cuda().eval();predictions=[];records=[];caps=0
 with torch.inference_mode():
  for i,r in enumerate(rows['val']):
   x,_=batch([r]);p=model.generate(**x,max_new_tokens=1280,use_cache=True);text=tok.decode(p[0],skip_special_tokens=True);cap=tok.eos_token_id not in p[0].tolist();caps+=cap
   try:graph=codec.decode(text);error=None
   except ValueError as e:graph=None;error=str(e)
   predictions.append(graph);records.append({'id':r['id'],'reference':r['lsfb_graph'],'prediction':graph,'raw_prediction':text,'error':error,'cap_hit':cap});save('development_predictions.json',records);save('progress.json',{'phase':'decoding','evaluated':i+1,'total':len(rows['val']),'seconds':time.time()-start});print('decoded',i+1,'valid',graph is not None,flush=True)
 baseline={}
 for name,word in [('char_tfidf',False),('word_tfidf',True)]:
  ids=retrieve(rows['train'],rows['val'],word);baseline[name]=score([r['lsfb_graph'] for r in rows['val']],[rows['train'][i]['lsfb_graph'] for i in ids])
 result={'status':'complete','metrics':graph_metrics(rows['val'],predictions),'cap_hits':caps,'baselines_same_clean_split':baseline,'selected':json.loads((out/'BEST.json').read_text()),'seconds':time.time()-start,'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2,'linguistically_validated':False};save('RESULT.json',result);save('progress.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
