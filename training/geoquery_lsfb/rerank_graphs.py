"""Training-graph reranking diagnostic. Cannot generate novel graph structures."""
import json,collections,math,time,random
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from peft import PeftModel
from .audit_baseline import features,score

def main():
 out=Path('.runtime/geoquery-rerank-v1');out.mkdir(exist_ok=False);start=time.time();torch.set_num_threads(2);root=Path('.runtime/geoquery-graph-codec-v1');rows={s:[json.loads(x) for x in (root/f'{s}.jsonl').read_text().splitlines()] for s in ['train','val']};tr,va=rows['train'],rows['val'];fs=[features(r['french']) for r in tr];vocab=sorted({t for f in fs for t in f});idf=np.array([math.log((1+len(tr))/(1+sum(t in f for f in fs)))+1 for t in vocab])
 def vec(fs):
  x=np.array([[f[t] for t in vocab] for f in fs],float)*idf;return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
 sims=vec([features(r['french']) for r in va])@vec(fs).T;top=np.argsort(-sims,axis=1,kind='stable')[:,:8]
 run=Path('.runtime/geoquery-full-graph-v1');tok=AutoTokenizer.from_pretrained(run/'tokenizer',local_files_only=True);base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',local_files_only=True,torch_dtype=torch.bfloat16);model=PeftModel.from_pretrained(base,run/'adapter').cuda().eval();selected=[];shuffled=[];details=[]
 with torch.inference_mode():
  for i,row in enumerate(va):
   candidates=[tr[j] for j in top[i]];losses=[];shuffle_losses=[]
   for source,losslist in [(row['input'],losses),(va[(i+1)%len(va)]['input'],shuffle_losses)]:
    for c in candidates:
     x=tok(source,return_tensors='pt').to('cuda');y=tok(c['target_text'],return_tensors='pt')['input_ids'].cuda();losslist.append(float(model(**x,labels=y,use_cache=False).loss))
   a,b=int(np.argmin(losses)),int(np.argmin(shuffle_losses));selected.append(candidates[a]['lsfb_graph']);shuffled.append(candidates[b]['lsfb_graph']);details.append({'id':row['id'],'candidate_ids':[c['id'] for c in candidates],'losses':losses,'cyclic_shuffled_source_losses':shuffle_losses,'selected_id':candidates[a]['id'],'shuffled_selected_id':candidates[b]['id']});(out/'progress.json').write_text(json.dumps({'evaluated':i+1,'total':len(va)}));print('reranked',i+1,flush=True)
 refs=[r['lsfb_graph'] for r in va];baseline=[tr[j[0]]['lsfb_graph'] for j in top]
 result={'task':'closed training-graph candidate reranking;cannot compose novel structures','candidate_count':8,'selection':'lowest mean teacher-forced token loss, frozen selected epoch1model','examples':37,'retrieval':score(refs,baseline),'reranked':score(refs,selected),'cyclic_shuffled_input_same_candidates':score(refs,shuffled),'oracle_candidate_triple_f1':sum(max(score([r['lsfb_graph']],[tr[j]['lsfb_graph']])['triple_micro_f1'] for j in ids) for r,ids in zip(va,top))/len(va),'oracle_note':'mean per-example best candidate F1;uses references only for this diagnostic ceiling,not selection','seconds':time.time()-start,'human_review':False,'split':'previously inspected structure-heldout development;not independent test'}
 (out/'RESULT.json').write_text(json.dumps(result,indent=2));(out/'predictions.json').write_text(json.dumps(details,indent=2));Path('reports/GEOQUERY_RERANK_RESULT.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
