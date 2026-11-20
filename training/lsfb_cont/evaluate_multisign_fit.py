"""One developmental evaluation after a train-only multi-sign fit gate is frozen."""
import json,hashlib
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from peft import PeftModel
from .train_narrow_context import measure

def main():
 root=Path('.runtime/lsfb-multisign-fit-v1');dest=root/'DEVELOPMENT_RESULT.json'
 if dest.exists():raise ValueError('Development result already exists; do not silently retune/re-evaluate')
 gate=json.loads((root/'RESULT.json').read_text());assert gate['status']=='fit_gate_passed'
 rows=[json.loads(x) for x in Path('.runtime/lsfb-narrow-context-data-v1/val.jsonl').read_text().splitlines()]
 tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True);base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',local_files_only=True,torch_dtype=torch.bfloat16);model=PeftModel.from_pretrained(base,root/'adapter').cuda().eval();texts=[];caps=0
 with torch.inference_mode():
  for i in range(0,len(rows),4):
   x=tok(['Planifier les signes lexicaux en LSFB.\nTexte actuel: '+r['current_french'] for r in rows[i:i+4]],padding=True,return_tensors='pt').to('cuda');p=model.generate(**x,max_new_tokens=192,use_cache=True);texts+=tok.batch_decode(p,skip_special_tokens=True);caps+=sum(tok.eos_token_id not in y.tolist() for y in p)
 score=measure(rows,texts);result={'task':'development-only post-gate evaluation; no checkpoint selection','checkpoint_selection':'train-only >=95%exact gate, first pass at100steps','metrics':score,'cap_hits':caps,'source_adapter_sha256':hashlib.sha256((root/'adapter/adapter_model.safetensors').read_bytes()).hexdigest(),'linguistically_validated':False}
 dest.write_text(json.dumps(result,indent=2));(root/'development_predictions.json').write_text(json.dumps([{'id':r['id'],'reference':r['target'],'prediction':p} for r,p in zip(rows,texts)],ensure_ascii=False,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
