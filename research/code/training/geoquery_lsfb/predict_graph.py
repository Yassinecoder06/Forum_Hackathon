"""Research candidate inference preserving graph structure and explicit failure."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from peft import PeftModel
from .codec import GraphCodec

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--text',required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args();torch.set_num_threads(2)
 manifest=json.loads((a.run/'MANIFEST.json').read_text());codec=GraphCodec(json.loads((a.run/'hns_dictionary.json').read_text()));tok=AutoTokenizer.from_pretrained(a.run/'tokenizer',local_files_only=True);dtype=torch.bfloat16 if a.device=='cuda' else torch.float32
 base=AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision=manifest.get('model_revision',manifest.get('warmstart_gate_manifest',{}).get('model_revision')),local_files_only=True,torch_dtype=dtype);model=PeftModel.from_pretrained(base,a.run/'adapter').to(a.device).eval();x=tok('French to LSFB temporal graph: '+a.text,return_tensors='pt').to(a.device)
 if x['input_ids'].shape[1]>256:raise ValueError('Research input budget exceeded')
 with torch.inference_mode():pred=model.generate(**x,max_new_tokens=manifest['generation_cap'],use_cache=True)
 text=tok.decode(pred[0],skip_special_tokens=True);graph=None;error=None
 try:graph=codec.decode(text)
 except ValueError as e:error=str(e)
 print(json.dumps({'sign_language':'LSFB','task':'graph candidate;not France-LSFavatar motion','status':'structurally_valid_candidate' if graph else 'invalid_candidate','graph':graph,'raw_text':text,'error':error,'review_status':'pending','linguistically_validated':False,'model_scope':manifest.get('task','unvalidated research'),'cap_hit':tok.eos_token_id not in pred[0].tolist()},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
