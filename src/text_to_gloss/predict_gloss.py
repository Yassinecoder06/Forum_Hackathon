"""Guarded inference for a locally trained, explicitly scoped France-LSF adapter."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from peft import PeftModel

def main():
 p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--text',required=True);a=p.parse_args();cfg=json.loads((a.run/'config.json').read_text());assert cfg['sign_language']=='fsl'
 tok=AutoTokenizer.from_pretrained(a.run/'tokenizer',local_files_only=True);model=AutoModelForSeq2SeqLM.from_pretrained(cfg['local_model'],local_files_only=True);model=PeftModel.from_pretrained(model,a.run/'adapter').eval();x=tok(cfg['prefix']+a.text,return_tensors='pt')
 with torch.inference_mode():y=model.generate(**x,max_new_tokens=cfg['target_limit'])
 print(json.dumps({'candidate':tok.decode(y[0],skip_special_tokens=True),'sign_language':'fsl','review_status':'pending','linguistically_validated':False},ensure_ascii=False))
if __name__=='__main__':main()
