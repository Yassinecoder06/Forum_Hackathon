"""Use the atomic LSFB checkpoint; never interpret ordinary T5 tokens as glosses."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from metrics import unpack

def main():
    p=argparse.ArgumentParser();p.add_argument('model',type=Path);p.add_argument('text');p.add_argument('--raw',action='store_true');a=p.parse_args()
    torch.set_num_threads(2);vocab=json.loads((a.model/'unit_vocabulary.json').read_text());reverse={int(i):u for u,i in vocab['unit_to_id'].items()}
    tok=AutoTokenizer.from_pretrained(a.model,use_fast=False);model=AutoModelForSeq2SeqLM.from_pretrained(a.model).to('cuda' if torch.cuda.is_available() else 'cpu').eval()
    inputs=tok('translate French to LSFB glosses: '+a.text,return_tensors='pt',return_token_type_ids=False).to(model.device)
    if inputs['input_ids'].shape[1]>256:raise ValueError('Input exceeds the registered source-token limit')
    options={} if a.raw else {'prefix_allowed_tokens_fn':lambda batch_id,sequence:list(reverse)+[tok.eos_token_id]}
    with torch.inference_mode():output=model.generate(**inputs,num_beams=1,max_new_tokens=128,use_cache=True,**options)[0].cpu().tolist()
    units,invalid,stopped=unpack(output,reverse,tok.eos_token_id,tok.pad_token_id)
    print(json.dumps({'input':a.text,'language':'LSFB','units':units,'constrained':not a.raw,'invalid_tokens':invalid,'eos':stopped,'translation_validated':False},ensure_ascii=False,indent=2))
if __name__=='__main__':main()
