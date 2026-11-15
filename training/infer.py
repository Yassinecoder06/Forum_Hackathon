"""Return a candidate gloss sequence with research provenance, never an approval."""
import argparse,json
from pathlib import Path
import torch
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM

def main():
    p=argparse.ArgumentParser();p.add_argument('--text',required=True);p.add_argument('--model',type=Path,required=True);p.add_argument('--device',choices=['cpu','cuda'],default='cpu');a=p.parse_args()
    meta=json.loads((a.model/'research_metadata.json').read_text());tok=AutoTokenizer.from_pretrained(a.model)
    if meta['lora']:
        from peft import PeftModel,PeftConfig
        c=PeftConfig.from_pretrained(a.model)
        base=AutoModelForSeq2SeqLM.from_pretrained(c.base_model_name_or_path,local_files_only=True)
        model=PeftModel.from_pretrained(base,a.model)
    else:model=AutoModelForSeq2SeqLM.from_pretrained(a.model)
    model.to(a.device).eval();torch.set_num_threads(4)
    with torch.no_grad():
        inputs=tok(meta['prefix']+a.text,return_tensors='pt').to(a.device)
        y=model.generate(**inputs,max_new_tokens=128,do_sample=False,num_beams=1)
    print(json.dumps(dict(text=a.text,gloss_sequence=tok.decode(y[0],skip_special_tokens=True).split(),candidate=True,model_base=meta['model_base'],model_revision=meta['model_revision'],run_id=meta['run_id'],checkpoint_sha256=meta['checkpoint_sha256'],dataset_manifest_sha256=meta['dataset_manifest_sha256'],linguistic_review='PENDING_LOCAL_REVIEW',translation_validated=False,render_allowed_for_publication=False),ensure_ascii=False))
if __name__=='__main__':main()
