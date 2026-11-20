"""Batched evaluation of a validation-selected, already saved LSFB model."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
from difflib import SequenceMatcher
from lsfb_poc import PREFIX,distance,dump

def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('data',type=Path);p.add_argument('--batch-size',type=int,default=8);a=p.parse_args()
    config=json.loads((a.run/'run_config.json').read_text());manifest=json.loads((a.data/'split_report.json').read_text())
    for split,item in manifest['splits'].items():
        assert hashlib.sha256((a.data/(split+'.jsonl')).read_bytes()).hexdigest()==item['sha256']
    read=lambda split:[json.loads(l) for l in (a.data/(split+'.jsonl')).read_text().splitlines()]
    train=read('train');rows=read('test');maximum=config.get('max_length',96)
    torch.set_num_threads(2)
    tok=AutoTokenizer.from_pretrained(a.run/'final',use_fast=False)
    model=AutoModelForSeq2SeqLM.from_pretrained(a.run/'final').to('cuda' if torch.cuda.is_available() else 'cpu').eval();model.config.use_cache=True
    predictions=[];baselines=[];start=time.monotonic()
    for begin in range(0,len(rows),a.batch_size):
        part=rows[begin:begin+a.batch_size]
        inputs=tok([PREFIX+r['french_text'] for r in part],padding=True,return_tensors='pt',return_token_type_ids=False).to(model.device)
        assert inputs['input_ids'].shape[1]<=maximum
        with torch.inference_mode():output=model.generate(**inputs,max_new_tokens=maximum,num_beams=4)
        for r,prediction in zip(part,tok.batch_decode(output,skip_special_tokens=True)):
            predictions.append({'french':r['french_text'],'reference':r['gloss_sequence'],'prediction':prediction,'recording_id':r['recording_id']})
            baselines.append(max(train,key=lambda t:SequenceMatcher(None,t['french_text'].casefold(),r['french_text'].casefold()).ratio())['gloss_sequence'])
        dump(a.run/'predictions.partial.json',predictions)
        print(f'Evaluated {len(predictions)}/{len(rows)} in {time.monotonic()-start:.1f}s',flush=True)
    refs=[r['gloss_sequence'] for r in rows]
    def score(pred):return {'exact_match':sum(a==b for a,b in zip(refs,pred))/len(refs),'wer':sum(distance(a.split(),b.split()) for a,b in zip(refs,pred))/sum(len(a.split()) for a in refs)}
    metrics={'model':score([r['prediction'] for r in predictions]),'train_only_retrieval':score(baselines),'examples':len(rows),'official_benchmark':False,'linguistic_validation':False,'decoding':{'batch_size':a.batch_size,'num_beams':4,'max_new_tokens':maximum,'precision':'float32','seconds':time.monotonic()-start}}
    dump(a.run/'predictions.json',predictions);dump(a.run/'metrics.json',metrics)
    proof=json.loads((a.run/'DRIVE_VERIFIED.json').read_text()) if (a.run/'DRIVE_VERIFIED.json').exists() else {}
    dump(a.run/'COMPLETED.json',{'steps':config['steps'],'best_checkpoint':proof.get('best_checkpoint'),'final_model':'final','target_language':'Belgian LSFB; not France LSF','evaluation':'batched saved-checkpoint decoding'})
    print(json.dumps(metrics),flush=True)
if __name__=='__main__':main()
