"""Compare free generation to a training-phrase trie; the latter cannot compose new phrases."""
import json,argparse,torch
from pathlib import Path
from transformers import AutoTokenizer,AutoModelForSeq2SeqLM
from peft import PeftModel
from .experiments import metrics,normalize

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',default='lsfb-bounded-v1');args=ap.parse_args()
    root=Path('.runtime')/args.run
    data=json.load(open('.runtime/lsfb-bounded-v1/DATA.json'));rows=data['val'];vocab=set(data['vocabulary'])
    seen={normalize(r['input']) for r in data['train']};novel=[i for i,r in enumerate(rows) if normalize(r['input']) not in seen]
    if (root/'MANIFEST.json').exists():
        retained=set(json.load(open(root/'MANIFEST.json'))['train_ids'])
        data['train']=[r for r in data['train'] if r['id'] in retained]
    tok=AutoTokenizer.from_pretrained(root/'tokenizer',local_files_only=True)
    path=root/('best_adapter' if (root/'best_adapter').exists() else 'final_adapter')
    model=PeftModel.from_pretrained(AutoModelForSeq2SeqLM.from_pretrained('google-t5/t5-base',revision='a9723ea7f1b39c1eae772870f3b547bf6ef7e6c1',torch_dtype=torch.bfloat16,local_files_only=True),path).cuda().eval()
    torch.set_num_threads(2);trie={}
    # Candidate phrases come only from training, never validation references.
    for target in {r['target_text'] for r in data['train']}:
        node=trie
        for t in [model.config.decoder_start_token_id]+tok.encode(target):node=node.setdefault(t,{})
    def allowed(batch,prefix):
        node=trie
        for t in prefix.tolist():node=node.get(t,{})
        return list(node) or [tok.eos_token_id]
    results={}
    for name,beam,constrained,shuffle in [('greedy',1,False,False),('beam4',4,False,False),('training_phrase_trie_beam4',4,True,False),('trie_shuffled_source',4,True,True)]:
        inputs=[r['input'] for r in rows]
        if shuffle:inputs=inputs[1:]+inputs[:1]
        texts=[]
        for i in range(0,len(rows),4):
            enc=tok(inputs[i:i+4],padding=True,return_tensors='pt').to('cuda')
            opts={'prefix_allowed_tokens_fn':allowed} if constrained else {}
            with torch.inference_mode():out=model.generate(**enc,max_new_tokens=128,num_beams=beam,do_sample=False,use_cache=True,**opts)
            texts+=tok.batch_decode(out,skip_special_tokens=True)
        results[name]={'all':metrics(rows,texts,vocab),'novel':metrics([rows[i] for i in novel],[texts[i] for i in novel],vocab),
                       'predictions':[dict(id=r['id'],input=inp,prediction=p,reference=r['target']) for r,inp,p in zip(rows,inputs,texts)]}
        (root/'DECODING_PROBE.json').write_text(json.dumps(results,ensure_ascii=False,indent=2))
        print(name,json.dumps({k:v for k,v in results[name].items() if k!='predictions'}),flush=True)

if __name__=='__main__':main()
