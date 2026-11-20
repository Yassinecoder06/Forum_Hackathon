"""Small, recording/signer-disjoint French → Belgian LSFB experiment."""
import argparse, csv, hashlib, json, math, random, tarfile
from pathlib import Path

PREFIX = 'translate French to LSFB glosses: '
REPO = 'ppoitier/lsfb-cont'

def dump(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')

def extract(archive, output, limit=300, overlap=.6):
    recordings = {}
    with tarfile.open(archive) as tar:
        for member in tar:
            suffix = next((s for s in ['annotations.translation.json', 'annotations.both_hands.json', 'signer.txt'] if member.name.endswith('.'+s)), None)
            if not suffix or not member.isfile(): continue
            if member.size > 20_000_000: raise ValueError('Unexpectedly large annotation')
            key = member.name[:-(len(suffix)+1)]
            value = tar.extractfile(member).read().decode('utf-8')
            recordings.setdefault(key, {})[suffix] = value.strip() if suffix=='signer.txt' else json.loads(value)
    rows=[]; seen=set()
    for recording, data in sorted(recordings.items()):
        translations=data.get('annotations.translation.json',[])
        signs=data.get('annotations.both_hands.json',[])
        if not isinstance(translations,list) or not isinstance(signs,list): raise ValueError('Unrecognized corpus JSON: expected segment lists')
        for t in translations:
            start,end=float(t['start_ms']),float(t['end_ms'])
            if end<=start or not t.get('text','').strip(): continue
            events=set()
            for s in signs:
                a,b=float(s['start_ms']),float(s['end_ms']); gloss=s.get('gloss')
                if b>a and isinstance(gloss,str) and gloss.strip() and max(0,min(end,b)-max(start,a))/(b-a)>=overlap:
                    events.add((a,b,gloss.strip()))
            target=' '.join(g for a,b,g in sorted(events))
            text=' '.join(t['text'].split())
            # Exact repeated French text is removed globally, even when targets differ.
            if not target or text.casefold() in seen: continue
            seen.add(text.casefold())
            rows.append(dict(id=f'{recording}:{start}:{end}',recording_id=recording,signer_id=data.get('signer.txt'),language='LSFB',french_text=text,gloss_sequence=target,source=REPO,annotation_tier='both_hands',review_status='unreviewed_auto_alignment',start_ms=start,end_ms=end))
    # Sample across recordings before capping; never take only the first recording.
    random.Random(42).shuffle(rows); rows=rows[:limit]
    Path(output).parent.mkdir(parents=True,exist_ok=True)
    Path(output).write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
    report=dict(candidates=len(rows),recordings=len({r['recording_id'] for r in rows}),signers=sorted({r['signer_id'] for r in rows if r['signer_id']}),archive_sha256=hashlib.file_digest(open(archive,'rb'),'sha256').hexdigest(),alignment='weak temporal overlap; simultaneous signs flattened',license='CC-BY-NC-SA-4.0')
    dump(str(output)+'.audit.json',report); return report

def split(source, folder, allow_weak=False):
    rows=[json.loads(l) for l in Path(source).read_text().splitlines() if l.strip()]
    rows=[r for r in rows if r['review_status']=='approved' or allow_weak and r['review_status']=='unreviewed_auto_alignment']
    if not rows: raise ValueError('No eligible examples. Review candidates or explicitly enable weak experimental labels.')
    signer_groups={r.get('signer_id') for r in rows}
    group='signer_id' if None not in signer_groups and len(signer_groups)>=3 else 'recording_id'
    groups=sorted({r[group] for r in rows});random.Random(42).shuffle(groups)
    if len(groups)<3: raise ValueError('Fewer than three independent groups: obtain another training shard')
    n=max(1,round(.15*len(groups))); selected={'test':set(groups[:n]),'val':set(groups[n:2*n]),'train':set(groups[2*n:])}
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True);report={'grouping':group,'weak_supervision':allow_weak,'evaluation_scope':'internal held-out subset of official training shard, not official benchmark','splits':{}}
    for name, ids in selected.items():
        part=[r for r in rows if r[group] in ids]
        if not part: raise ValueError('Empty split')
        path=folder/(name+'.jsonl');path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in part))
        report['splits'][name]={'examples':len(part),'groups':sorted(ids),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    dump(folder/'split_report.json',report);return report

def distance(a,b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        cur=[i]
        for j,y in enumerate(b,1):cur.append(min(cur[-1]+1,prev[j]+1,prev[j-1]+(x!=y)))
        prev=cur
    return prev[-1]

def train(data, output, steps=30, base='google/mt5-small', revision=None, max_length=96, eval_every=0):
    import torch
    from torch.utils.data import Dataset
    from transformers import AutoTokenizer,AutoModelForSeq2SeqLM,DataCollatorForSeq2Seq,Seq2SeqTrainer,Seq2SeqTrainingArguments,TrainerCallback,set_seed
    set_seed(42);data=Path(data);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    report=json.loads((data/'split_report.json').read_text())
    for name, entry in report['splits'].items():
        if hashlib.sha256((data/(name+'.jsonl')).read_bytes()).hexdigest()!=entry['sha256']:raise ValueError('Split changed since freezing')
    fingerprint=hashlib.sha256((data/'split_report.json').read_bytes()).hexdigest()
    manifest={'base':base,'revision':revision,'split_sha256':fingerprint,'steps':steps,'language':'LSFB','weak_supervision':report['weak_supervision'],'precision':'bf16 when supported; otherwise fp32, never fp16'}
    if max_length!=96:manifest['max_length']=max_length
    if eval_every:manifest['eval_every']=eval_every
    plan=output/'run_config.json'
    if plan.exists() and json.loads(plan.read_text())!=manifest:raise ValueError('Different configuration: use a new Drive run folder')
    dump(plan,manifest)
    if (output/'COMPLETED.json').is_file() and all((output/name).is_file() for name in ['final/model.safetensors','final/config.json','final/tokenizer_config.json','metrics.json']):
        print('Run already complete; preserving the saved model and results.',flush=True)
        return
    tok=AutoTokenizer.from_pretrained(base,revision=revision,use_fast=False)
    model=AutoModelForSeq2SeqLM.from_pretrained(base,revision=revision);model.config.use_cache=False
    class Pairs(Dataset):
        def __init__(self,name):
            self.rows=[json.loads(l) for l in (data/(name+'.jsonl')).read_text().splitlines()]
            self.items=[]
            for r in self.rows:
                item=tok(PREFIX+r['french_text'],return_token_type_ids=False);item['labels']=tok(text_target=r['gloss_sequence'])['input_ids']
                if max(len(item['input_ids']),len(item['labels']))>max_length: raise ValueError('Overlength example: review or segment it; no silent target truncation')
                self.items.append(item)
        def __len__(self): return len(self.items)
        def __getitem__(self,i): return self.items[i]
    class Finite(TrainerCallback):
        def on_log(self,args,state,control,logs=None,**kwargs):
            if any(isinstance(v,(float,int)) and not math.isfinite(v) for v in (logs or {}).values()): raise ValueError('Nonfinite training log')
    interval=eval_every or max(1,steps//2)
    args=Seq2SeqTrainingArguments(output_dir=str(output/'checkpoints'),max_steps=steps,per_device_train_batch_size=1,per_device_eval_batch_size=1,gradient_accumulation_steps=4,learning_rate=5e-5,optim='adafactor',gradient_checkpointing=True,fp16=False,bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),eval_strategy='steps',eval_steps=interval,save_steps=interval,save_total_limit=2,load_best_model_at_end=True,metric_for_best_model='eval_loss',logging_steps=1,logging_nan_inf_filter=False,report_to='none',seed=42)
    trainer=Seq2SeqTrainer(model=model,args=args,train_dataset=Pairs('train'),eval_dataset=Pairs('val'),data_collator=DataCollatorForSeq2Seq(tok,model=model),processing_class=tok,callbacks=[Finite()])
    # A disconnect can leave a half-written directory on Drive. Resume only complete saves.
    checkpoints=[]
    for checkpoint in (output/'checkpoints').glob('checkpoint-*'):
        if all((checkpoint/name).is_file() for name in ['trainer_state.json','optimizer.pt','scheduler.pt']) and ((checkpoint/'model.safetensors').is_file() or (checkpoint/'model.safetensors.index.json').is_file()):
            checkpoints.append(checkpoint)
    latest=str(max(checkpoints,key=lambda p:int(p.name.split('-')[-1]))) if checkpoints else None
    trainer.train(resume_from_checkpoint=latest)
    trainer.save_model(str(output/'final'));tok.save_pretrained(output/'final');dump(output/'history.json',trainer.state.log_history)
    # Test is decoded only after validation checkpoint selection.
    from difflib import SequenceMatcher
    model.eval();model.config.use_cache=True;rows=Pairs('test').rows;training=Pairs('train').rows;predictions=[];refs=[];baseline=[]
    torch.set_num_threads(2)
    for begin in range(0,len(rows),8):
        part=rows[begin:begin+8]
        with torch.inference_mode():
            x=tok([PREFIX+r['french_text'] for r in part],padding=True,return_tensors='pt',return_token_type_ids=False).to(model.device)
            outputs=model.generate(**x,max_new_tokens=max_length,num_beams=4)
        for r,prediction in zip(part,tok.batch_decode(outputs,skip_special_tokens=True)):
            predictions.append({'french':r['french_text'],'reference':r['gloss_sequence'],'prediction':prediction,'recording_id':r['recording_id']});refs.append(r['gloss_sequence'])
            baseline.append(max(training,key=lambda t:SequenceMatcher(None,t['french_text'].casefold(),r['french_text'].casefold()).ratio())['gloss_sequence'])
        dump(output/'predictions.partial.json',predictions)
        print(f'Evaluated {len(predictions)}/{len(rows)} held-out examples',flush=True)
    def score(pred):return {'exact_match':sum(a==b for a,b in zip(refs,pred))/len(refs),'wer':sum(distance(a.split(),b.split()) for a,b in zip(refs,pred))/sum(len(a.split()) for a in refs)}
    dump(output/'predictions.json',predictions);dump(output/'metrics.json',{'model':score([r['prediction'] for r in predictions]),'train_only_retrieval':score(baseline),'examples':len(rows),'official_benchmark':False,'linguistic_validation':False})
    dump(output/'COMPLETED.json',{'steps':trainer.state.global_step,'best_checkpoint':trainer.state.best_model_checkpoint,'final_model':'final','target_language':'Belgian LSFB; not France LSF'})

if __name__=='__main__':
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='cmd',required=True)
    a=sub.add_parser('extract');a.add_argument('archive');a.add_argument('output');a.add_argument('--limit',type=int,default=300)
    a=sub.add_parser('split');a.add_argument('input');a.add_argument('output');a.add_argument('--allow-weak',action='store_true')
    a=sub.add_parser('train');a.add_argument('data');a.add_argument('output');a.add_argument('--steps',type=int,default=30);a.add_argument('--base',default='google/mt5-small');a.add_argument('--revision');a.add_argument('--max-length',type=int,default=96);a.add_argument('--eval-every',type=int,default=0)
    a=p.parse_args()
    if a.cmd=='extract':print(extract(a.archive,a.output,a.limit))
    elif a.cmd=='split':print(split(a.input,a.output,a.allow_weak))
    else:train(a.data,a.output,a.steps,a.base,a.revision,a.max_length,a.eval_every)
