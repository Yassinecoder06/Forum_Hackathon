"""ACL-inspired LSFB pilot. PyTorch adaptation, not a Marian reproduction."""
import argparse, collections, hashlib, json, math, random, shutil, time
from pathlib import Path
from urllib.parse import unquote
import numpy as np
import sentencepiece as spm
import torch
from torch import nn
from sacrebleu.metrics import BLEU, CHRF, TER

PAD,UNK,BOS,EOS=0,1,2,3

def distance(a,b):
    row=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        nxt=[i]
        for j,y in enumerate(b,1):nxt.append(min(nxt[-1]+1,row[j]+1,row[j-1]+(x!=y)))
        row=nxt
    return row[-1]

def metrics(refs,preds,vocab):
    assert len(refs)==len(preds) and refs
    groups={}
    for name,ids in [('all',list(range(len(refs)))),('multi_sign',[i for i,r in enumerate(refs) if len(r)>1]),('one_sign',[i for i,r in enumerate(refs) if len(r)==1]),('seen_vocabulary',[i for i,r in enumerate(refs) if all(t in vocab for t in r)])]:
        if not ids:continue
        groups[name]={'examples':len(ids),'exact':sum(refs[i]==preds[i] for i in ids)/len(ids),'wer':sum(distance(refs[i],preds[i]) for i in ids)/sum(len(refs[i]) for i in ids)}
    groups['unknown_prediction_labels']=sum(x not in vocab for p in preds for x in p)
    groups['predicted_operations']=sum(map(len,preds))
    # Reversible one-token encoding keeps labels with internal whitespace intact for MT metrics.
    from urllib.parse import quote
    texts=lambda rows:[' '.join(quote(t,safe="-'") for t in r) for r in rows]
    hyp,ref=texts(preds),texts(refs)
    for name,m in [('bleu',BLEU(tokenize='none')),('chrf',CHRF()),('ter',TER(normalized=False,no_punct=False,case_sensitive=True))]:
        groups[name]=m.corpus_score(hyp,[ref]).score;groups[name+'_signature']=str(m.get_signature())
    return groups

class Model(nn.Module):
    def __init__(self,vocab,dim=512,dropout=.2):
        super().__init__();self.dim=dim
        self.emb=nn.Embedding(vocab,dim,padding_idx=PAD)
        self.net=nn.Transformer(d_model=dim,nhead=8,num_encoder_layers=1,num_decoder_layers=2,dim_feedforward=dim*4,dropout=dropout,batch_first=True,activation='relu')
        self.out=nn.Linear(dim,vocab,bias=False);self.out.weight=self.emb.weight
        self.drop=nn.Dropout(dropout)
        pos=torch.arange(1024).unsqueeze(1);freq=torch.exp(torch.arange(0,dim,2)*(-math.log(10000)/dim));pe=torch.zeros(1024,dim);pe[:,0::2]=torch.sin(pos*freq);pe[:,1::2]=torch.cos(pos*freq);self.register_buffer('pe',pe)
        for p in self.parameters():
            if p.dim()>1:nn.init.xavier_uniform_(p)
        with torch.no_grad():self.emb.weight[PAD].zero_()
    def embed(self,x):return self.drop(self.emb(x)*math.sqrt(self.dim)+self.pe[:x.shape[1]])
    def encode(self,x):return self.net.encoder(self.embed(x),src_key_padding_mask=x.eq(PAD))
    def decode(self,y,memory,pad):
        mask=torch.triu(torch.ones(y.shape[1],y.shape[1],device=y.device,dtype=torch.bool),1)
        z=self.net.decoder(self.embed(y),memory,tgt_mask=mask,tgt_key_padding_mask=y.eq(PAD),memory_key_padding_mask=pad)
        return self.out(z)
    def forward(self,x,y):return self.decode(y,self.encode(x),x.eq(PAD))

@torch.inference_mode()
def generate(model,x,cap=96):
    model.eval();memory=model.encode(x);y=torch.full((len(x),1),BOS,device=x.device,dtype=torch.long);done=torch.zeros(len(x),device=x.device,dtype=torch.bool)
    for _ in range(cap):
        logits=model.decode(y,memory,x.eq(PAD))[:,-1];logits[:,PAD]=-float('inf');logits[:,BOS]=-float('inf')
        nxt=logits.argmax(-1);nxt=torch.where(done,torch.full_like(nxt,EOS),nxt);y=torch.cat([y,nxt[:,None]],1);done|=nxt.eq(EOS)
        if done.all():break
    return y[:,1:].tolist(),int((~done).sum())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--drive',type=Path,required=True);ap.add_argument('--epochs',type=int,default=40);args=ap.parse_args()
    if args.output.exists():raise ValueError('Refusing existing output')
    assert args.drive.parent.is_dir(),'Drive parent absent'
    args.output.mkdir(parents=True);args.drive.mkdir();out=args.output;started=time.time();torch.set_num_threads(2)
    random.seed(1111);np.random.seed(1111);torch.manual_seed(1111);torch.cuda.manual_seed_all(1111)
    def save(name,obj):
        p=out/name;t=p.with_suffix('.tmp');t.write_text(json.dumps(obj,ensure_ascii=False,indent=2));t.replace(p)
        shutil.copy2(p,args.drive/name)
    def checkpoint(name,model):
        p=out/name;torch.save(model.state_dict(),p);shutil.copy2(p,args.drive/name)
        assert hashlib.sha256(p.read_bytes()).digest()==hashlib.sha256((args.drive/name).read_bytes()).digest()
    manifest=json.loads((args.data/'MANIFEST.json').read_text())
    for name,h in manifest['files'].items():assert hashlib.sha256((args.data/name).read_bytes()).hexdigest()==h
    lines=lambda name:(args.data/name).read_text().splitlines()
    train_src,train_tgt=lines('train.fr'),lines('train.gloss');val_src,val_tgt=lines('val.fr'),lines('val.gloss')
    train_meta=[json.loads(x) for x in lines('train.metadata.jsonl')];val_meta=[json.loads(x) for x in lines('val.metadata.jsonl')]
    for field in ('signer_id','conversation_id'):assert not {x[field] for x in train_meta}&{x[field] for x in val_meta}
    assert all(x['language']=='LSFB' for x in train_meta+val_meta)
    (out/'tokenizer_train.txt').write_text('\n'.join(train_src+train_tgt)+'\n')
    spm.SentencePieceTrainer.train(input=str(out/'tokenizer_train.txt'),model_prefix=str(out/'joint'),vocab_size=2000,model_type='bpe',character_coverage=1.,pad_id=PAD,unk_id=UNK,bos_id=BOS,eos_id=EOS,hard_vocab_limit=False,normalization_rule_name='identity',minloglevel=2)
    tok=spm.SentencePieceProcessor(model_file=str(out/'joint.model'))
    for name in ('joint.model','joint.vocab'):shutil.copy2(out/name,args.drive/name)
    source=lambda s:tok.encode(s)+[EOS]
    target=lambda s:[BOS]+tok.encode(s)+[EOS]
    X=[source(s) for s in train_src];Y=[target(s) for s in train_tgt];VX=[source(s) for s in val_src]
    assert all(tok.decode(tok.encode(s))==s for s in train_tgt)
    assert max(map(len,X))<512 and max(map(len,Y))<256
    vocab=set(t for r in train_meta for t in r['target'])
    cfg={'implementation':'PyTorch adaptation, not exact Marian reproduction','encoder_layers':1,'decoder_layers':2,'dim':512,'heads':8,'ffn':2048,'vocab':len(tok),'seed':1111,'precision':'float32','batch':32,'epochs':args.epochs,'lr':.0003,'dropout':.2,'label_smoothing':.2,'train':len(X),'validation':len(VX),'tokenizer_train_only':True,'validation_unknown_subtokens':sum(tok.encode(s).count(UNK) for s in val_src+val_tgt),'data_manifest':manifest,'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),'generation':'greedy; max96 subtokens','selection':'lowest development WER among cap-free checkpoints; minimum20 epochs; no official test opened'}
    save('CONFIG.json',cfg)
    def pad(rows):return nn.utils.rnn.pad_sequence([torch.tensor(x,device='cuda') for x in rows],batch_first=True,padding_value=PAD)
    def evaluate(model,ids,train=False,phase='validation'):
        inputs=X if train else VX;meta=train_meta if train else val_meta;preds=[];caps=0
        for start in range(0,len(ids),16):
            group=ids[start:start+16];ys,n=generate(model,pad([inputs[i] for i in group]));caps+=n
            for y in ys:
                if EOS in y:y=y[:y.index(EOS)]
                preds.append([unquote(t) for t in tok.decode(y).split()])
        refs=[meta[i]['target'] for i in ids];result=metrics(refs,preds,vocab);result['cap_hits']=caps
        return result,[{'id':meta[i]['id'],'reference':meta[i]['target'],'prediction':p} for i,p in zip(ids,preds)]
    def update(model,optimizer,ids,smoothing,lr):
        model.train();optimizer.zero_grad(set_to_none=True);x=pad([X[i] for i in ids]);y=pad([Y[i] for i in ids]);logits=model(x,y[:,:-1]);loss=nn.functional.cross_entropy(logits.reshape(-1,len(tok)),y[:,1:].reshape(-1),ignore_index=PAD,label_smoothing=smoothing)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
        for group in optimizer.param_groups:group['lr']=lr
        optimizer.step();return float(loss.detach()),float(norm)
    # Diverse train-only rows with distinct French inputs; no validation in the fit gate.
    gate_ids=[];seen=set()
    for i in random.Random(1111).sample(range(len(X)),len(X)):
        if train_src[i] not in seen:gate_ids.append(i);seen.add(train_src[i])
        if len(gate_ids)==32:break
    model=Model(len(tok),dropout=0.).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.001,betas=(.9,.98),eps=1e-9)
    gate_history=[];passed=False
    for step in range(1,801):
        loss,_=update(model,optimizer,gate_ids,0.,.001*min(1.,step/50))
        if step%50==0:
            score,preds=evaluate(model,gate_ids,train=True);entry={'phase':'overfit_gate','step':step,'loss':loss,'metrics':score,'seconds':time.time()-started};gate_history.append(entry);save('progress.json',entry);save('gate_history.json',gate_history);save('gate_predictions.json',preds);print(json.dumps(entry),flush=True)
            if score['all']['exact']>=.95:passed=True;break
    save('GATE.json',{'passed':passed,'required_exact':.95,'examples':32,'result':gate_history[-1]})
    if not passed:save('RESULT.json',{'status':'gate_failed','reason':'Do not launch full training until learnability failure is diagnosed'});return
    # Reset weights after gate; full run starts from a fresh initialization.
    del optimizer,model;torch.cuda.empty_cache();torch.manual_seed(1111);torch.cuda.manual_seed_all(1111)
    model=Model(len(tok),dropout=.2).cuda();optimizer=torch.optim.Adam(model.parameters(),lr=.0003,betas=(.9,.98),eps=1e-9)
    history=[];best=float('inf');bad=0;step=0
    for epoch in range(1,args.epochs+1):
        order=list(range(len(X)));random.shuffle(order);losses=[]
        for start in range(0,len(order),32):
            step+=1;lr=.0003*min(step/200,(200/max(step,200))**.5);loss,norm=update(model,optimizer,order[start:start+32],.2,lr);losses.append(loss)
            if step%50==0:
                progress={'phase':'training','epoch':epoch,'step':step,'loss':loss,'lr':lr,'seconds':time.time()-started};save('progress.json',progress);print(json.dumps(progress),flush=True)
        if epoch%2==0 or epoch==1:
            score,preds=evaluate(model,list(range(len(VX))));train_score,_=evaluate(model,gate_ids,train=True)
            entry={'phase':'evaluation','epoch':epoch,'step':step,'mean_loss':sum(losses)/len(losses),'validation':score,'train_sample':train_score,'seconds':time.time()-started};history.append(entry);save('history.json',history);save('progress.json',entry);print(json.dumps(entry),flush=True)
            eligible=score['cap_hits']==0
            entry['selection_eligible']=eligible
            if eligible and score['all']['wer']<best:
                best=score['all']['wer'];bad=0;checkpoint('best.pt',model);save('BEST.json',entry);save('validation_predictions.json',preds)
            else:bad+=1
            if epoch>=20 and bad>=12:break
    checkpoint('last.pt',model)
    if not (out/'best.pt').exists():
        save('RESULT.json',{'status':'no_eligible_checkpoint','reason':'Every evaluation had unfinished generations'});return
    model.load_state_dict(torch.load(out/'best.pt',weights_only=True));final,preds=evaluate(model,list(range(len(VX))))
    save('validation_predictions.json',preds);save('RESULT.json',{'status':'complete','best':json.loads((out/'BEST.json').read_text()),'validation':final,'seconds':time.time()-started,'linguistically_validated':False});save('progress.json',{'phase':'complete','seconds':time.time()-started});print('COMPLETE',json.dumps(final),flush=True)

if __name__=='__main__':main()
