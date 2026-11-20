"""Matched-window baselines and GAN ablations with resumable optimizer state."""
import argparse,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
from src.motion.inbetween import Generator,Critic,interpolate,losses,measure,visible_endpoint_interpolation

def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--epochs',type=int,default=40);p.add_argument('--resume',action='store_true');p.add_argument('--visible-endpoints',action='store_true');a=p.parse_args()
 torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu';a.out.mkdir(parents=True,exist_ok=a.resume);beg=time.time()
 manifest=json.loads((a.data/'manifest.json').read_text());assert manifest['sign_language']=='fsl' and manifest['fps']==24
 for s in ('train','val','test'):
  assert manifest['groups'][s]
  for other in ('train','val','test'):
   if other!=s:assert not set(manifest['groups'][s])&set(manifest['groups'][other])
 # Test labels withheld until all three model selections are locked.
 def load(s):
  file=a.data/(s+'.npz');assert hashlib.sha256(file.read_bytes()).hexdigest()==manifest['sha256'][s]
  with np.load(file,allow_pickle=False) as f:return torch.tensor(f['xy'],device=device),torch.tensor(f['mask'],device=device)
 train,tm=load('train');val,vm=load('val');models={};histories={}
 def savejson(name,value):(a.out/name).write_text(json.dumps(value,indent=2))
 for variant in ('reconstruction','adversarial','full'):
  torch.manual_seed(71);np.random.seed(71);random.seed(71);g=Generator(visible_endpoints=a.visible_endpoints).to(device);d=Critic(visible_endpoints=a.visible_endpoints).to(device);go=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999));do=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(.5,.999));best=float('inf');stale=0;history=[];start=1
  resume=a.out/(variant+'-last.pt')
  if a.resume and resume.exists():
   state=torch.load(resume,map_location=device,weights_only=True)
   if state.get('visible_endpoints',False)!=a.visible_endpoints:raise ValueError('Checkpoint conditioning differs from requested run')
   g.load_state_dict(state['g']);d.load_state_dict(state['d']);go.load_state_dict(state['go']);do.load_state_dict(state['do']);best=state['best'];stale=state['stale'];start=state['epoch']+1;history=state['history'];torch.set_rng_state(state['rng'].cpu())
   if device=='cuda':torch.cuda.set_rng_state(state['cuda_rng'].cpu())
  for epoch in range(start,a.epochs+1):
   g.train();d.train();batchloss=[]
   for ids in torch.randperm(len(train),device=device).split(16):
    x,m=train[ids],tm[ids]
    if variant!='reconstruction':
     do.zero_grad();fake=g(x,m).detach();dl=torch.relu(1-d(x,m,x[:,4:12])).mean()+torch.relu(1+d(x,m,fake)).mean();dl.backward();do.step()
    for q in d.parameters():q.requires_grad_(False)
    go.zero_grad();pred=g(x,m);ls=losses(pred,x,m);loss=ls['position']
    if variant!='reconstruction':loss=loss-.01*d(x,m,pred).mean()
    if variant=='full':loss=loss+.5*ls['velocity']+.1*ls['acceleration']+.1*ls['bone']
    if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
    loss.backward();torch.nn.utils.clip_grad_norm_(g.parameters(),1,error_if_nonfinite=True);go.step();batchloss.append(float(loss.detach()))
    for q in d.parameters():q.requires_grad_(True)
   g.eval()
   with torch.no_grad():score=measure(g(val,vm),val,vm)
   row={'epoch':epoch,'loss':float(np.mean(batchloss)),'validation':score};history.append(row)
   if score['mpjpe_2d']<best:best=score['mpjpe_2d'];stale=0;torch.save(g.state_dict(),a.out/(variant+'-best.pt'))
   else:stale+=1
   state={'visible_endpoints':a.visible_endpoints,'g':g.state_dict(),'d':d.state_dict(),'go':go.state_dict(),'do':do.state_dict(),'best':best,'stale':stale,'epoch':epoch,'history':history,'rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else torch.get_rng_state()};torch.save(state,resume)
   savejson('progress.json',{'variant':variant,**row});print(variant,epoch,score['mpjpe_2d'],flush=True)
   if stale>=8:break
  g.load_state_dict(torch.load(a.out/(variant+'-best.pt'),map_location=device,weights_only=True));g.eval();models[variant]=g;histories[variant]=history
 savejson('selection.json',histories);test,mask=load('test');results={}
 with torch.no_grad():
  predictions={name:interpolate(test,method=name) for name in ('linear','hermite')}
  predictions['visible_endpoints']=visible_endpoint_interpolation(test,mask)
  predictions.update({k:g(test,mask) for k,g in models.items()})
  for k,pred in predictions.items():results[k]=measure(pred,test,mask)
 np.savez_compressed(a.out/'predictions.npz',target=test.cpu().numpy(),mask=mask.cpu().numpy(),**{k:v.cpu().numpy() for k,v in predictions.items()})
 report={'visible_endpoint_conditioning':a.visible_endpoints,'task':'2D continuous-signing masked-motion inbetweening; not lexical coarticulation','sign_language':'fsl','seed':71,'device':device,'peak_cuda_mb':torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else 0,'seconds':time.time()-beg,'counts':{s:manifest['counts'][s] for s in ('train','val','test')},'metrics':results,'signer_disjoint':'not established; source videos disjoint','test_tuning':False,'data_manifest':manifest}
 savejson('RESULTS.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
