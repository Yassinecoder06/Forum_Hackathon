"""Actual CPU conditional GAN training, validation selection, ONE heldout pass."""
import argparse,json,random,re,time
from collections import defaultdict
from pathlib import Path
import sys
if __package__ in (None,''): sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from torch import nn
from training.transitions.common import ROOT,ARTIFACTS,FEATURES,CLAIM,sha,dump,interpolate,errors
from training.transitions.models import Generator,Discriminator

RUN='transition-intrasign-v1'
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default=RUN);args=parser.parse_args()
    if not re.fullmatch(r'transition-[a-z0-9-]+',args.run):raise ValueError('Run name must start transition- and contain only lowercase letters/numbers/hyphens')
    torch.set_num_threads(4);torch.manual_seed(42);np.random.seed(42);random.seed(42)
    manifest=json.loads((ROOT/'research/TRANSITION_DATA_MANIFEST.json').read_text())
    out=ROOT/'research/runs'/args.run;out.mkdir(parents=True,exist_ok=False)
    ckpt=ARTIFACTS/(args.run+'.pt')
    # Test NPZ is not opened before training and validation selection finishes.
    x={s:np.load(ARTIFACTS/(s+'.npz'))['windows'] for s in ['train','val']}
    for s in x:assert sha(ARTIFACTS/(s+'.npz'))==manifest['files'][s+'.npz']
    mean=np.asarray(manifest['normalization']['mean'],dtype=np.float32);std=np.asarray(manifest['normalization']['std'],dtype=np.float32)
    def normalize(a):return (a-mean)/std
    train=torch.from_numpy(normalize(x['train']));val=torch.from_numpy(normalize(x['val']))
    g,d=Generator(),Discriminator();gopt=torch.optim.Adam(g.parameters(),lr=2e-4,betas=(.5,.999));dopt=torch.optim.Adam(d.parameters(),lr=2e-4,betas=(.5,.999))
    bce=nn.BCEWithLogitsLoss();l1=nn.L1Loss()
    plan=dict(kind=CLAIM,seed=42,device='cpu',max_epochs=50,patience=8,batch_size=64,lr=2e-4,adam_betas=[.5,.999],lambda_l1=1000.,noise_dimensions=16,eval_noise='all zeros; deterministic',generator_hidden_widths=[256,256,128],generator='Linear/BN/ReLU, Dropout.4after first2hiddenlayers; final100linear; additive linear-interpolation residual shortcut and zero final initialization are engineering additions',discriminator_hidden_widths=[256,128,64],discriminator='Conditional endpoint20+target100 ->MLP, LeakyReLU.2/Dropout.3after each hidden; logit output',loss_g='BCEWithLogits(D(condition,G),1)+1000*L1(G,target)',loss_d='BCEWithLogits(D(condition,target),1)+BCEWithLogits(D(condition,G.detach),0)',selection='Minimum validation normalized RMSE; patience8; trained checkpoints only, not epoch0',normalization=manifest['normalization'],dataset_manifest_sha256=sha(ROOT/'research/TRANSITION_DATA_MANIFEST.json'),generator_parameters=sum(p.numel() for p in g.parameters()),discriminator_parameters=sum(p.numel() for p in d.parameters()),paper_scope='Inspected actual supplied PDFp4section3.5; widths, noise, residual shortcut, conditioning and optimization choices not fully specified by paper; NOT exact reproduction')
    if not (ROOT/'research/TRANSITION_TRAINING_PLAN.json').exists():dump(ROOT/'research/TRANSITION_TRAINING_PLAN.json',plan)
    dump(out/'plan.json',plan)
    endpoints_val=val[:,[0,11]].numpy();target_val=val[:,1:11].numpy()
    baseline_val={m:errors(interpolate(endpoints_val,m),target_val,endpoints_val) for m in ['linear','quintic']}
    selected_baseline=min(baseline_val,key=lambda k:baseline_val[k]['rmse'])
    start=time.monotonic();history=[];best=float('inf');best_epoch=None;stale=0
    for epoch in range(1,51):
        g.train();d.train();order=torch.randperm(len(train));losses=[]
        for indexes in order.split(64):
            batch=train[indexes];endpoints=batch[:,[0,11]];target=batch[:,1:11];noise=torch.randn(len(batch),16)
            dopt.zero_grad();fake=g(endpoints,noise)
            real_logit=d(endpoints,target);fake_logit=d(endpoints,fake.detach())
            dl=bce(real_logit,torch.ones_like(real_logit))+bce(fake_logit,torch.zeros_like(fake_logit));dl.backward();dopt.step()
            for p in d.parameters():p.requires_grad_(False)
            gopt.zero_grad();fake=g(endpoints,torch.randn(len(batch),16));logit=d(endpoints,fake)
            adv=bce(logit,torch.ones_like(logit));recon=l1(fake,target);gl=adv+1000*recon
            if not torch.isfinite(gl+dl):raise ValueError('Nonfinite adversarial training loss')
            gl.backward();gopt.step()
            for p in d.parameters():p.requires_grad_(True)
            losses.append([gl.item(),adv.item(),recon.item(),dl.item()])
        g.eval()
        with torch.no_grad():pred=g(val[:,[0,11]]).numpy()
        score=errors(pred,target_val,endpoints_val);avg=np.mean(losses,axis=0)
        row=dict(epoch=epoch,generator_loss=float(avg[0]),adversarial_loss=float(avg[1]),l1_loss=float(avg[2]),discriminator_loss=float(avg[3]),val_rmse=score['rmse'],val_mae=score['mae'],seconds=time.monotonic()-start)
        history.append(row);print(json.dumps(row),flush=True)
        if score['rmse']<best:
            best=score['rmse'];best_epoch=epoch;stale=0
            torch.save(dict(generator_state=g.state_dict(),normalization=manifest['normalization'],feature_order=FEATURES,kind=CLAIM,plan=plan,best_epoch=epoch),ckpt)
        else:stale+=1
        dump(out/'loss_history.json',history)
        if stale>=8:break
    training_seconds=time.monotonic()-start
    state=torch.load(ckpt,map_location='cpu',weights_only=True);g.load_state_dict(state['generator_state']);g.eval()
    dump(out/'selection_complete.json',dict(best_epoch=best_epoch,epochs=len(history),best_validation_rmse=best,validation_baselines=baseline_val,selected_deterministic_baseline=selected_baseline,test_opened=False))
    assert sha(ARTIFACTS/'test.npz')==manifest['files']['test.npz']
    test=np.load(ARTIFACTS/'test.npz')['windows'];norm_test=normalize(test);endpoints=norm_test[:,[0,11]];target=norm_test[:,1:11]
    with torch.no_grad():gan=g(torch.from_numpy(endpoints)).numpy()
    predictions={m:interpolate(endpoints,m) for m in ['linear','quintic']};predictions['gan']=gan
    metrics={};perwindow=[]
    for method,pred in predictions.items():
        raw=pred*std+mean;metrics[method]=dict(normalized=errors(pred,target,endpoints),native_rig_xy=errors(raw,test[:,1:11],test[:,[0,11]]))
        group_values=defaultdict(list)
        for i,meta in enumerate(manifest['windows']['test']):group_values[meta['group_id']].append(float(np.mean((pred[i]-target[i])**2)))
        metrics[method]['macro_source_clip_rmse']=float(np.mean([np.sqrt(np.mean(v)) for v in group_values.values()]))
    for i,meta in enumerate(manifest['windows']['test']):
        perwindow.append(dict(**meta,kind=CLAIM,feature_order=FEATURES,start=test[i,0].tolist(),end=test[i,11].tolist(),target_native_xy=test[i,1:11].tolist(),predictions_native_xy={m:(p[i]*std+mean).tolist() for m,p in predictions.items()},normalized_errors={m:errors(p[i],target[i],endpoints[i]) for m,p in predictions.items()}))
    (out/'test_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in perwindow))
    gan_worse=metrics['gan']['normalized']['rmse']>metrics[selected_baseline]['normalized']['rmse']
    recommendation=selected_baseline if gan_worse or best>=baseline_val[selected_baseline]['rmse'] else 'gan'
    report=dict(kind=CLAIM,device='cpu',cuda_allocated_peak_mib=0.,cuda_note='No CUDA tensors/context required; existing Qwen server untouched',training_seconds=training_seconds,total_seconds=time.monotonic()-start,counts=manifest['window_counts'],eligible_source_groups=manifest['eligible_group_counts'],epochs=len(history),best_epoch=best_epoch,validation_rmse=best,validation_baselines=baseline_val,selected_deterministic_baseline=selected_baseline,metrics=metrics,recommendation=recommendation,gan_worse_than_selected_baseline=gan_worse,checkpoint=str(ckpt),checkpoint_sha256=sha(ckpt),paper_score_comparison='NOT COMPARABLE: different captured within-sign windows, normalization, native fps, source and split',test_evaluation_passes=1,plan=plan,limitations=manifest['limitations'])
    dump(out/'metrics.json',report)
    sample=x['val'][0];dump(out/'example_endpoints.json',dict(start=sample[0].tolist(),end=sample[11].tolist(),feature_order=FEATURES,origin='Validation clip endpoint example, not test'))
    print(json.dumps({k:report[k] for k in ['epochs','best_epoch','training_seconds','total_seconds','recommendation','gan_worse_than_selected_baseline','metrics']},indent=2),flush=True)
if __name__=='__main__':main()
