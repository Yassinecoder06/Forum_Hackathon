import io,json,pickle,tarfile
from pathlib import Path
import numpy as np
import pytest
import torch
from src.data.signverse import read_npz,preprocess,select_track,safe_members,validate_caption
from src.motion.inbetween import Generator,interpolate,losses,measure

def fixture(n=20):
 x=np.zeros((n,128,3),np.float32);x[:,:,2]=.9;x[:,:,0]=100;x[:,:,1]=200;x[:,2,0]=150;x[:,5,0]=50
 return x

def test_language_gate():
 for lang in ['sfb','ssr','FSL',None]:
  with pytest.raises(ValueError):validate_caption({'video_id':'v','sign_language':lang},{'video_id':'v','sign_language':'fsl'})
 validate_caption({'video_id':'v','sign_language':'fsl'},{'video_id':'v','sign_language':'fsl'})

def test_pose_masks_and_depth():
 x=fixture();x[3,18]=[np.nan,10,1];x[4,19,2]=.1
 a,q=preprocess(x,np.arange(len(x)));assert a['normalized_xy'].shape==(20,128,2);assert not a['mask'][3,18] and not a['mask'][4,19];assert np.isfinite(a['normalized_xy']).all();assert np.isnan(a['original_xyc'][3,18,0])

def test_safe_npz_and_pickle(tmp_path):
 x={'frames':np.array([{'person_0':{'body':np.zeros((18,3))}}],dtype=object)};p=tmp_path/'x.npz';np.savez(p,**x);assert read_npz(p)['frames'].shape==(1,)
 class Attack:
  def __reduce__(self):return (eval,('1+1',))
 np.savez(p,frames=np.array([Attack()],dtype=object))
 with pytest.raises(ValueError):read_npz(p)

def test_tar_traversal(tmp_path):
 p=tmp_path/'bad.tar'
 with tarfile.open(p,'w') as t:
  m=tarfile.TarInfo('../escape');m.size=1;t.addfile(m,io.BytesIO(b'x'))
 with pytest.raises(ValueError):list(safe_members(p,{'v'}))

def test_multiple_people():
 raw=fixture(1)[0];person={'body':raw[:18],'left_hand':raw[18:39],'right_hand':raw[39:60],'face':raw[60:]};bad={'body':np.zeros((18,3))}
 x,ids,amb=select_track([{'person_0':bad,'person_1':person}]);assert ids==['person_1'] and not amb[0]

def test_no_hidden_target_leakage_and_linear():
 torch.manual_seed(1);x=torch.randn(2,16,128,2);m=torch.ones(2,16,128,dtype=torch.bool);g=Generator().eval();changed=x.clone();changed[:,4:12]+=1000
 with torch.no_grad():assert torch.allclose(g(x,m),g(changed,m))
 line=torch.arange(16).float()[None,:,None,None].expand(2,16,128,2)
 for method in ['linear','hermite']:assert torch.allclose(interpolate(line,method=method),line[:,4:12],atol=1e-5)

def test_masked_losses_finite_and_nonzero_gradient():
 x=torch.rand(2,16,128,2);m=torch.ones(2,16,128,dtype=torch.bool);m[:,:,19]=False;p=torch.rand(2,8,128,2,requires_grad=True);l=losses(p,x,m);sum(l.values()).backward();assert torch.isfinite(p.grad).all();assert p.grad.abs().sum()>0
 assert measure(x[:,4:12],x,m)['mpjpe_2d']==0
