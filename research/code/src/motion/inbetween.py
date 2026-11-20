"""Confidence-aware 2D inbetweening. No lexical-boundary/coarticulation claim."""
import torch
from torch import nn
from src.data.signverse import BONES

def visible_endpoint_interpolation(x, mask, context=4, gap=8):
    """Use the nearest visible sample on each side of the masked interval."""
    frame = torch.arange(x.shape[1], device=x.device)[None, :, None]
    left = torch.where(mask & (frame < context), frame, -1).amax(dim=1)
    right = torch.where(mask & (frame >= context + gap), frame, x.shape[1]).amin(dim=1)

    def positions(indices):
        indices = indices.clamp(0, x.shape[1] - 1)
        return x.gather(1, indices[:, None, :, None].expand(-1, 1, -1, x.shape[-1]))

    a, b = positions(left), positions(right)
    has_left, has_right = (left >= 0)[:, None, :, None], (right < x.shape[1])[:, None, :, None]
    t = torch.arange(context, context + gap, device=x.device, dtype=x.dtype)[None, :, None, None]
    ratio = (t - left[:, None, :, None]) / (right - left).clamp_min(1)[:, None, :, None]
    line = a + ratio * (b - a)
    return torch.where(has_left & has_right, line,
                       torch.where(has_left, a, torch.where(has_right, b, 0)))

def wrist_relative_interpolation(x, mask, context=4, gap=8):
    """Interpolate finger offsets from the wrist of each DWPose hand."""
    prediction = visible_endpoint_interpolation(x, mask, context, gap)
    for wrist in (18, 39):
        fingers = slice(wrist + 1, wrist + 21)
        offsets = x[:, :, fingers] - x[:, :, wrist:wrist + 1]
        observed = mask[:, :, fingers] & mask[:, :, wrist:wrist + 1]
        support = observed[:, :context].any(dim=1) | observed[:, context + gap:].any(dim=1)
        relative = visible_endpoint_interpolation(offsets, observed, context, gap)
        prediction[:, :, fingers] = torch.where(
            support[:, None, :, None],
            prediction[:, :, wrist:wrist + 1] + relative,
            prediction[:, :, fingers],
        )
    return prediction

def interpolate(x,context=4,gap=8,method='linear'):
    a,b=x[:,context-1],x[:,context+gap];t=torch.arange(1,gap+1,device=x.device,dtype=x.dtype)[None,:,None,None]/(gap+1)
    if method=='linear':return a[:,None]*(1-t)+b[:,None]*t
    if method!='hermite':raise ValueError(method)
    ma=(a-x[:,context-2])*(gap+1);mb=(x[:,context+gap+1]-b)*(gap+1)
    return (2*t**3-3*t**2+1)*a[:,None]+(t**3-2*t**2+t)*ma[:,None]+(-2*t**3+3*t**2)*b[:,None]+(t**3-t**2)*mb[:,None]

def condition(x, mask, context=4, gap=8, *, visible_endpoints=False):
    visible = mask.clone()
    visible[:, context:context + gap] = False
    base = x.clone()
    if visible_endpoints:
        base = base.masked_fill(~visible.unsqueeze(-1), 0)
        base[:, context:context + gap] = visible_endpoint_interpolation(x, visible, context, gap)
    else:
        base[:, context:context + gap] = interpolate(x, context, gap)
    return base, visible

class Generator(nn.Module):
    def __init__(self,joints=128,context=4,gap=8,*,visible_endpoints=False):
        super().__init__();self.context=context;self.gap=gap;self.visible_endpoints=visible_endpoints
        self.net=nn.Sequential(nn.Conv1d(joints*3,128,3,padding=1),nn.GELU(),nn.Conv1d(128,128,3,padding=1),nn.GELU(),nn.Conv1d(128,joints*2,3,padding=1))
        nn.init.zeros_(self.net[-1].weight);nn.init.zeros_(self.net[-1].bias)
    def forward(self,x,mask):
        base,visible=condition(x,mask,self.context,self.gap,visible_endpoints=self.visible_endpoints);features=torch.cat([base.flatten(2),visible.float()],-1).transpose(1,2)
        residual=self.net(features).transpose(1,2).reshape_as(x)
        return base[:,self.context:self.context+self.gap]+residual[:,self.context:self.context+self.gap]

class Critic(nn.Module):
    def __init__(self,joints=128,*,visible_endpoints=False):
        super().__init__();self.visible_endpoints=visible_endpoints;sn=nn.utils.parametrizations.spectral_norm
        self.net=nn.Sequential(sn(nn.Conv1d(joints*5,128,3,padding=1)),nn.LeakyReLU(.2),sn(nn.Conv1d(128,64,3,padding=1)),nn.LeakyReLU(.2),sn(nn.Conv1d(64,1,3,padding=1)))
    def forward(self,x,mask,candidate):
        base,visible=condition(x,mask,visible_endpoints=self.visible_endpoints)
        joined=base.clone()
        joined[:,4:12]=candidate.masked_fill(~mask[:,4:12].unsqueeze(-1), 0) if self.visible_endpoints else candidate
        return self.net(torch.cat([base.flatten(2),visible.float(),joined.flatten(2)],-1).transpose(1,2))

def masked_mean(value,mask):
    while mask.ndim<value.ndim:mask=mask.unsqueeze(-1)
    expanded=mask.expand_as(value);return (value*expanded).sum()/expanded.sum().clamp_min(1)

def losses(pred,x,mask):
    joined=x.clone();joined[:,4:12]=pred;truth=x[:,4:12];m=mask[:,4:12]
    position=masked_mean((pred-truth).abs(),m)
    vel=masked_mean((torch.diff(joined,dim=1)-torch.diff(x,dim=1)).abs(),mask[:,1:]&mask[:,:-1])
    accel=masked_mean((torch.diff(joined,n=2,dim=1)-torch.diff(x,n=2,dim=1)).abs(),mask[:,2:]&mask[:,1:-1]&mask[:,:-2])
    a,b=zip(*BONES);bone=masked_mean((torch.linalg.vector_norm(pred[:,:,a]-pred[:,:,b],dim=-1)-torch.linalg.vector_norm(truth[:,:,a]-truth[:,:,b],dim=-1)).abs(),m[:,:,a]&m[:,:,b])
    return dict(position=position,velocity=vel,acceleration=accel,bone=bone)

@torch.no_grad()
def measure(pred,x,mask):
    target=x[:,4:12];m=mask[:,4:12];err=torch.linalg.vector_norm(pred-target,dim=-1)
    wrist=[4,7];tips=[22,26,30,34,38,43,47,51,55,59];hands=list(range(18,60))
    result={'mpjpe_2d':float(masked_mean(err,m)),'wrist_error':float(masked_mean(err[:,:,wrist],m[:,:,wrist])),'fingertip_error':float(masked_mean(err[:,:,tips],m[:,:,tips])),'hand_joint_error':float(masked_mean(err[:,:,hands],m[:,:,hands])),'invalid_fraction':float((~m).float().mean())}
    joined=x.clone();joined[:,4:12]=pred
    for n,name in [(1,'velocity_error'),(2,'acceleration_error'),(3,'jerk_error')]:
        valid=mask[:,:mask.shape[1]-n].clone()
        for k in range(1,n+1):valid &= mask[:,k:k+valid.shape[1]]
        diff=(torch.diff(joined,n=n,dim=1)-torch.diff(x,n=n,dim=1))*24**n
        result[name]=float(masked_mean(torch.linalg.vector_norm(diff,dim=-1),valid))
    v=(torch.diff(joined,dim=1)-torch.diff(x,dim=1))[:,[3,11]]*24
    vm=(mask[:,1:]&mask[:,:-1])[:,[3,11]];result['boundary_velocity_error']=float(masked_mean(torch.linalg.vector_norm(v,dim=-1),vm))
    result['projected_bone_error']=float(losses(pred,x,mask)['bone'])
    return result
